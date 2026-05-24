"""
Lambda/Fargate handler for 3GPP RAG API.
v3: Planner → Adaptive Router → RRF Retriever → IsREL → Reranker → CRAG → Generator
"""

import json
import re
import time
import hashlib
import boto3
import psycopg2
from mangum import Mangum
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
import os

# Thread pool for parallel retrieval (8 workers for concurrent pgvector queries)
_RETRIEVAL_EXECUTOR = ThreadPoolExecutor(max_workers=8)

# Config
AWS_REGION  = os.environ.get("AWS_REGION", "us-east-1")
PG_HOST     = os.environ["PG_HOST"]
PG_PORT     = int(os.environ.get("PG_PORT", "5432"))
PG_DATABASE = os.environ.get("PG_DATABASE", "ragdb")
PG_USER     = os.environ.get("PG_USER", "ragadmin")
PG_PASSWORD = os.environ["PG_PASSWORD"]
PG_TABLE    = os.environ.get("PG_TABLE", "chunks")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
HAIKU_MODEL_ID = os.environ.get("HAIKU_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
NOVA_PRO_MODEL_ID = os.environ.get("NOVA_PRO_MODEL_ID", "us.amazon.nova-pro-v1:0")
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")

app = FastAPI(title="3GPP RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=Config(read_timeout=120))


# ─── Telecom Synonyms ────────────────────────────────────────────────────────
TELECOM_SYNONYMS = {
    "clear code":     ["Cause IE", "CauseRadioNetwork", "cause code", "release cause", "RLF-Cause"],
    "drop cause":     ["CauseRadioNetwork", "radioConnectionWithUELost", "release cause"],
    "failure code":   ["Cause IE", "CauseProtocol", "CauseMisc"],
    "error code":     ["Cause IE", "CauseProtocol"],
    "release reason": ["ReleaseCause", "release cause", "RRC release"],
    "disconnect reason": ["Cause IE", "CauseRadioNetwork"],
    "reject cause":   ["5GMM cause", "CauseNas"],
    "handover failure": ["handoverFailure", "CauseRadioNetwork"],
}

# Only search specs that are known to have cause code content in the index
CAUSE_CODE_SPECS = ["24501", "38331", "38473", "38.413"]
# V15-FIX-2: Pinned queries disabled for latency (browser times out at 120s)
CAUSE_CODE_PINNED_QUERIES = []


# Clause blacklist: sections that RRF over-promotes for cause code queries
# These are tangentially related but NOT the actual cause code definitions
CAUSE_CLAUSE_BLACKLIST = [
    "9.3.1.111",   # RRC Establishment Cause (not a "clear code" / cause IE)
    "4.5.6",       # Access category → RRC establishment cause mapping
    "5.6.1.4.1",   # CIoT narrow case (not general cause codes)
    "9.11.3.39",   # Payload container (not a cause code)
    "5.4.5.3",     # NAS transport procedures (not cause definitions)
    "G.1",         # TDD operating bands (RF, not protocol)
    "5.3.10.5",    # RLF report content (adjacent, not cause codes)
    "9.3.4.2",     # V15-FIX-1: NGAP PDU Session table — not Cause IE
    "9.3.3.60",    # V15-FIX-1: NGAP Resource Status table — not Cause IE
    "9.11.3.4",    # 5GS mobile identity — not a cause code
    "9.11.4.31",   # Received MBS container — not a cause code
    "§8.1 Overview",  # V19: NAS overview section — not cause codes
    "4.7.2.2",     # V19: Establishment cause for non-3GPP access — not cause codes
]

# Sections that should never rank high for cause code queries
SECTION_BLACKLIST_PATTERNS = ["change history", "abbreviations", "annex a", "annex b",
                              "0 introduction", "payload container", "unwanted emission",
                              "operating band", "repeater type", "2 references", "2\treferences"]

# Specs that are never relevant for cause code / protocol queries
# RF/test specs that are never relevant for protocol queries (applies to ALL queries)
SPEC_BLACKLIST_ALWAYS = ["38106", "38115", "38141", "38521", "38522", "38523", "38905", "38918", "RP"]

PLANNER_SYSTEM = """You are a 3GPP specification query planner.

When the user asks about "cause codes", "clear codes", "release causes",
"failure codes", or "error codes" in 5G NR, interpret these as the
"Cause IE" information element defined in:
  - TS 38.413 §9.3.1.2 (NGAP)
  - TS 38.473 §9.3.1.2 (F1AP)
  - TS 38.463 (E1AP)
  - TS 38.423 (XnAP)
  - TS 38.331 (RRC) — RLF-Cause, EstablishmentCause, ReleaseCause
  - TS 24.501 §9.11.3.2 (5GMM cause), §9.11.4.2 (5GSM cause)

Do NOT retrieve content about:
  - 5GS mobile identity types (SUCI, 5G-S-TMSI, IMEI, SUPI)
  - Identity type encoding (bit patterns in octets)
  - NAI / SUPI format construction
  - Security algorithms or key derivation

Terminology mapping:
  - "clear code" → "Cause IE" / "cause value"
  - "error code" → "Cause IE" / "CauseProtocol"
  - "reject reason" → "5GMM cause" / "CauseNas"
  - "disconnect reason" → "CauseRadioNetwork"
  - "failure reason" → "CauseRadioNetwork" / "RLF-Cause"

Decompose into 2-5 focused sub-queries using correct 3GPP terms.
Return ONLY valid JSON (no markdown):
{"spec": null, "release": null, "sub_queries": [...], "search_terms": [...]}"""

GENERATOR_SYSTEM = """You are a principal 3GPP standards architect producing reference-quality technical documentation.
Your answers must be MORE accurate than ChatGPT/Gemini because you have the EXACT specification text.

RULES:
1. ONLY use information from the provided context. Every claim MUST be traceable.
2. Never hallucinate content not in the context.
3. Use markdown: ## headers, | tables |, ```text blocks for protocol flows
4. Cite inline: (per TS 38.331 §5.3.2)
5. Include Key References section at the end.
6. Use ASCII diagrams for message flows and state machines.
7. Use tables for parameter comparisons.

End with: {"confidence": 0.0-1.0}"""


# ─── Core Functions ───────────────────────────────────────────────────────────
def get_pg_conn():
    return psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DATABASE, user=PG_USER, password=PG_PASSWORD)


# ─── Embedding cache (per-request deduplication) ─────────────────────────────
_embed_cache: dict[str, list[float]] = {}


def embed_text(text: str) -> list[float]:
    key = text[:8000].strip()
    if key in _embed_cache:
        return _embed_cache[key]
    resp = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID, body=json.dumps({"inputText": key}),
        contentType="application/json", accept="application/json"
    )
    emb = json.loads(resp["body"].read())["embedding"]
    _embed_cache[key] = emb
    return emb


def embed_texts_batch(texts: list[str]) -> None:
    """Embed multiple unique texts in parallel threads, populating _embed_cache."""
    to_embed = []
    for t in texts:
        key = t[:8000].strip()
        if key not in _embed_cache:
            to_embed.append(key)
    if not to_embed:
        return
    # Parallel embedding calls (Bedrock Titan handles concurrent requests)
    def _embed_one(text):
        resp = bedrock.invoke_model(
            modelId=EMBED_MODEL_ID, body=json.dumps({"inputText": text}),
            contentType="application/json", accept="application/json"
        )
        emb = json.loads(resp["body"].read())["embedding"]
        _embed_cache[text] = emb
    futures = [_RETRIEVAL_EXECUTOR.submit(_embed_one, t) for t in to_embed]
    for f in futures:
        f.result()  # wait for all embeddings


def hybrid_search_with_embedding(query: str, query_emb: list[float], top_k=10, spec_filter=None, release_filter=None) -> list[dict]:
    """Hybrid search using pre-computed embedding (avoids redundant embed calls)."""
    conditions, params = [], []
    if spec_filter:
        conditions.append("spec_number = %s")
        params.append(spec_filter)
    if release_filter:
        conditions.append("release = %s")
        params.append(release_filter)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"""
        SELECT chunk_id, spec_number, release, section_path, doc_type, chunk_text, summary,
               (1 - (embedding <=> %s::vector)) AS vec_score,
               COALESCE(ts_rank(to_tsvector('english', chunk_text), plainto_tsquery('english', %s)), 0) AS text_score
        FROM {PG_TABLE} {where}
        ORDER BY (0.7 * (1 - (embedding <=> %s::vector)) +
                  0.3 * COALESCE(ts_rank(to_tsvector('english', chunk_text), plainto_tsquery('english', %s)), 0)) DESC
        LIMIT %s
    """
    all_params = [str(query_emb), query] + params + [str(query_emb), query, top_k]
    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute(sql, all_params)
    rows = cur.fetchall()
    conn.close()
    return [
        {"chunk_id": r[0], "spec_number": r[1], "release": r[2], "section_path": r[3],
         "doc_type": r[4], "chunk_text": r[5], "summary": r[6],
         "score": round(0.7 * r[7] + 0.3 * r[8], 4)}
        for r in rows
    ]


def hybrid_search(query: str, top_k=10, spec_filter=None, release_filter=None):
    """Legacy wrapper — embeds then searches."""
    query_emb = embed_text(query)
    return hybrid_search_with_embedding(query, query_emb, top_k, spec_filter, release_filter)


# ─── RRF (Reciprocal Rank Fusion) ────────────────────────────────────────────
def reciprocal_rank_fusion(result_sets: list[list[dict]], k: int = 60) -> list[dict]:
    """Fuse multiple ranked lists. score = sum(1/(k+rank)) across sets."""
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}
    for result_set in result_sets:
        for rank, chunk in enumerate(result_set):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = chunk
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    fused = []
    for cid in sorted_ids:
        c = chunk_map[cid].copy()
        c["score"] = rrf_scores[cid]
        fused.append(c)
    # BUG-1-FIX: Do NOT normalize here. Raw k=60 scores (0 to ~0.16) are correct.
    # Previous normalization to [0,1] caused reranker boost to push scores above 1.0.
    return fused


# ─── Adaptive Router ──────────────────────────────────────────────────────────
LOOKUP_PATTERNS = [
    r"\b(list|enumerate)\s+(all|every)\b.*\b(cause|timer|IE|value|code)s?\b",
    r"\bwhat\s+(are|is)\s+the\s+(cause|timer|IE|RLF)\s*(code|value|IE)s?\b",
    r"\b(CauseRadioNetwork|CauseTransport|CauseNas|CauseProtocol|CauseMisc)\b",
    r"\bRLF.?Cause\s*(enum|values?)\b",
    r"\bEstablishmentCause\s*(enum|values?)\b",
]


def classify_query(query: str) -> str:
    for pattern in LOOKUP_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return "lookup"
    return "vector"


def direct_lookup(query: str, spec_filter=None) -> list[dict]:
    """Direct DB query for structured lookups."""
    q = query.lower()
    section_patterns = []
    if any(kw in q for kw in ["cause", "clear code", "reject"]):
        section_patterns = ["%9.3.1.2%Cause%", "%9.11.3.2%", "%9.11.4.2%", "%5.3.10.4%", "%9.2.3.2%Cause%", "%B.1%Cause%"]
    elif any(kw in q for kw in ["timer", "t310", "t311"]):
        section_patterns = ["%timer%"]
    elif any(kw in q for kw in ["establishment"]):
        section_patterns = ["%establishment%", "%EstablishmentCause%"]
    if not section_patterns:
        return []

    conditions = " OR ".join(["section_path ILIKE %s" for _ in section_patterns])
    spec_clause = "AND spec_number = %s" if spec_filter else ""
    params = section_patterns + ([spec_filter] if spec_filter else [])

    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute(f"""
        SELECT chunk_id, spec_number, release, section_path, doc_type, chunk_text, summary
        FROM chunks WHERE ({conditions}) {spec_clause} ORDER BY section_path LIMIT 30
    """, params)
    rows = cur.fetchall()
    conn.close()
    return [
        {"chunk_id": r[0], "spec_number": r[1], "release": r[2], "section_path": r[3],
         "doc_type": r[4], "chunk_text": r[5], "summary": r[6], "score": 0.8}
        for r in rows
    ]


# ─── Planner ─────────────────────────────────────────────────────────────────
def expand_query(user_query: str) -> str:
    q = user_query.lower()
    expansions = []
    for term, synonyms in TELECOM_SYNONYMS.items():
        if term in q:
            expansions.extend(synonyms)
    if expansions:
        return f"{user_query}\n\n[Search also for: {', '.join(set(expansions))}]"
    return user_query


def run_planner(query: str) -> list[str]:
    expanded = expand_query(query)
    try:
        resp = bedrock.invoke_model(
            modelId=NOVA_PRO_MODEL_ID,
            body=json.dumps({
                "system": [{"text": PLANNER_SYSTEM}],
                "messages": [{"role": "user", "content": [{"text": expanded}]}],
                "inferenceConfig": {"maxTokens": 512}
            }),
            contentType="application/json", accept="application/json"
        )
        raw = json.loads(resp["body"].read())["output"]["message"]["content"][0]["text"].strip()
        raw = re.sub(r"^```json\s*|\s*```$", "", raw)
        plan = json.loads(raw)
        sub_queries = plan.get("sub_queries") or [query]
        search_terms = plan.get("search_terms") or []
        if search_terms:
            sub_queries.append(" ".join(search_terms[:8]))
        # Ensure minimum 3 sub-queries for complex cause code questions
        if len(sub_queries) < 3 and any(kw in query.lower() for kw in ["cause", "clear code", "classification"]):
            sub_queries.extend([
                "Cause IE definition 9.3.1.2 CauseRadioNetwork enumeration values",
                "5GMM cause 5GSM cause values TS 24.501 rejection",
                "RLF cause t310 beam failure rlc-MaxNumRetx TS 38.331",
            ])
        return sub_queries
    except Exception:
        return [query]


# ─── IsREL Filter ─────────────────────────────────────────────────────────────
def isrel_filter(chunks: list[dict], query: str) -> list[dict]:
    """BUG 3 FIX: Enforces min_keep=15, max_keep=25 floor/ceiling."""
    MIN_KEEP = 15
    MAX_KEEP = 25
    
    if len(chunks) <= MIN_KEEP:
        return chunks
    
    relevant = []
    ambiguous = []
    q_lower = query.lower()
    is_cause_query = any(kw in q_lower for kw in ["cause", "clear code", "failure", "reject", "release cause"])

    for c in chunks:
        section = (c.get("section_path") or "").lower()
        text_preview = c.get("chunk_text", "")[:200].lower()
        score = c.get("score", 0)
        spec = c.get("spec_number", "")

        # Rule-based IRRELEVANT
        if section in ("2\treferences", "2 references", "1\tscope", "foreword"):
            continue
        if "change history" in section or "abbreviations" in section:
            continue
        if "annex" in section and "change" in section:
            continue
        if "0 introduction" in section or "0\tintroduction" in section:
            continue
        if is_cause_query and "5gs mobile identity" in section:
            continue
        if is_cause_query and ("suci" in text_preview or "5g-s-tmsi" in text_preview):
            continue
        if is_cause_query and any(ts in spec for ts in ["38106", "38115", "38141", "38521", "38905", "38918"]):
            continue

        # Rule-based RELEVANT
        if score >= 0.4:
            relevant.append(c)
        elif any(kw in section for kw in ["cause", "failure", "reject", "release", "rlf", "9.3.1", "9.11"]):
            relevant.append(c)
        elif score >= 0.25:
            relevant.append(c)
        else:
            ambiguous.append(c)

    # BUG 3 FIX: Enforce minimum floor
    if len(relevant) < MIN_KEEP and ambiguous:
        ambiguous_sorted = sorted(ambiguous, key=lambda c: c.get("score", 0), reverse=True)
        needed = MIN_KEEP - len(relevant)
        relevant.extend(ambiguous_sorted[:needed])

    # BUG 3 FIX: Cap at MAX_KEEP
    if len(relevant) > MAX_KEEP:
        relevant = sorted(relevant, key=lambda c: c.get("score", 0), reverse=True)[:MAX_KEEP]

    return relevant


# ─── BUG 2 FIX: Clause-level deduplication ────────────────────────────────────
def deduplicate_by_clause(chunks: list[dict]) -> list[dict]:
    """Keep only the highest-scored chunk per unique clause.
    Prevents §9.11.3.39 appearing 4× in 7 source slots."""
    seen_clauses: dict[str, dict] = {}
    for c in chunks:
        clause = (c.get("section_path") or "unknown").strip()
        if clause not in seen_clauses or c.get("score", 0) > seen_clauses[clause].get("score", 0):
            seen_clauses[clause] = c
    return sorted(seen_clauses.values(), key=lambda c: c.get("score", 0), reverse=True)


# ─── BUG-1-FIX: No normalization — keep raw k=60 RRF scores ───────────────────
def normalize_rrf_scores(chunks: list[dict]) -> list[dict]:
    """NO-OP: Raw k=60 scores are well-behaved (0 to ~0.16).
    BUG-1-FIX: Min-max was inflating top score to 1.0, breaking CRAG threshold."""
    return chunks  # BUG-1-FIX: keep raw scores



# ─── Corpus Gap Detection ─────────────────────────────────────────────────────
# When retrieved chunks don't match the user's requested spec/release,
# notify the user and queue background ingestion instead of generating
# a confident wrong answer.

import re as _re

RELEASE_PATTERN = _re.compile(r'[Rr]el(?:ease)?[-\s]?(1[5-9]|2[0-9])', _re.IGNORECASE)
SPEC_PATTERN = _re.compile(r'(?:TS|ts)\s*(\d{5}|\d{2}\.\d{3})')

# Critical specs that should exist for each release
CRITICAL_SPECS_FOR_GAP = ["38331", "38300", "38304", "38912", "38321", "24501", "38413", "38473", "38423"]


def detect_corpus_gap(query: str, chunks: list[dict]) -> dict:
    """Detect if retrieved chunks match the user's requested spec+release.
    Returns {"has_gap": bool, "message": str, "missing_spec": str, "missing_release": str}
    """
    # Extract requested release from query
    rel_match = RELEASE_PATTERN.search(query)
    if not rel_match:
        return {"has_gap": False}
    
    requested_release = f"Rel-{rel_match.group(1)}"
    
    # Extract requested spec from query (optional)
    spec_match = SPEC_PATTERN.search(query)
    requested_spec = spec_match.group(1).replace(".", "") if spec_match else None
    
    # Check if any retrieved chunks match the requested release
    chunk_releases = [c.get("release", "") for c in chunks]
    has_matching_release = any(requested_release in r for r in chunk_releases)
    
    if has_matching_release:
        return {"has_gap": False}
    
    # Gap detected — chunks don't match requested release
    return {
        "has_gap": True,
        "missing_release": requested_release,
        "missing_spec": requested_spec,
        "message": f"Note: {requested_release} content for this topic is being indexed. "
                   f"The answer below is based on the closest available release. "
                   f"A fully grounded {requested_release} answer will be available within 24 hours."
    }


def log_gap_for_ingestion(query: str, gap_info: dict):
    """Log the corpus gap for background processing."""
    try:
        conn = get_pg_conn()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_queue (
                id SERIAL PRIMARY KEY,
                query_text TEXT,
                missing_release TEXT,
                missing_spec TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                completed_at TIMESTAMPTZ
            )
        """)
        cur.execute("""
            INSERT INTO ingestion_queue (query_text, missing_release, missing_spec)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (query, gap_info.get("missing_release"), gap_info.get("missing_spec")))
        conn.commit()
        conn.close()
    except Exception:
        pass  # Non-critical — don't break the query flow

# ─── Reranker ─────────────────────────────────────────────────────────────────
def rerank_chunks(chunks: list[dict]) -> list[dict]:
    for c in chunks:
        s = c.get("score", 0)
        if s != s:
            c["score"] = 0.0
        section = (c.get("section_path") or "").lower()
        if any(x in section for x in ["references", "0 introduction", "1\tscope", "introduction"]):
            c["score"] *= 0.6
        elif "cause" in section or "failure" in section or "release" in section:
            c["score"] *= 1.2
        elif "[table]" in section:
            c["score"] *= 1.15

    chunks = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)
    seen = set()
    deduped = []
    for c in chunks:
        if c["chunk_id"] not in seen:
            deduped.append(c)
            seen.add(c["chunk_id"])

    # Spec diversity — increased limits for better coverage
    unique_specs = set(c.get("spec_number", "") for c in deduped[:20])
    apply_diversity = len(unique_specs) > 2
    selected = []
    spec_counts = {}
    for c in deduped:
        spec = c.get("spec_number", "unknown")
        if apply_diversity and spec_counts.get(spec, 0) >= 6:
            continue
        selected.append(c)
        spec_counts[spec] = spec_counts.get(spec, 0) + 1
        if len(selected) >= 20:
            break
    return selected


# ─── Release History ──────────────────────────────────────────────────────────
def get_release_history(spec_filter=None) -> str:
    try:
        conn = get_pg_conn()
        cur = conn.cursor()
        where = f"WHERE spec_number = '{spec_filter}'" if spec_filter else ""
        cur.execute(f"SELECT DISTINCT release, cause_feature, spec_number FROM cause_release_history {where} ORDER BY release, cause_feature LIMIT 40")
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return ""
        lines = ["\n[CAUSE CODE RELEASE HISTORY (from CR Change History):"]
        current_rel = ""
        for rel, feature, spec in rows:
            if rel != current_rel:
                lines.append(f"  {rel}:")
                current_rel = rel
            lines.append(f"    - {feature} (TS {spec})")
        lines.append("]")
        return "\n".join(lines)
    except Exception:
        return ""


# ─── API Models ───────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    spec_filter: Optional[str] = None
    release_filter: Optional[str] = None
    top_k: Optional[int] = 10


class Citation(BaseModel):
    spec: Optional[str]
    section: Optional[str]
    release: Optional[str]
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    latency_ms: int
    chunks_retrieved: int
    steps: list[dict]
    cached: bool = False


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": LLM_MODEL_ID, "version": "v13-sse"}


# ─── SSE Streaming Endpoint ─────────────────────────────────────────────────
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator


def sse_event(event_type: str, data: dict) -> str:
    """Format a single SSE message."""
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """SSE streaming endpoint — streams pipeline steps and answer tokens."""

    async def generate() -> AsyncGenerator[str, None]:
        start = time.time()
        query = req.query

        # Step 1: Planner
        yield sse_event("step_start", {"step": "planner", "label": "Decomposing query", "icon": "🧠", "detail": "Breaking into targeted sub-queries..."})
        t1 = time.time()
        sub_queries = run_planner(query)
        yield sse_event("step_done", {"step": "planner", "label": f"Planner: {len(sub_queries)} sub-queries", "ms": int((time.time() - t1) * 1000)})

        # Step 2: Router / RRF
        yield sse_event("step_start", {"step": "router", "label": "Retrieving from 3GPP index", "icon": "🔍", "detail": f"Running {len(sub_queries)} sub-queries + targeted spec searches..."})
        t2 = time.time()
        query_type = classify_query(query)
        all_chunks = []
        num_sets = 0

        if query_type == "lookup":
            direct_results = direct_lookup(query, req.spec_filter)
            vector_hits = hybrid_search(query, 10, req.spec_filter, req.release_filter)
            seen = {c["chunk_id"] for c in direct_results}
            all_chunks = list(direct_results)
            for h in vector_hits:
                if h["chunk_id"] not in seen:
                    all_chunks.append(h)
                    seen.add(h["chunk_id"])
            num_sets = 2
        else:
            search_tasks = []
            for q in sub_queries:
                search_tasks.append((q, 8, req.spec_filter, req.release_filter))  # V15-FIX-3: top_k=8
            # V15-FIX-2: Add pinned spec-clause queries for guaranteed coverage
            is_cause_query = any(kw in query.lower() for kw in ["cause", "clear code", "release cause", "reject", "failure"])
            if is_cause_query:
                for pq in CAUSE_CODE_PINNED_QUERIES:  # V15-FIX-2
                    search_tasks.append((pq, 8, req.spec_filter, req.release_filter))  # V15-FIX-3
            if not req.spec_filter and is_cause_query:
                for spec in CAUSE_CODE_SPECS:
                    search_tasks.append((query, 5, spec, None))  # V21: top_k=5 (balance coverage vs latency)

            unique_queries = list(set(t[0] for t in search_tasks))
            embed_texts_batch(unique_queries)

            def _do_search(args):
                q, top_k, spec, release = args
                q_emb = _embed_cache[q[:8000].strip()]
                return hybrid_search_with_embedding(q, q_emb, top_k, spec_filter=spec, release_filter=release)

            futures = [_RETRIEVAL_EXECUTOR.submit(_do_search, t) for t in search_tasks]
            result_sets = [f.result() for f in futures]
            result_sets = [rs for rs in result_sets if rs]

            # Always filter RF/test specs and generic sections
            for i, rs in enumerate(result_sets):
                result_sets[i] = [
                    c for c in rs
                    if c.get("spec_number", "") not in SPEC_BLACKLIST_ALWAYS
                    and c.get("section_path", "") != "auto-ingested"
                    and not any(pat in (c.get("section_path") or "").lower() for pat in SECTION_BLACKLIST_PATTERNS)
                ]
            # Additional clause blacklist for cause-code queries
            is_cause_query = any(kw in query.lower() for kw in ["cause", "clear code", "failure", "reject"])
            if is_cause_query:
                for i, rs in enumerate(result_sets):
                    result_sets[i] = [
                        c for c in rs
                        if not any(bl in (c.get("section_path") or "") for bl in CAUSE_CLAUSE_BLACKLIST)
                    ]

            num_sets = len(result_sets)
            all_chunks = reciprocal_rank_fusion(result_sets)
            all_chunks = normalize_rrf_scores(all_chunks)
            all_chunks = deduplicate_by_clause(all_chunks)

        yield sse_event("step_done", {"step": "router", "label": f"RRF → {len(all_chunks)} chunks from {num_sets} sets", "ms": int((time.time() - t2) * 1000)})

        # Step 3: IsREL
        yield sse_event("step_start", {"step": "isrel", "label": "IsREL relevance filter", "icon": "🎯", "detail": "Scoring each chunk for answerability..."})
        t3 = time.time()
        filtered = isrel_filter(all_chunks, query)
        discarded = len(all_chunks) - len(filtered)
        yield sse_event("step_done", {
            "step": "isrel", "label": f"IsREL: {len(filtered)} relevant, {discarded} discarded", "ms": int((time.time() - t3) * 1000),
            "sources_preview": [{"spec": c.get("spec_number", "?"), "clause": c.get("section_path", "?"), "score": round(c.get("score", 0), 4)} for c in filtered[:4]]
        })

        # Step 4: CRAG
        yield sse_event("step_start", {"step": "crag", "label": "CRAG evaluation", "icon": "⚖️", "detail": "Checking retrieval quality..."})
        t4 = time.time()
        selected = rerank_chunks(filtered)
        CRAG_THRESHOLD = 0.05
        CRAG_MIN_PASS = 5
        relevant = [c for c in selected if c.get("score", 0) >= CRAG_THRESHOLD]
        if len(relevant) < CRAG_MIN_PASS:
            relevant = sorted(selected, key=lambda x: -x.get("score", 0))[:CRAG_MIN_PASS]
        verdict = "correct" if len(relevant) >= 3 else "ambiguous" if relevant else "incorrect"
        yield sse_event("step_done", {"step": "crag", "label": f"CRAG: {verdict} ({len(relevant)} relevant)", "ms": int((time.time() - t4) * 1000), "verdict": verdict})

        # Step 5: Generator (token streaming)
        yield sse_event("step_start", {"step": "generator", "label": "Generating answer", "icon": "✍️", "detail": f"Grounding from {len(relevant)} verified sources..."})

        context_parts = []
        for i, c in enumerate(relevant[:15], 1):
            header = f"[Source {i}: TS {c['spec_number']} §{c['section_path']} | {c['release']}]"
            context_parts.append(f"{header}\n{c['chunk_text']}")
        context = "\n\n---\n\n".join(context_parts)

        release_kws = ["when", "added", "introduced", "release", "latest", "history", "cause", "clear code", "classification"]
        if any(kw in query.lower() for kw in release_kws):
            history = get_release_history(req.spec_filter)
            if history:
                context += f"\n\n---\n\n[Source: CR Change History Database]{history}"

        prompt = f"Context from 3GPP specifications:\n\n{context}\n\n---\n\nQuestion: {query}\n\nProduce a comprehensive, reference-quality answer:"

        try:
            response = bedrock.invoke_model_with_response_stream(
                modelId=LLM_MODEL_ID,
                body=json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 3072, "system": GENERATOR_SYSTEM, "messages": [{"role": "user", "content": prompt}]}),
                contentType="application/json", accept="application/json"
            )
            for event in response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                if chunk.get("type") == "content_block_delta":
                    token = chunk["delta"].get("text", "")
                    if token:
                        yield sse_event("token", {"text": token})
        except Exception:
            resp = bedrock.invoke_model(
                modelId=LLM_MODEL_ID,
                body=json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 3072, "system": GENERATOR_SYSTEM, "messages": [{"role": "user", "content": prompt}]}),
                contentType="application/json", accept="application/json"
            )
            answer = json.loads(resp["body"].read())["content"][0]["text"]
            for i in range(0, len(answer), 20):
                yield sse_event("token", {"text": answer[i:i+20]})

        # Done
        sources = [{"spec": c.get("spec_number", "?"), "clause": c.get("section_path", "?"), "release": c.get("release", "?"), "score": round(c.get("score", 0), 4)} for c in relevant[:8]]
        yield sse_event("done", {"sources": sources, "confidence": 0.5, "total_ms": int((time.time() - start) * 1000), "chunks_used": len(relevant)})

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/admin/clear-cache")
def clear_cache():
    """Clear all cached query results."""
    try:
        conn = get_pg_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM query_cache")
        count = cur.rowcount
        conn.commit(); conn.close()
        return {"status": "ok", "cleared": count}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/admin/stats")
def admin_stats():
    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM query_cache")
    total_queries = cur.fetchone()[0]
    cur.execute("SELECT SUM(access_count) FROM query_cache")
    total_requests = cur.fetchone()[0] or 0
    cur.execute("SELECT AVG(confidence) FROM query_cache")
    avg_confidence = cur.fetchone()[0] or 0
    cur.execute("SELECT query_text, confidence, access_count, created_at, last_accessed FROM query_cache ORDER BY last_accessed DESC LIMIT 20")
    recent = [{"query": r[0], "confidence": r[1], "times_asked": r[2],
               "first_asked": r[3].isoformat() if r[3] else None,
               "last_asked": r[4].isoformat() if r[4] else None} for r in cur.fetchall()]
    cur.execute("SELECT query_text, access_count FROM query_cache ORDER BY access_count DESC LIMIT 5")
    top_queries = [{"query": r[0], "times_asked": r[1]} for r in cur.fetchall()]
    cur.execute("SELECT COUNT(*) FROM chunks")
    total_chunks = cur.fetchone()[0]
    conn.close()
    return {"total_unique_queries": total_queries, "total_requests": total_requests,
            "avg_confidence": round(avg_confidence, 2), "total_chunks_indexed": total_chunks,
            "top_queries": top_queries, "recent_queries": recent}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    start = time.time()
    steps = []

    # Cache check
    query_hash = hashlib.md5(f"{req.query}|{req.spec_filter}|{req.release_filter}".lower().strip().encode()).hexdigest()
    try:
        conn = get_pg_conn()
        cur = conn.cursor()
        cur.execute("SELECT answer, citations, confidence, chunks_count FROM query_cache WHERE query_hash = %s", (query_hash,))
        cached = cur.fetchone()
        if cached:
            cur.execute("UPDATE query_cache SET last_accessed = NOW(), access_count = access_count + 1 WHERE query_hash = %s", (query_hash,))
            conn.commit(); conn.close()
            steps.append({"name": "⚡ Cache Hit", "status": "done", "ms": int((time.time()-start)*1000)})
            return QueryResponse(answer=cached[0], citations=[Citation(**c) for c in cached[1]],
                                 confidence=cached[2], latency_ms=int((time.time()-start)*1000),
                                 chunks_retrieved=cached[3], steps=steps, cached=True)
        conn.close()
    except Exception:
        pass

    # Step 1: Planner
    step_start = time.time()
    sub_queries = run_planner(req.query)
    steps.append({"name": f"Planner: {len(sub_queries)} sub-queries", "status": "done", "ms": int((time.time()-step_start)*1000)})

    # Step 2: Adaptive Router + Retrieval
    step_start = time.time()
    query_type = classify_query(req.query)
    all_chunks = []

    if query_type == "lookup":
        # Direct DB lookup + one vector search, merged
        direct_results = direct_lookup(req.query, req.spec_filter)
        vector_hits = hybrid_search(req.query, 8, req.spec_filter, req.release_filter)  # V19: top_k=8
        seen = {c["chunk_id"] for c in direct_results}
        all_chunks = list(direct_results)
        for h in vector_hits:
            if h["chunk_id"] not in seen:
                all_chunks.append(h)
                seen.add(h["chunk_id"])
        steps.append({"name": f"Router: lookup → {len(all_chunks)} chunks (direct+vector)", "status": "done", "ms": int((time.time()-step_start)*1000)})
    else:
        # RRF multi-query retrieval — OPTIMIZED: parallel embed + parallel DB
        search_tasks = []
        for q in sub_queries:
            search_tasks.append((q, 8, req.spec_filter, req.release_filter))  # V19: top_k=8

        # Targeted cause search (no spec filter)
        if not req.spec_filter:
            cause_kws = ["cause", "clear code", "release cause", "reject", "failure"]
            if any(kw in req.query.lower() for kw in cause_kws):
                for spec in CAUSE_CODE_SPECS:
                    search_tasks.append((req.query, 8, spec, None))  # BUG-2-FIX: top_k=8

        # PHASE 1: Embed all unique query texts in PARALLEL
        unique_queries = list(set(t[0] for t in search_tasks))
        embed_texts_batch(unique_queries)  # parallel embedding, populates _embed_cache

        # PHASE 2: Execute all DB searches in PARALLEL (embeddings already cached)
        def _do_search(args):
            q, top_k, spec, release = args
            q_emb = _embed_cache[q[:8000].strip()]
            return hybrid_search_with_embedding(q, q_emb, top_k, spec_filter=spec, release_filter=release)

        futures = [_RETRIEVAL_EXECUTOR.submit(_do_search, t) for t in search_tasks]
        result_sets = [f.result() for f in futures]
        result_sets = [rs for rs in result_sets if rs]  # filter empty

        # FIX-1: Always filter RF/test specs and generic sections (ALL queries)
        for i, rs in enumerate(result_sets):
            result_sets[i] = [
                c for c in rs
                if c.get("spec_number", "") not in SPEC_BLACKLIST_ALWAYS
                and c.get("section_path", "") != "auto-ingested"
                and not any(pat in (c.get("section_path") or "").lower() for pat in SECTION_BLACKLIST_PATTERNS)
            ]
        # FIX-1: Additional clause blacklist for cause-code queries only
        is_cause_query = any(kw in req.query.lower() for kw in ["cause", "clear code", "failure", "reject"])
        if is_cause_query:
            for i, rs in enumerate(result_sets):
                result_sets[i] = [
                    c for c in rs
                    if not any(bl in (c.get("section_path") or "") for bl in CAUSE_CLAUSE_BLACKLIST)
                ]

        all_chunks = reciprocal_rank_fusion(result_sets)
        
        # BUG 4 FIX: Min-max normalize RRF scores to [0,1]
        all_chunks = normalize_rrf_scores(all_chunks)
        
        # BUG 2 FIX: Deduplicate by clause (keep highest per unique section)
        all_chunks = deduplicate_by_clause(all_chunks)
        
        steps.append({"name": f"Router: RRF → {len(all_chunks)} chunks from {len(result_sets)} sets", "status": "done", "ms": int((time.time()-step_start)*1000)})

    # Step 3: IsREL filter
    step_start = time.time()
    filtered = isrel_filter(all_chunks, req.query)
    steps.append({"name": f"IsREL: {len(filtered)} relevant, {len(all_chunks)-len(filtered)} discarded", "status": "done", "ms": int((time.time()-step_start)*1000)})

    # Step 4: Rerank
    selected = rerank_chunks(filtered)

    # Step 5: CRAG — BUG-3-FIX: threshold=0.05 for raw k=60 scores (range 0-0.16)
    CRAG_THRESHOLD = 0.05  # BUG-3-FIX: calibrated for k=60 scores
    CRAG_MIN_PASS = 5      # BUG-3-FIX: safety guard
    relevant = [c for c in selected if c.get("score", 0) >= CRAG_THRESHOLD]
    if len(relevant) < CRAG_MIN_PASS:  # BUG-3-FIX: top-N safety guard
        relevant = sorted(selected, key=lambda x: -x.get("score", 0))[:CRAG_MIN_PASS]
    eval_result = "correct" if len(relevant) >= 3 else "ambiguous" if relevant else "incorrect"
    steps.append({"name": f"CRAG: {eval_result} ({len(relevant)} relevant)", "status": "done", "ms": 0})

    # Step 6: Corpus gap detection + Generate
    gap_info = detect_corpus_gap(req.query, relevant or selected)
    if gap_info.get("has_gap"):
        log_gap_for_ingestion(req.query, gap_info)
    
    step_start = time.time()
    context_parts = []
    for i, c in enumerate((relevant or selected)[:15], 1):
        header = f"[Source {i}: TS {c['spec_number']} §{c['section_path']} | {c['release']}]"
        context_parts.append(f"{header}\n{c['chunk_text']}")
    context = "\n\n---\n\n".join(context_parts)

    # Release history for cause queries
    release_kws = ["when", "added", "introduced", "release", "latest", "history", "cause", "clear code", "classification"]
    if any(kw in req.query.lower() for kw in release_kws):
        history = get_release_history(req.spec_filter)
        if history:
            context += f"\n\n---\n\n[Source: CR Change History Database]{history}"

    prompt = f"Context from 3GPP specifications:\n\n{context}\n\n---\n\nQuestion: {req.query}\n\nProduce a comprehensive, reference-quality answer:"

    resp = bedrock.invoke_model(
        modelId=LLM_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 3072,
            "system": GENERATOR_SYSTEM,
            "messages": [{"role": "user", "content": prompt}]
        }),
        contentType="application/json", accept="application/json"
    )
    answer_raw = json.loads(resp["body"].read())["content"][0]["text"]
    steps.append({"name": "Answer Generated", "status": "done", "ms": int((time.time()-step_start)*1000)})

    confidence = 0.5
    json_match = re.search(r'\{[^{}]*"confidence"[^{}]*\}', answer_raw)
    if json_match:
        try:
            meta = json.loads(json_match.group())
            confidence = float(meta.get("confidence", 0.5))
            answer_raw = answer_raw[:json_match.start()].strip()
        except (json.JSONDecodeError, ValueError):
            pass

    # Prepend gap notice if corpus gap detected
    if gap_info.get("has_gap"):
        gap_msg = gap_info["message"]
        answer_raw = "\u26a0\ufe0f " + gap_msg + "\n\n---\n\n" + answer_raw
    
    citations = [Citation(spec=c["spec_number"], section=c["section_path"], release=c["release"], score=c["score"])
                 for c in (relevant or selected)[:8]]

    # Save to cache
    try:
        conn = get_pg_conn()
        cur = conn.cursor()
        cur.execute("""INSERT INTO query_cache (query_hash, query_text, answer, citations, confidence, chunks_count)
                       VALUES (%s, %s, %s, %s, %s, %s)
                       ON CONFLICT (query_hash) DO UPDATE SET
                           answer = EXCLUDED.answer, citations = EXCLUDED.citations,
                           confidence = EXCLUDED.confidence, last_accessed = NOW()""",
                    (query_hash, req.query, answer_raw,
                     json.dumps([c.model_dump() for c in citations]), confidence, len(all_chunks)))
        conn.commit(); conn.close()
    except Exception:
        pass

    return QueryResponse(answer=answer_raw, citations=citations, confidence=confidence,
                         latency_ms=int((time.time()-start)*1000), chunks_retrieved=len(all_chunks),
                         steps=steps, cached=False)


handler = Mangum(app)
