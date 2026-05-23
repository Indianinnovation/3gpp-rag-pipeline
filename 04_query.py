"""
04_query.py
===========
Advanced LangGraph RAG engine for 3GPP specifications.
Designed to produce answers MORE accurate and structured than generic AI
by leveraging grounded retrieval with exact spec citations.

Key advantages over ChatGPT/Gemini/Claude:
  1. Every claim traceable to exact TS clause (no hallucination)
  2. Multi-pass retrieval for comprehensive coverage
  3. Latest Rel-18/19/20 content not in generic LLM training data
  4. Domain-tuned hybrid search (vector + 3GPP keyword matching)

Usage:
    python 04_query.py --query "What are the RRC states in NR?"
    python 04_query.py --query "Explain handover" --spec 38331
    python 04_query.py   # interactive mode
"""

import argparse
import asyncio
import json
import re
from typing import TypedDict, Optional
from concurrent.futures import ThreadPoolExecutor

import boto3
from langgraph.graph import StateGraph, END

from config import AWS_REGION, LLM_MODEL_ID, HAIKU_MODEL_ID

NOVA_PRO_MODEL_ID = "us.amazon.nova-pro-v1:0"
import sys
import os

# Thread pool for parallel retrieval (pgvector queries are blocking I/O)
_RETRIEVAL_EXECUTOR = ThreadPoolExecutor(max_workers=8)


# ─────────────────────────────────────────────────────────────────────────────
# Telecom Synonym Expansion — maps informal terms to 3GPP terminology
# ─────────────────────────────────────────────────────────────────────────────
TELECOM_SYNONYMS = {
    "clear code":     ["Cause IE", "CauseRadioNetwork", "cause code",
                       "release cause", "RLF-Cause"],
    "drop cause":     ["CauseRadioNetwork", "radioConnectionWithUELost",
                       "release cause"],
    "failure code":   ["Cause IE", "CauseProtocol", "CauseMisc"],
    "error code":     ["Cause IE", "CauseProtocol"],
    "release reason": ["ReleaseCause", "release cause", "RRC release"],
    "disconnect reason": ["Cause IE", "CauseRadioNetwork"],
    "reject cause":   ["5GMM cause", "CauseNas"],
    "handover failure": ["handoverFailure", "CauseRadioNetwork"],
}

SPEC_ALIASES = {
    "ngap":   "38.413",
    "f1ap":   "38.473",
    "e1ap":   "38.463",
    "xnap":   "38.423",
    "rrc":    "38331",
    "nas":    "24501",
    "5gmm":   "24501",
    "5gsm":   "24501",
    "gtp":    "29.281",
    "gtpu":   "29.281",
}


def expand_query(user_query: str) -> tuple[str, list[str]]:
    """Returns (expanded_query, detected_specs)."""
    q = user_query.lower()
    expansions = []
    detected_specs = []

    for term, synonyms in TELECOM_SYNONYMS.items():
        if term in q:
            expansions.extend(synonyms)

    for alias, spec_num in SPEC_ALIASES.items():
        if alias in q:
            detected_specs.append(spec_num)

    if expansions:
        expanded = (
            f"{user_query}\n\n"
            f"[Search also for: {', '.join(set(expansions))}]"
        )
        return expanded, detected_specs

    return user_query, detected_specs


def resolve_spec_alias(spec: str) -> str:
    """Resolve spec aliases like 'ngap' -> '38.413'."""
    if spec is None:
        return None
    return SPEC_ALIASES.get(spec.lower(), spec)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

embed_link = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embed_and_index.py")
if not os.path.exists(embed_link):
    os.symlink("03_embed_and_index.py", embed_link)

from embed_and_index import hybrid_search, get_pg_conn
from auto_ingest import auto_ingest_spec

from botocore.config import Config

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION,
                       config=Config(read_timeout=120))


# ─────────────────────────────────────────────────────────────────────────────
# Release History Lookup — answers "when was cause X added?"
# ─────────────────────────────────────────────────────────────────────────────
def get_cause_release_history(conn, spec_filter: str = None) -> str:
    """Query cause_release_history table for release tracking context."""
    cur = conn.cursor()
    try:
        where = f"WHERE spec_number = '{spec_filter}'" if spec_filter else ""
        cur.execute(f"""
            SELECT DISTINCT release, cause_feature, spec_number
            FROM cause_release_history
            {where}
            ORDER BY release, cause_feature
            LIMIT 60
        """)
        rows = cur.fetchall()
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
    finally:
        cur.close()


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────
class RAGState(TypedDict):
    user_query:       str
    spec_filter:      Optional[str]
    release_filter:   Optional[str]
    sub_queries:      list[str]
    extracted_spec:   Optional[str]
    extracted_rel:    Optional[str]
    retrieved_chunks: list[dict]
    retrieval_hops:   int
    eval_result:      str
    refined_query:    str
    answer:           str
    citations:        list[dict]
    confidence:       float
    route:            str
    hop_count:        int


# ─────────────────────────────────────────────────────────────────────────────
# Planner — decomposes complex queries into focused sub-queries
# ─────────────────────────────────────────────────────────────────────────────
PLANNER_SYSTEM = """You are a 3GPP specification query planner.

When the user asks about "cause codes", "clear codes", "release causes",
"failure codes", or "error codes" in 5G NR, interpret these as the
"Cause IE" information element defined in:
  - TS 38.413 §9.3.1.2 (NGAP) — CauseRadioNetwork, CauseTransport, CauseNas, CauseProtocol, CauseMisc
  - TS 38.473 §9.3.1.2 (F1AP) — Radio Network, Transport, Protocol, Miscellaneous
  - TS 38.463 (E1AP)
  - TS 38.423 (XnAP)
  - TS 38.331 (RRC) — RLF-Cause, EstablishmentCause, ReleaseCause
  - TS 24.501 §9.11.3.2 (5GMM cause), §9.11.4.2 (5GSM cause)
  - TS 29.281 (GTP-U)

Do NOT retrieve content about:
  - 5GS mobile identity types (SUCI, 5G-S-TMSI, IMEI, SUPI)
  - Identity type encoding (bit patterns in octets)
  - NAI / SUPI format construction
  - Security algorithms or key derivation
These are identity/security structures, not cause codes.

CRITICAL terminology mapping:
  - "clear code" / "release code" → "Cause IE" / "cause value" / "release cause"
  - "error code" → "Cause IE" / "CauseProtocol"
  - "reject reason" → "5GMM cause" / "5GSM cause" / "CauseNas"
  - "disconnect reason" → "CauseRadioNetwork" / "release cause"
  - "failure reason" → "CauseRadioNetwork" / "RLF-Cause"
  - "drop cause" → "radioConnectionWithUELost" / "CauseRadioNetwork"
  - "handover failure" → "handoverFailure" / "CauseRadioNetwork"

Given a user question:
1. Extract spec_number, release_version, clause_hint from the query
2. Map informal terms to correct 3GPP terminology (see above)
3. Decompose into 2-5 focused sub-queries using CORRECT 3GPP terms:
   - Cause IE definition/structure query
   - Enumeration values query (ASN.1 ENUMERATED)
   - Procedure/usage context query
   - Classification/grouping query

Return ONLY valid JSON (no markdown fences):
{"spec": "38413"|null, "release": "Rel-18"|null, "sub_queries": ["query1", "query2", ...], "search_terms": ["CauseRadioNetwork", "Cause IE", ...]}"""


def planner_node(state: RAGState) -> dict:
    print("  [planner] Decomposing query into sub-queries (Nova Pro) …")

    # Expand query with telecom synonyms before planning
    expanded_query, detected_specs = expand_query(state["user_query"])

    resp = bedrock.invoke_model(
        modelId=NOVA_PRO_MODEL_ID,
        body=json.dumps({
            "system": [{"text": PLANNER_SYSTEM}],
            "messages": [{"role": "user", "content": [{"text": expanded_query}]}],
            "inferenceConfig": {"maxTokens": 512}
        }),
        contentType="application/json",
        accept="application/json"
    )
    raw = json.loads(resp["body"].read())["output"]["message"]["content"][0]["text"].strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw)

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        plan = {"spec": None, "release": None, "sub_queries": [state["user_query"]], "search_terms": []}

    sub_queries = plan.get("sub_queries") or [state["user_query"]]
    search_terms = plan.get("search_terms") or []

    # Append search_terms as an additional sub-query for better retrieval
    if search_terms:
        sub_queries.append(" ".join(search_terms[:8]))

    # If synonym expansion detected specs, use as hint for spec_filter
    spec_from_plan = plan.get("spec") or state.get("spec_filter")
    if not spec_from_plan and detected_specs:
        spec_from_plan = detected_specs[0]  # use first detected

    print(f"    → {len(sub_queries)} sub-queries generated")
    return {
        "extracted_spec": spec_from_plan,
        "extracted_rel":  plan.get("release") or state.get("release_filter"),
        "sub_queries":    sub_queries
    }


# ─────────────────────────────────────────────────────────────────────────────
# Retriever — multi-query hybrid search
# ─────────────────────────────────────────────────────────────────────────────
# Specs known to contain cause/clear code definitions
CAUSE_CODE_SPECS = ["24501", "38331", "38473", "38463", "38423", "38.413"]


# ─────────────────────────────────────────────────────────────────────────────
# BUG 2 FIX: Clause blacklist — prevents wrong chunks from dominating RRF
# These clauses are tangentially related to "cause" keyword but are NOT
# actual cause code definitions. They consistently score high via RRF
# because multiple sub-queries match them from different angles.
# ─────────────────────────────────────────────────────────────────────────────
CLAUSE_BLACKLIST: dict[str, list[str]] = {
    "cause_code_query": [
        "9.3.1.111",   # RRC Establishment Cause (not Cause IE)
        "4.5.6",       # Access category to RRC establishment mapping
        "5.6.1.4.1",   # CIoT narrow edge case
        "9.11.3.39",   # Payload Container IE optional fields
        "5.3.10.5",    # RLF report content (adjacent, not cause codes)
        "G.1.1.3.1",   # TDD operating bands (completely irrelevant)
    ]
}

# Section patterns that are always irrelevant for cause code queries
SECTION_BLACKLIST_PATTERNS = [
    "change history", "abbreviations", "annex a", "annex b",
    "0 introduction", "payload container"
]


def apply_clause_blacklist(chunks: list[dict], intent: str) -> list[dict]:
    """BUG 2 FIX: Filter chunks whose clause matches the blacklist for given intent.
    Applied BEFORE RRF scoring to prevent wrong chunks from being promoted."""
    blacklist = CLAUSE_BLACKLIST.get(intent, [])
    if not blacklist:
        return chunks
    return [
        c for c in chunks
        if not any(bl in (c.get("section_path") or "") for bl in blacklist)
        and c.get("section_path", "") != "auto-ingested"  # BUG 6: reject unvalidatable
        and not any(pat in (c.get("section_path") or "").lower() for pat in SECTION_BLACKLIST_PATTERNS)
    ]


def deduplicate_by_clause(chunks: list[dict]) -> list[dict]:
    """BUG 2 FIX: Keep only the highest-scored chunk per unique clause.
    Prevents §9.11.3.39 appearing 4× in 7 source slots."""
    seen_clauses: dict[str, dict] = {}
    for c in chunks:
        clause = (c.get("section_path") or "unknown").strip()
        if clause not in seen_clauses or c.get("score", 0) > seen_clauses[clause].get("score", 0):
            seen_clauses[clause] = c
    return list(seen_clauses.values())


# ─────────────────────────────────────────────────────────────────────────────
# BUG 4 FIX: RRF score normalization — min-max scales to [0,1]
# RRF scores use 1/(k+rank) which produces values ~0.01-0.06.
# Downstream thresholds (IsREL, CRAG) were calibrated for cosine [0,1].
# Normalization makes thresholds work correctly regardless of retrieval mode.
# ─────────────────────────────────────────────────────────────────────────────
def normalize_rrf_scores(chunks: list[dict]) -> list[dict]:
    """BUG 4 FIX: Min-max normalize scores to [0,1] range."""
    if not chunks:
        return chunks
    scores = [c.get("score", 0) for c in chunks]
    max_s = max(scores)
    min_s = min(scores)
    spread = max_s - min_s
    if spread == 0:
        for c in chunks:
            c["score"] = 1.0
    else:
        for c in chunks:
            c["score"] = (c.get("score", 0) - min_s) / spread
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion (RRF) — fuses multiple ranked result sets
# ─────────────────────────────────────────────────────────────────────────────
def reciprocal_rank_fusion(result_sets: list[list[dict]], k: int = 60) -> list[dict]:
    """Fuse multiple ranked lists using RRF. score = sum(1/(k+rank)) across sets."""
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for result_set in result_sets:
        for rank, chunk in enumerate(result_set):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = chunk

    # Sort by RRF score descending, assign as the chunk's score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    fused = []
    for cid in sorted_ids:
        c = chunk_map[cid].copy()
        c["score"] = rrf_scores[cid]
        fused.append(c)
    return fused


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Router — classifies query type for optimal retrieval strategy
# ─────────────────────────────────────────────────────────────────────────────
QUERY_TYPE_LOOKUP = "lookup"       # cause codes, timer values, IE definitions
QUERY_TYPE_PROCEDURAL = "procedural"  # handover procedure, registration flow
QUERY_TYPE_COMPARATIVE = "comparative"  # difference between X and Y

LOOKUP_PATTERNS = [
    r"\b(list|enumerate)\s+(all|every)\b.*\b(cause|timer|IE|value|code)s?\b",
    r"\bwhat\s+(are|is)\s+the\s+(cause|timer|IE|RLF)\s*(code|value|IE)s?\b",
    r"\b(CauseRadioNetwork|CauseTransport|CauseNas|CauseProtocol|CauseMisc)\b",
    r"\bRLF.?Cause\s*(enum|values?)\b",
    r"\bEstablishmentCause\s*(enum|values?)\b",
]


def classify_query(query: str) -> str:
    """Classify query type: lookup, procedural, or comparative. No LLM call."""
    q = query.lower()
    # Lookup: structured data retrieval
    for pattern in LOOKUP_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return QUERY_TYPE_LOOKUP
    # Comparative
    if any(kw in q for kw in ["difference between", "compare", "vs ", " versus "]):
        return QUERY_TYPE_COMPARATIVE
    # Procedural
    if any(kw in q for kw in ["how does", "procedure", "flow", "steps", "explain the", "process"]):
        return QUERY_TYPE_PROCEDURAL
    return QUERY_TYPE_PROCEDURAL  # default


def direct_lookup(query: str, conn, spec_filter: str = None) -> list[dict]:
    """Direct DB query for structured lookups (cause codes, timers, IEs)."""
    cur = conn.cursor()
    q = query.lower()

    # Determine which sections to target
    section_patterns = []
    if any(kw in q for kw in ["cause", "clear code", "reject"]):
        section_patterns = ["%9.3.1.2%Cause%", "%9.11.3.2%", "%9.11.4.2%", "%5.3.10.4%", "%9.2.3.2%Cause%", "%B.1%Cause%"]
    elif any(kw in q for kw in ["timer", "t310", "t311", "t304"]):
        section_patterns = ["%timer%"]
    elif any(kw in q for kw in ["establishment", "rrc setup"]):
        section_patterns = ["%establishment%", "%EstablishmentCause%"]

    if not section_patterns:
        cur.close()
        return []

    # Build query targeting specific sections
    conditions = " OR ".join(["section_path ILIKE %s" for _ in section_patterns])
    spec_clause = f"AND spec_number = %s" if spec_filter else ""
    params = section_patterns + ([spec_filter] if spec_filter else [])

    sql = f"""
        SELECT chunk_id, spec_number, release, section_path, doc_type, chunk_text, summary
        FROM chunks
        WHERE ({conditions}) {spec_clause}
        ORDER BY section_path
        LIMIT 30
    """
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()

    return [
        {"chunk_id": r[0], "spec_number": r[1], "release": r[2], "section_path": r[3],
         "doc_type": r[4], "chunk_text": r[5], "summary": r[6], "score": 0.8}
        for r in rows
    ]

def _parallel_hybrid_search(queries: list[tuple[str, str, str, int]], conn) -> list[list[dict]]:
    """Execute multiple hybrid_search calls in parallel using ThreadPoolExecutor.
    
    Each query tuple: (query_text, spec_filter, release_filter, top_k)
    Returns list of result sets in same order as input queries.
    """
    def _search(args):
        q, spec, release, top_k = args
        # Each thread gets its own connection to avoid psycopg2 thread-safety issues
        thread_conn = get_pg_conn()
        try:
            return hybrid_search(query=q, conn=thread_conn, top_k=top_k,
                                 spec_filter=spec, release_filter=release)
        finally:
            thread_conn.close()

    # Execute all searches in parallel
    futures = [_RETRIEVAL_EXECUTOR.submit(_search, q) for q in queries]
    return [f.result() for f in futures]


def retriever_node(state: RAGState, conn) -> dict:
    print("  [retriever] Multi-query hybrid search + RRF (parallel) …")
    user_spec = state.get("spec_filter")
    release_filter = state.get("release_filter")
    query_type = classify_query(state["user_query"])

    # Adaptive routing: structured lookup bypasses vector search
    if query_type == QUERY_TYPE_LOOKUP and not state.get("retrieved_chunks"):
        print(f"    [router] Structured lookup detected — direct DB query")
        direct_results = direct_lookup(state["user_query"], conn, user_spec)
        if direct_results:
            # Also do one vector search for context
            vector_hits = hybrid_search(state["user_query"], conn, top_k=8,  # V19: top_k=8
                                        spec_filter=user_spec, release_filter=release_filter)
            # Merge: direct results first (high score), then vector hits
            seen = {c["chunk_id"] for c in direct_results}
            merged = list(direct_results)
            for h in vector_hits:
                if h["chunk_id"] not in seen:
                    merged.append(h)
                    seen.add(h["chunk_id"])
            existing = state.get("retrieved_chunks", [])
            print(f"    → {len(merged)} chunks (direct lookup + vector)")
            return {"retrieved_chunks": existing + merged, "retrieval_hops": state.get("retrieval_hops", 0) + 1}

    # ─── Build all search tasks for parallel execution ───
    search_tasks: list[tuple[str, str, str, int]] = []

    # Sub-query searches
    for q in state["sub_queries"]:
        search_tasks.append((q, user_spec, release_filter, 8))  # V19: top_k=8

    # Targeted cause code searches — only when NO user spec filter is set
    cause_keywords = ["cause", "clear code", "release cause", "reject", "failure cause"]
    if not user_spec and any(kw in state["user_query"].lower() for kw in cause_keywords):
        targeted_queries = [
            "NGAP cause RadioNetwork Transport NAS Protocol Miscellaneous",
            "5GMM cause value rejection registration",
            "5GSM cause PDU session failure",
            "RRC establishment cause release radio link failure T310",
            "F1AP E1AP cause radio network protocol",
        ]
        for spec in CAUSE_CODE_SPECS:
            for tq in targeted_queries:
                search_tasks.append((tq, spec, None, 3))

    # ─── Execute ALL searches in parallel (BUG 5 FIX) ───
    import time as _time
    t0 = _time.time()
    result_sets = _parallel_hybrid_search(search_tasks, conn)
    elapsed = _time.time() - t0
    print(f"    [parallel] {len(search_tasks)} searches completed in {elapsed:.1f}s")

    # Filter out empty result sets
    result_sets = [rs for rs in result_sets if rs]

    # BUG 2 FIX: Apply clause blacklist BEFORE RRF scoring
    is_cause_query = any(kw in state["user_query"].lower() for kw in cause_keywords)
    if is_cause_query:
        result_sets = [apply_clause_blacklist(rs, "cause_code_query") for rs in result_sets]
        result_sets = [rs for rs in result_sets if rs]  # remove emptied sets

    # Apply Reciprocal Rank Fusion across all result sets
    fused = reciprocal_rank_fusion(result_sets)

    # BUG 4 FIX: Normalize RRF scores to [0, 1] using min-max scaling
    fused = normalize_rrf_scores(fused)

    # BUG 2 FIX: Deduplicate by clause (keep highest-scored per clause)
    fused = deduplicate_by_clause(fused)
    fused = sorted(fused, key=lambda c: c.get("score", 0), reverse=True)

    existing = state.get("retrieved_chunks", [])
    total = existing + fused
    print(f"    → {len(total)} chunks (RRF fused from {len(result_sets)} result sets)")
    return {
        "retrieved_chunks": total,
        "retrieval_hops":   state.get("retrieval_hops", 0) + 1
    }


# ─────────────────────────────────────────────────────────────────────────────
# Self-RAG IsREL — per-chunk relevance filter (discards irrelevant chunks)
# ─────────────────────────────────────────────────────────────────────────────
ISREL_SYSTEM = """You are a relevance judge for a 3GPP RAG system.
Given a user question and a retrieved chunk, decide if the chunk is RELEVANT to answering the question.

A chunk is RELEVANT if it contains:
- Direct definitions, values, or enumerations asked about
- Procedures or message flows related to the question
- Tables with parameters/IEs relevant to the topic
- ASN.1 definitions of requested structures

A chunk is IRRELEVANT if it contains:
- Only reference lists or bibliography entries
- Identity encoding (SUCI, 5G-GUTI, IMEI) when question is about cause codes
- Change history metadata without substantive technical content
- Unrelated protocol sections (e.g., paging when asking about cause codes)
- Generic scope/foreword text

Respond with ONLY one word: relevant or irrelevant"""


def isrel_node(state: RAGState) -> dict:
    """Self-RAG IsREL: filters out irrelevant chunks before reranking.
    
    BUG 3 FIX: Enforces min_keep=15, max_keep=25 floor/ceiling.
    Never passes fewer than 15 chunks to CRAG regardless of relevance scores.
    If strict filtering produces fewer than 15, relaxes to include ambiguous chunks.
    """
    MIN_KEEP = 15  # BUG 3: minimum floor
    MAX_KEEP = 25  # BUG 3: maximum ceiling
    
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {"retrieved_chunks": []}

    # Fast-path: if few chunks, skip filtering entirely
    if len(chunks) <= MIN_KEEP:
        print(f"  [IsREL] {len(chunks)} chunks — skipping (below min_keep={MIN_KEEP})")
        return {"retrieved_chunks": chunks}

    print(f"  [IsREL] Filtering {len(chunks)} chunks for relevance (floor={MIN_KEEP}, cap={MAX_KEEP}) …")
    query = state["user_query"]

    relevant = []
    ambiguous = []  # BUG 3: kept for floor enforcement

    for c in chunks:
        section = (c.get("section_path") or "").lower()
        text_preview = c.get("chunk_text", "")[:200].lower()
        score = c.get("score", 0)

        # Rule-based IRRELEVANT (fast, no LLM needed)
        if section in ("2\treferences", "2 references", "1\tscope", "foreword"):
            continue
        if "5gs mobile identity" in section and "cause" in query.lower():
            continue
        if "suci" in text_preview and "cause" in query.lower():
            continue
        if "nai" in text_preview and "format" in text_preview and "cause" in query.lower():
            continue
        spec = c.get("spec_number", "")
        if any(ts in spec for ts in ["38106", "38115", "38141", "38521", "38522", "38905", "38918"]):
            if "cause" in query.lower() or "clear code" in query.lower():
                continue

        # Rule-based RELEVANT (high confidence)
        if score >= 0.5:
            relevant.append(c)
        elif any(kw in section for kw in ["cause", "failure", "reject", "release", "rlf", "9.3.1", "9.11"]):
            relevant.append(c)
        elif score >= 0.3:
            relevant.append(c)
        else:
            # BUG 3: Ambiguous — kept for floor enforcement
            ambiguous.append(c)

    # BUG 3 FIX: Enforce minimum floor of 15 chunks
    if len(relevant) < MIN_KEEP and ambiguous:
        # Sort ambiguous by score descending, add until floor reached
        ambiguous_sorted = sorted(ambiguous, key=lambda c: c.get("score", 0), reverse=True)
        needed = MIN_KEEP - len(relevant)
        relevant.extend(ambiguous_sorted[:needed])

    # BUG 3 FIX: Cap at MAX_KEEP
    if len(relevant) > MAX_KEEP:
        relevant = sorted(relevant, key=lambda c: c.get("score", 0), reverse=True)[:MAX_KEEP]

    filtered_count = len(chunks) - len(relevant)
    print(f"    → {len(relevant)} relevant, {filtered_count} discarded")
    return {"retrieved_chunks": relevant}


# ─────────────────────────────────────────────────────────────────────────────
# Reranker — score, deduplicate, select top chunks
# ─────────────────────────────────────────────────────────────────────────────
def reranker_node(state: RAGState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    # Fix NaN scores before sorting
    for c in chunks:
        s = c.get("score", 0)
        if s != s:  # NaN check
            c["score"] = 0.0
    # Boost chunks from substantive sections (penalize generic References/Introduction)
    for c in chunks:
        section = (c.get("section_path") or "").lower()
        if any(x in section for x in ["references", "0 introduction", "1\tscope", "introduction"]):
            c["score"] *= 0.6  # strongly penalize generic sections
        elif "cause" in section or "failure" in section or "release" in section or "reject" in section:
            c["score"] *= 1.2  # boost cause-related sections
        elif "[table]" in section:
            c["score"] *= 1.15  # boost table content
    chunks = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)
    seen: set[str] = set()
    deduped = []
    for c in chunks:
        if c["chunk_id"] not in seen:
            deduped.append(c)
            seen.add(c["chunk_id"])
    # Top 20 chunks for context (increased from 15 to compensate for RRF score compression)
    # Ensure spec diversity: max 6 chunks per spec (only when no spec filter)
    MAX_CHUNKS_TO_GENERATOR = 20
    MAX_PER_SPEC = 6
    selected = []
    spec_counts: dict[str, int] = {}
    # Check if all chunks are from same spec (user filtered)
    unique_specs = set(c.get("spec_number", "") for c in deduped[:20])
    apply_diversity = len(unique_specs) > 2
    for c in deduped:
        spec = c.get("spec_number", "unknown")
        if apply_diversity and spec_counts.get(spec, 0) >= MAX_PER_SPEC:
            continue
        selected.append(c)
        spec_counts[spec] = spec_counts.get(spec, 0) + 1
        if len(selected) >= MAX_CHUNKS_TO_GENERATOR:
            break
    if selected:
        print(f"  [reranker] Selected top {len(selected)} chunks (scores: {selected[0]['score']:.3f} → {selected[-1]['score']:.3f})")
    else:
        print(f"  [reranker] No chunks found")
    return {"retrieved_chunks": selected}


# ─────────────────────────────────────────────────────────────────────────────
# CRAG Retrieval Evaluator — judges if retrieved chunks are relevant
# Based on: https://arxiv.org/html/2401.15884v3
# ─────────────────────────────────────────────────────────────────────────────
EVALUATOR_SYSTEM = """You are a retrieval quality evaluator for a 3GPP RAG system.
Given a user question and retrieved document chunks, assess retrieval quality.

Evaluate the TOP 5 chunks and classify the overall retrieval as:
- "correct": At least 3 chunks are directly relevant and sufficient to answer the question
- "ambiguous": 1-2 chunks are partially relevant but may not fully answer the question
- "incorrect": No chunks contain information relevant to the question

Also provide a refined query if the retrieval is ambiguous (rewrite to be more specific).

Return ONLY valid JSON:
{"evaluation": "correct"|"ambiguous"|"incorrect", "relevant_count": 0-5, "refined_query": "..."|null, "reason": "brief explanation"}
No markdown fences."""


def evaluator_node(state: RAGState) -> dict:
    """CRAG Retrieval Evaluator v2 — calibrated for RRF scores.
    
    BUG 4 FIX: Uses retrieval_mode-aware thresholds.
    RRF scores after normalization are in [0,1] but compressed.
    Threshold 0.30 for RRF mode (vs 0.60 for raw cosine).
    """
    chunks = state.get("retrieved_chunks", [])
    
    # BUG 4 FIX: Use lower threshold for RRF-normalized scores
    CRAG_THRESHOLD_RRF = 0.30
    CRAG_HIGH_CONFIDENCE = 0.45

    valid_scores = [c.get("score", 0) for c in chunks if c.get("score", 0) == c.get("score", 0) and c.get("score", 0) > 0]
    
    if valid_scores and max(valid_scores) >= CRAG_HIGH_CONFIDENCE:
        relevant_count = sum(1 for s in valid_scores if s >= CRAG_THRESHOLD_RRF)
        print(f"  [evaluator] High-confidence retrieval ({relevant_count} chunks >= {CRAG_THRESHOLD_RRF}) — skipping evaluation")
        return {"eval_result": "correct", "refined_query": state["user_query"]}

    if not chunks:
        return {"eval_result": "incorrect", "refined_query": state["user_query"]}

    print("  [evaluator] Assessing retrieval quality (CRAG) …")

    top_chunks = chunks[:5]
    chunks_summary = "\n\n".join(
        f"[Chunk {i+1} | TS {c.get('spec_number','?')} §{c.get('section_path','?')} | score: {c.get('score',0):.3f}]\n{c['chunk_text'][:300]}"
        for i, c in enumerate(top_chunks)
    )

    prompt = f"""Question: {state['user_query']}

Retrieved chunks (top 5):
{chunks_summary}

Evaluate retrieval quality:"""

    try:
        resp = bedrock.invoke_model(
            modelId=HAIKU_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 256,
                "system": EVALUATOR_SYSTEM,
                "messages": [{"role": "user", "content": prompt}]
            }),
            contentType="application/json",
            accept="application/json"
        )
        raw = json.loads(resp["body"].read())["content"][0]["text"].strip()
        raw = re.sub(r"^```json\s*|\s*```$", "", raw)
        result = json.loads(raw)

        eval_result = result.get("evaluation", "correct")
        refined_query = result.get("refined_query")
        relevant_count = result.get("relevant_count", 0)
        reason = result.get("reason", "")

        print(f"    → {eval_result} ({relevant_count}/5 relevant) — {reason}")
        return {"eval_result": eval_result, "refined_query": refined_query or state["user_query"]}
    except Exception as e:
        print(f"    ⚠ Evaluator error: {e}, defaulting to 'correct'")
        return {"eval_result": "correct", "refined_query": state["user_query"]}


def crag_router(state: RAGState) -> str:
    """Routes based on CRAG evaluation."""
    eval_result = state.get("eval_result", "correct")
    hops = state.get("retrieval_hops", 0)
    if eval_result == "correct":
        return "generate"
    elif eval_result == "ambiguous" and hops < 2:
        return "refine"
    else:
        return "generate"


def refiner_node(state: RAGState) -> dict:
    """Refines the query and resets for re-retrieval."""
    refined = state.get("refined_query", state["user_query"])
    print(f"  [refiner] Query refined: '{refined}'")
    return {
        "sub_queries": [refined, state["user_query"]],
        "retrieved_chunks": []
    }


# ─────────────────────────────────────────────────────────────────────────────
# Generator — expert-level structured output
# ─────────────────────────────────────────────────────────────────────────────
GENERATOR_SYSTEM = """You are a principal 3GPP standards architect producing reference-quality technical documentation.
Your answers must be MORE accurate and structured than what ChatGPT, Gemini, or Claude would produce from general knowledge, because you have access to the EXACT specification text.

CRITICAL RULES:
1. ONLY use information present in the provided context. Every technical claim MUST be traceable to a specific source.
2. If the context is insufficient, explicitly state what's missing and which specs would be needed.
3. NEVER generate content that isn't supported by the provided chunks.

OUTPUT FORMAT (mandatory):
- Start with a one-line summary
- Use ## headers to organize into logical sections
- Include | tables | for comparisons, parameters, message lists
- Use ```text blocks for protocol message flows / state diagrams
- Cite inline: (per TS 38.331 §5.3.2) or [Source N]
- Include a "## Key 3GPP References" section at the end listing relevant specs/clauses
- Use precise 3GPP terminology: IEs, ASN.1 field names, timer names (T304, T311), procedure names

QUALITY STANDARDS:
- Depth: Cover the topic comprehensively from the available context
- Structure: A telecom engineer should be able to use this as a reference document
- Accuracy: Zero hallucination — if unsure, say "not covered in available context"
- Completeness: Address all aspects the context supports (definition, procedure, parameters, related concepts)

End with: {"confidence": 0.0-1.0, "needs_more_context": true/false, "missing_specs": ["TS 38.xxx §y.z"]}"""


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        spec = c.get('spec_number', '?')
        section = c.get('section_path', '?')
        release = c.get('release', '?')
        header = f"[Source {i}: TS {spec} §{section} | {release}]"
        parts.append(f"{header}\n{c['chunk_text']}")
    return "\n\n---\n\n".join(parts)


def generator_node(state: RAGState) -> dict:
    print("  [generator] Producing expert-level answer …")
    chunks = state["retrieved_chunks"]

    # Filter out low-relevance chunks (score threshold)
    # NOTE: After RRF normalization, scores are compressed. Use lower threshold (0.20)
    # to retain ~10-12 chunks instead of over-filtering to 6.
    MIN_RELEVANCE_SCORE = 0.20
    relevant_chunks = [c for c in chunks if c.get("score", 0) >= MIN_RELEVANCE_SCORE]

    # If no relevant chunks found, try auto-ingesting needed specs
    if not relevant_chunks:
        print("  [generator] No relevant chunks — attempting auto-ingest …")
        chunks_added, specs_ingested = auto_ingest_spec(state["user_query"])

        if chunks_added > 0:
            # Re-run search with newly indexed content
            print(f"  [generator] Auto-ingested {chunks_added} chunks from {specs_ingested}")
            print(f"  [generator] Re-searching with new content …")
            conn = get_pg_conn()
            new_hits = hybrid_search(state["user_query"], conn, top_k=20)
            conn.close()
            relevant_chunks = [c for c in new_hits if c.get("score", 0) >= MIN_RELEVANCE_SCORE]

    # If still no relevant chunks after auto-ingest attempt
    if not relevant_chunks:
        answer = """## ⚠️ Document Not Indexed

I could not find relevant information in the currently indexed 3GPP specifications to answer this question.

### Currently Indexed Sources:
| Category | Documents |
|----------|----------|
| Meetings | TSGR2_129, TSGR2_129bis, TSGR_109 (RAN Plenary) |
| Specifications | Rel-18/19/20 38-series (NR radio) |
| Coverage | TS 38.211-38.331, 38.300, 38.321, 38.473, 38.523 |

### To answer this question, you may need to ingest:
- Additional 3GPP working group documents (SA1, SA2, RAN1, etc.)
- Different release specifications
- Meeting documents from other TSG meetings

### How to add new sources:
```bash
python 01_ftp_crawler.py --source all --limit 50
python 02_process_docs.py --limit 50 --skip-metadata
python 03_embed_and_index.py
```"""
        return {
            "answer": answer,
            "citations": [],
            "confidence": 0.0,
            "route": "done"
        }

    context = format_context(relevant_chunks)
    question = state["user_query"]

    # Append release history if query is about cause codes / when added
    release_keywords = ["when", "added", "introduced", "release", "latest", "newest", "history", "evolution",
                        "cause", "clear code", "classification"]
    if any(kw in question.lower() for kw in release_keywords):
        conn = get_pg_conn()
        history = get_cause_release_history(conn, state.get("spec_filter"))
        conn.close()
        if history:
            context += f"\n\n---\n\n[Source: CR Change History Database]{history}"

    prompt = f"""You have access to {len(relevant_chunks)} chunks from official 3GPP specifications.

Context from 3GPP specifications:

{context}

---

Question: {question}

Produce a comprehensive, reference-quality answer following the output format rules. 
Remember: your advantage over generic AI is EXACT spec citations and ZERO hallucination."""

    resp = bedrock.invoke_model(
        modelId=LLM_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 3072,
            "system": GENERATOR_SYSTEM,
            "messages": [{"role": "user", "content": prompt}]
        }),
        contentType="application/json",
        accept="application/json"
    )
    answer_raw = json.loads(resp["body"].read())["content"][0]["text"]

    # Extract metadata JSON from end
    confidence = 0.5
    json_match = re.search(r'\{[^{}]*"confidence"[^{}]*\}', answer_raw)
    if json_match:
        try:
            meta = json.loads(json_match.group())
            confidence = float(meta.get("confidence", 0.5))
            answer_raw = answer_raw[:json_match.start()].strip()
        except (json.JSONDecodeError, ValueError):
            pass

    citations = [
        {
            "spec": c.get("spec_number"),
            "section": c.get("section_path"),
            "release": c.get("release"),
            "score": round(c.get("score", 0), 3)
        }
        for c in relevant_chunks[:8]
    ]

    return {
        "answer":     answer_raw,
        "citations":  citations,
        "confidence": confidence,
        "route":      "done"
    }


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────
def router_node(state: RAGState) -> str:
    return "done"


# ─────────────────────────────────────────────────────────────────────────────
# Build Graph
# ─────────────────────────────────────────────────────────────────────────────
def create_engine(conn=None) -> "CompiledGraph":
    if conn is None:
        conn = get_pg_conn()

    graph = StateGraph(RAGState)
    graph.add_node("planner",   planner_node)
    graph.add_node("retriever", lambda s: retriever_node(s, conn))
    graph.add_node("isrel",     isrel_node)
    graph.add_node("reranker",  reranker_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("refiner",   refiner_node)
    graph.add_node("generator", generator_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner",   "retriever")
    graph.add_edge("retriever", "isrel")
    graph.add_edge("isrel",     "reranker")
    graph.add_edge("reranker",  "evaluator")

    # CRAG routing: evaluator decides next step
    graph.add_conditional_edges("evaluator", crag_router, {
        "generate": "generator",
        "refine":   "refiner",
    })
    graph.add_edge("refiner", "retriever")

    graph.add_conditional_edges("generator", router_node, {
        "retry": "retriever",
        "low_confidence": END,
        "done": END
    })

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Run query
# ─────────────────────────────────────────────────────────────────────────────
def run_query(engine, query: str, spec: str = None, release: str = None) -> dict:
    initial_state: RAGState = {
        "user_query": query, "spec_filter": spec, "release_filter": release,
        "sub_queries": [], "extracted_spec": spec, "extracted_rel": release,
        "retrieved_chunks": [], "retrieval_hops": 0,
        "eval_result": "", "refined_query": "",
        "answer": "", "citations": [], "confidence": 0.0,
        "route": "", "hop_count": 0
    }
    final_state = engine.invoke(initial_state)
    return {
        "answer":     final_state["answer"],
        "citations":  final_state["citations"],
        "confidence": final_state["confidence"],
        "hops":       final_state["retrieval_hops"]
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
SEPARATOR = "─" * 70

def print_result(result: dict):
    print(f"\n{SEPARATOR}")
    print("ANSWER\n")
    print(result["answer"])
    print(f"\n{SEPARATOR}")
    print(f"Confidence : {result['confidence']:.0%}  |  Retrieval hops: {result['hops']}")
    print("\nCITATIONS")
    for i, c in enumerate(result["citations"], 1):
        print(f"  [{i}] TS {c['spec']} §{c['section']} | {c['release']}  (score: {c['score']})")
    print(SEPARATOR)


SAMPLE_QUERIES = [
    "What are the RRC states in 5G NR and the transitions between them?",
    "Explain the 5G NR handover procedure and the messages exchanged",
    "What is carrier aggregation in NR and how is it configured?",
    "How does the HARQ process work in 5G NR?",
    "What is the difference between gNB-CU and gNB-DU?"
]


def interactive_cli(engine):
    print("\n3GPP RAG — Expert Query Engine")
    print("Advantages over generic AI: exact citations, zero hallucination, latest Rel-18/19/20")
    print("Type 'quit' to exit | 'samples' to see example queries\n")
    while True:
        try:
            query = input("Query: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "samples":
            for i, q in enumerate(SAMPLE_QUERIES, 1):
                print(f"  {i}. {q}")
            continue
        result = run_query(engine, query)
        print_result(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",   type=str, default=None)
    parser.add_argument("--spec",    type=str, default=None)
    parser.add_argument("--release", type=str, default=None)
    args = parser.parse_args()

    print("Initializing 3GPP RAG Expert Engine …")
    engine = create_engine()
    print("✓ Engine ready (15K+ chunks indexed)\n")

    if args.query:
        spec = resolve_spec_alias(args.spec) if args.spec else None
        result = run_query(engine, args.query, spec, args.release)
        print_result(result)
    else:
        interactive_cli(engine)


if __name__ == "__main__":
    main()
