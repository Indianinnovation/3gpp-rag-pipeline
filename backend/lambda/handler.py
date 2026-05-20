"""
Lambda handler for 3GPP RAG API.
Uses Mangum to wrap FastAPI for Lambda + API Gateway.
"""

import json
import re
import time
import boto3
import psycopg2
from mangum import Mangum
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

# Config from environment variables
AWS_REGION  = os.environ.get("AWS_REGION", "us-east-1")
PG_HOST     = os.environ["PG_HOST"]
PG_PORT     = int(os.environ.get("PG_PORT", "5432"))
PG_DATABASE = os.environ.get("PG_DATABASE", "ragdb")
PG_USER     = os.environ.get("PG_USER", "ragadmin")
PG_PASSWORD = os.environ["PG_PASSWORD"]
PG_TABLE    = os.environ.get("PG_TABLE", "chunks")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")

app = FastAPI(title="3GPP RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def get_pg_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DATABASE,
        user=PG_USER, password=PG_PASSWORD
    )


def embed_text(text: str) -> list[float]:
    resp = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps({"inputText": text[:8000]}),
        contentType="application/json",
        accept="application/json"
    )
    return json.loads(resp["body"].read())["embedding"]


def hybrid_search(query: str, top_k=10, spec_filter=None, release_filter=None):
    query_emb = embed_text(query)
    conditions, params = [], []
    if spec_filter:
        conditions.append("spec_number = %s")
        params.append(spec_filter)
    if release_filter:
        conditions.append("release = %s")
        params.append(release_filter)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    sql = f"""
        SELECT chunk_id, spec_number, release, section_path, doc_type,
               chunk_text, summary,
               (1 - (embedding <=> %s::vector)) AS vec_score,
               ts_rank(to_tsvector('english', chunk_text), plainto_tsquery('english', %s)) AS text_score
        FROM {PG_TABLE} {where}
        ORDER BY (0.7 * (1 - (embedding <=> %s::vector)) +
                  0.3 * ts_rank(to_tsvector('english', chunk_text), plainto_tsquery('english', %s))) DESC
        LIMIT %s
    """
    all_params = [str(query_emb), query] + params + [str(query_emb), query, top_k]
    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute(sql, all_params)
    rows = cur.fetchall()
    conn.close()
    return [
        {"chunk_id": r[0], "spec_number": r[1], "release": r[2],
         "section_path": r[3], "doc_type": r[4], "chunk_text": r[5],
         "summary": r[6], "score": round(0.7 * r[7] + 0.3 * r[8], 4)}
        for r in rows
    ]


GENERATOR_SYSTEM = """You are a principal 3GPP standards architect producing reference-quality technical documentation.
Your answers must be MORE accurate than ChatGPT/Gemini because you have the EXACT specification text.

RULES:
1. ONLY use information from the provided context. Every claim MUST be traceable.
2. Never hallucinate content not in the context.
3. Use markdown: ## headers, | tables |, ```code blocks for protocol flows
4. Cite inline: (per TS 38.331 §5.3.2)
5. Include a Key References section at the end.

End with: {"confidence": 0.0-1.0}"""


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


@app.get("/health")
def health():
    return {"status": "ok", "model": LLM_MODEL_ID, "chunks": "40K+"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    start = time.time()

    hits = hybrid_search(req.query, req.top_k or 10, req.spec_filter, req.release_filter)

    context_parts = []
    for i, c in enumerate(hits[:15], 1):
        header = f"[Source {i}: TS {c['spec_number']} §{c['section_path']} | {c['release']}]"
        context_parts.append(f"{header}\n{c['chunk_text']}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"Context from 3GPP specifications:\n\n{context}\n\nQuestion: {req.query}\n\nProduce a comprehensive, reference-quality answer:"

    resp = bedrock.invoke_model(
        modelId=LLM_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "system": GENERATOR_SYSTEM,
            "messages": [{"role": "user", "content": prompt}]
        }),
        contentType="application/json",
        accept="application/json"
    )
    answer_raw = json.loads(resp["body"].read())["content"][0]["text"]

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
        Citation(spec=c["spec_number"], section=c["section_path"],
                 release=c["release"], score=c["score"])
        for c in hits[:8]
    ]

    return QueryResponse(
        answer=answer_raw,
        citations=citations,
        confidence=confidence,
        latency_ms=int((time.time() - start) * 1000),
        chunks_retrieved=len(hits)
    )


# Lambda handler
handler = Mangum(app)
