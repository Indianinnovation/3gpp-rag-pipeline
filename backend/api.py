"""
backend/api.py
==============
FastAPI server wrapping the 3GPP RAG query engine.
Provides REST API for the React frontend.

Usage:
    pip install fastapi uvicorn
    cd backend && uvicorn api:app --reload --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create symlink for module import
embed_link = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embed_and_index.py")
if not os.path.exists(embed_link):
    os.symlink("03_embed_and_index.py", embed_link)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time

from embed_and_index import hybrid_search, get_pg_conn
from config import AWS_REGION, LLM_MODEL_ID

import boto3
import json
import re

app = FastAPI(title="3GPP RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

GENERATOR_SYSTEM = """You are a principal 3GPP standards architect producing reference-quality technical documentation.
Your answers must be MORE accurate and structured than what ChatGPT, Gemini, or Claude would produce from general knowledge, because you have access to the EXACT specification text.

CRITICAL RULES:
1. ONLY use information present in the provided context. Every technical claim MUST be traceable to a specific source.
2. If the context is insufficient, explicitly state what's missing.
3. NEVER generate content that isn't supported by the provided chunks.

OUTPUT FORMAT:
- Start with a one-line summary
- Use ## headers to organize into logical sections
- Include | tables | for comparisons and parameters
- Use text blocks for protocol message flows
- Cite inline: (per TS 38.331 §5.3.2) or [Source N]
- Include a Key 3GPP References section at the end
- Use precise 3GPP terminology

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
    return {"status": "ok", "model": LLM_MODEL_ID}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    start = time.time()

    conn = get_pg_conn()
    hits = hybrid_search(
        query=req.query,
        conn=conn,
        top_k=req.top_k or 10,
        spec_filter=req.spec_filter,
        release_filter=req.release_filter,
    )
    conn.close()

    # Format context
    context_parts = []
    for i, c in enumerate(hits[:10], 1):
        header = f"[Source {i}: TS {c.get('spec_number','?')} §{c.get('section_path','?')} | {c.get('release','?')}]"
        context_parts.append(f"{header}\n{c['chunk_text']}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"Context from 3GPP specifications:\n\n{context}\n\nQuestion: {req.query}\n\nAnswer:"

    resp = bedrock.invoke_model(
        modelId=LLM_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "system": GENERATOR_SYSTEM,
            "messages": [{"role": "user", "content": prompt}]
        }),
        contentType="application/json",
        accept="application/json"
    )
    answer_raw = json.loads(resp["body"].read())["content"][0]["text"]

    # Extract confidence
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
        Citation(
            spec=c.get("spec_number"),
            section=c.get("section_path"),
            release=c.get("release"),
            score=round(c.get("score", 0), 3)
        )
        for c in hits[:5]
    ]

    latency = int((time.time() - start) * 1000)

    return QueryResponse(
        answer=answer_raw,
        citations=citations,
        confidence=confidence,
        latency_ms=latency,
        chunks_retrieved=len(hits)
    )
