"""
SSE Streaming handler for 3GPP RAG API.
Streams pipeline steps and generator tokens in real-time via Server-Sent Events.

Usage:
    uvicorn sse_handler:app --reload --port 8000
"""

import json
import re
import time
import hashlib
import boto3
import psycopg2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
import os

# Import everything from the main handler
from handler import (
    get_pg_conn, embed_text, embed_texts_batch, _embed_cache,
    hybrid_search, hybrid_search_with_embedding,
    reciprocal_rank_fusion, normalize_rrf_scores, deduplicate_by_clause,
    classify_query, direct_lookup, expand_query, run_planner,
    isrel_filter, rerank_chunks, get_release_history,
    CAUSE_CODE_SPECS, CAUSE_CLAUSE_BLACKLIST, SECTION_BLACKLIST_PATTERNS,
    SPEC_BLACKLIST_FOR_CAUSE, _RETRIEVAL_EXECUTOR,
    AWS_REGION, PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD,
    PG_TABLE, LLM_MODEL_ID, HAIKU_MODEL_ID, NOVA_PRO_MODEL_ID, EMBED_MODEL_ID,
    GENERATOR_SYSTEM
)

app = FastAPI(title="3GPP RAG SSE API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=Config(read_timeout=120))


class QueryRequest(BaseModel):
    query: str
    spec_filter: Optional[str] = None
    release_filter: Optional[str] = None


def sse_event(event_type: str, data: dict) -> str:
    """Format a single SSE message."""
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


@app.get("/health")
def health():
    return {"status": "ok", "model": LLM_MODEL_ID, "version": "v13-sse"}


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """SSE streaming endpoint — streams pipeline steps and answer tokens."""

    async def generate() -> AsyncGenerator[str, None]:
        start = time.time()
        query = req.query

        # ── Step 1: Planner ──────────────────────────────
        yield sse_event("step_start", {
            "step": "planner",
            "label": "Decomposing query",
            "icon": "🧠",
            "detail": "Breaking into targeted sub-queries..."
        })

        t1 = time.time()
        sub_queries = run_planner(query)
        yield sse_event("step_done", {
            "step": "planner",
            "label": f"Planner: {len(sub_queries)} sub-queries",
            "ms": int((time.time() - t1) * 1000),
            "detail": f"Generated {len(sub_queries)} focused retrieval queries"
        })

        # ── Step 2: Router / RRF retrieval ───────────────
        yield sse_event("step_start", {
            "step": "router",
            "label": "Retrieving from 3GPP index",
            "icon": "🔍",
            "detail": f"Running {len(sub_queries)} sub-queries + targeted spec searches..."
        })

        t2 = time.time()
        query_type = classify_query(query)

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
            # Build search tasks
            search_tasks = []
            for q in sub_queries:
                search_tasks.append((q, 10, req.spec_filter, req.release_filter))

            if not req.spec_filter:
                cause_kws = ["cause", "clear code", "release cause", "reject", "failure"]
                if any(kw in query.lower() for kw in cause_kws):
                    for spec in CAUSE_CODE_SPECS:
                        search_tasks.append((query, 8, spec, None))

            # Parallel embed + search
            unique_queries = list(set(t[0] for t in search_tasks))
            embed_texts_batch(unique_queries)

            def _do_search(args):
                q, top_k, spec, release = args
                q_emb = _embed_cache[q[:8000].strip()]
                return hybrid_search_with_embedding(q, q_emb, top_k, spec_filter=spec, release_filter=release)

            futures = [_RETRIEVAL_EXECUTOR.submit(_do_search, t) for t in search_tasks]
            result_sets = [f.result() for f in futures]
            result_sets = [rs for rs in result_sets if rs]

            # Apply blacklist before RRF
            is_cause_query = any(kw in query.lower() for kw in ["cause", "clear code", "failure", "reject"])
            if is_cause_query:
                for i, rs in enumerate(result_sets):
                    result_sets[i] = [
                        c for c in rs
                        if not any(bl in (c.get("section_path") or "") for bl in CAUSE_CLAUSE_BLACKLIST)
                        and c.get("section_path", "") != "auto-ingested"
                        and not any(pat in (c.get("section_path") or "").lower() for pat in SECTION_BLACKLIST_PATTERNS)
                        and c.get("spec_number", "") not in SPEC_BLACKLIST_FOR_CAUSE
                    ]

            num_sets = len(result_sets)
            all_chunks = reciprocal_rank_fusion(result_sets)
            all_chunks = normalize_rrf_scores(all_chunks)
            all_chunks = deduplicate_by_clause(all_chunks)

        yield sse_event("step_done", {
            "step": "router",
            "label": f"RRF → {len(all_chunks)} chunks from {num_sets} sets",
            "ms": int((time.time() - t2) * 1000)
        })

        # ── Step 3: IsREL filter ──────────────────────────
        yield sse_event("step_start", {
            "step": "isrel",
            "label": "IsREL relevance filter",
            "icon": "🎯",
            "detail": "Scoring each chunk for answerability..."
        })

        t3 = time.time()
        filtered = isrel_filter(all_chunks, query)
        discarded = len(all_chunks) - len(filtered)
        yield sse_event("step_done", {
            "step": "isrel",
            "label": f"IsREL: {len(filtered)} relevant, {discarded} discarded",
            "ms": int((time.time() - t3) * 1000),
            "sources_preview": [
                {"spec": c.get("spec_number", "?"), "clause": c.get("section_path", "?"), "score": round(c.get("score", 0), 4)}
                for c in filtered[:4]
            ]
        })

        # ── Step 4: Rerank + CRAG ─────────────────────────
        yield sse_event("step_start", {
            "step": "crag",
            "label": "CRAG evaluation",
            "icon": "⚖️",
            "detail": "Checking retrieval quality..."
        })

        t4 = time.time()
        selected = rerank_chunks(filtered)

        # CRAG threshold
        CRAG_THRESHOLD = 0.05
        CRAG_MIN_PASS = 5
        relevant = [c for c in selected if c.get("score", 0) >= CRAG_THRESHOLD]
        if len(relevant) < CRAG_MIN_PASS:
            relevant = sorted(selected, key=lambda x: -x.get("score", 0))[:CRAG_MIN_PASS]
        verdict = "correct" if len(relevant) >= 3 else "ambiguous" if relevant else "incorrect"

        yield sse_event("step_done", {
            "step": "crag",
            "label": f"CRAG: {verdict} ({len(relevant)} relevant)",
            "ms": int((time.time() - t4) * 1000),
            "verdict": verdict
        })

        # ── Step 5: Generator (token streaming) ──────────
        yield sse_event("step_start", {
            "step": "generator",
            "label": "Generating answer",
            "icon": "✍️",
            "detail": f"Grounding from {len(relevant)} verified sources..."
        })

        # Build context
        context_parts = []
        for i, c in enumerate(relevant[:15], 1):
            header = f"[Source {i}: TS {c['spec_number']} §{c['section_path']} | {c['release']}]"
            context_parts.append(f"{header}\n{c['chunk_text']}")
        context = "\n\n---\n\n".join(context_parts)

        # Release history for cause queries
        release_kws = ["when", "added", "introduced", "release", "latest", "history", "cause", "clear code", "classification"]
        if any(kw in query.lower() for kw in release_kws):
            history = get_release_history(req.spec_filter)
            if history:
                context += f"\n\n---\n\n[Source: CR Change History Database]{history}"

        prompt = f"Context from 3GPP specifications:\n\n{context}\n\n---\n\nQuestion: {query}\n\nProduce a comprehensive, reference-quality answer:"

        # Stream from Bedrock with response_stream
        try:
            response = bedrock.invoke_model_with_response_stream(
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

            full_answer = ""
            for event in response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                if chunk["type"] == "content_block_delta":
                    token = chunk["delta"].get("text", "")
                    if token:
                        full_answer += token
                        yield sse_event("token", {"text": token})

        except Exception as e:
            # Fallback to non-streaming
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
            full_answer = json.loads(resp["body"].read())["content"][0]["text"]
            # Send in chunks to simulate streaming
            words = full_answer.split(" ")
            for i in range(0, len(words), 3):
                chunk_text = " ".join(words[i:i+3]) + " "
                yield sse_event("token", {"text": chunk_text})

        # Extract confidence
        confidence = 0.5
        json_match = re.search(r'\{[^{}]*"confidence"[^{}]*\}', full_answer)
        if json_match:
            try:
                meta = json.loads(json_match.group())
                confidence = float(meta.get("confidence", 0.5))
            except (json.JSONDecodeError, ValueError):
                pass

        # ── Step 6: Done ──────────────────────────────────
        sources = [
            {
                "spec": c.get("spec_number", "?"),
                "clause": c.get("section_path", "?"),
                "release": c.get("release", "?"),
                "score": round(c.get("score", 0), 4)
            }
            for c in relevant[:8]
        ]

        yield sse_event("done", {
            "sources": sources,
            "confidence": confidence,
            "total_ms": int((time.time() - start) * 1000),
            "chunks_used": len(relevant)
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


# Keep the original non-streaming endpoint for backward compatibility
from handler import app as original_app, handler

# Mount original endpoints
for route in original_app.routes:
    if hasattr(route, 'path') and route.path not in ('/query/stream', '/health'):
        app.routes.append(route)
