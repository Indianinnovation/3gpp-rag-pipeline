"""
04_query.py
===========
LangGraph StateGraph reasoning engine for 3GPP RAG.

Architecture:
  User query
      ↓
  [planner]  — decomposes complex queries, extracts spec/release filters
      ↓
  [retriever] — hybrid AOSS search (from 03_embed_and_index.py)
      ↓
  [reranker]  — score + deduplicate chunks
      ↓  
  [generator] — Claude 3.5 Sonnet with retrieved context
      ↓
  [router]   — sufficient | needs_more_context | low_confidence
      ↓
  Answer + citations

Usage (CLI):
    python 04_query.py
    python 04_query.py --query "What are the RRC states in NR?"
    python 04_query.py --query "..." --spec 38300 --release Rel-17
    python 04_query.py --stream   # token-by-token streaming

Importable as a module:
    from query import create_engine, run_query
    engine = create_engine()
    result = run_query(engine, "What is the DSON energy saving procedure?")
"""

import argparse
import json
import re
from typing import Annotated, TypedDict, Optional

import boto3
from langgraph.graph import StateGraph, END

from config import AWS_REGION, LLM_MODEL_ID
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create symlink for module import if needed
embed_link = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embed_and_index.py")
if not os.path.exists(embed_link):
    os.symlink("03_embed_and_index.py", embed_link)

from embed_and_index import hybrid_search, get_pg_conn

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)


# ─────────────────────────────────────────────────────────────────────────────
# State definition
# ─────────────────────────────────────────────────────────────────────────────
class RAGState(TypedDict):
    # Input
    user_query:      str
    spec_filter:     Optional[str]
    release_filter:  Optional[str]
    # Planner output
    sub_queries:     list[str]
    extracted_spec:  Optional[str]
    extracted_rel:   Optional[str]
    # Retrieval output
    retrieved_chunks: list[dict]
    retrieval_hops:   int
    # Generation output
    answer:          str
    citations:       list[dict]
    confidence:      float
    # Routing
    route:           str     # "done" | "retry" | "low_confidence"
    hop_count:       int


# ─────────────────────────────────────────────────────────────────────────────
# Node: Planner
# ─────────────────────────────────────────────────────────────────────────────
PLANNER_SYSTEM = """You are a 3GPP standards query planner. Given a user question:
1. Extract any spec number mentioned (e.g. 38300, 38331, 36331)
2. Extract any 3GPP Release mentioned (e.g. Rel-17, Release 18)
3. If the question is complex, decompose into 1-3 focused sub-queries
4. Return ONLY valid JSON: {"spec": "38300"|null, "release": "Rel-17"|null, "sub_queries": ["..."]}
No markdown fences."""


def planner_node(state: RAGState) -> dict:
    print("  [planner] Decomposing query …")
    resp = bedrock.invoke_model(
        modelId=LLM_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "system": PLANNER_SYSTEM,
            "messages": [{"role": "user", "content": state["user_query"]}]
        }),
        contentType="application/json",
        accept="application/json"
    )
    raw = json.loads(resp["body"].read())["content"][0]["text"].strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw)

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        plan = {"spec": None, "release": None, "sub_queries": [state["user_query"]]}

    return {
        "extracted_spec": plan.get("spec") or state.get("spec_filter"),
        "extracted_rel":  plan.get("release") or state.get("release_filter"),
        "sub_queries":    plan.get("sub_queries") or [state["user_query"]]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Retriever
# ─────────────────────────────────────────────────────────────────────────────
def retriever_node(state: RAGState, conn) -> dict:
    print("  [retriever] Searching …")
    all_chunks: list[dict] = []
    seen_ids: set[str] = {c["chunk_id"] for c in state.get("retrieved_chunks", [])}

    for q in state["sub_queries"]:
        hits = hybrid_search(
            query=q,
            conn=conn,
            top_k=8,
            spec_filter=state.get("spec_filter"),
            release_filter=state.get("release_filter")
        )
        for h in hits:
            if h["chunk_id"] not in seen_ids:
                all_chunks.append(h)
                seen_ids.add(h["chunk_id"])

    existing = state.get("retrieved_chunks", [])
    return {
        "retrieved_chunks": existing + all_chunks,
        "retrieval_hops":   state.get("retrieval_hops", 0) + 1
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Reranker (score + deduplicate + trim)
# ─────────────────────────────────────────────────────────────────────────────
def reranker_node(state: RAGState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    # Sort by AOSS score descending
    chunks = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)
    # Deduplicate on chunk_id (safety net)
    seen: set[str] = set()
    deduped = []
    for c in chunks:
        if c["chunk_id"] not in seen:
            deduped.append(c)
            seen.add(c["chunk_id"])
    # Keep top 10 for context window efficiency
    return {"retrieved_chunks": deduped[:10]}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Generator
# ─────────────────────────────────────────────────────────────────────────────
GENERATOR_SYSTEM = """You are a 3GPP standards expert assistant with deep knowledge of
5G NR, LTE, and O-RAN specifications. Answer the question using ONLY the provided context
from official 3GPP specifications. 

Rules:
- Cite specific clause numbers and spec numbers (e.g. "per TS 38.300 §5.3.2")
- If the context does not contain sufficient information, say so explicitly
- Never hallucinate spec content
- Use precise 3GPP terminology (IEs, procedures, protocol states)
- End your answer with a JSON block: {"confidence": 0.0-1.0, "needs_more_context": true/false}
  where confidence reflects how well the retrieved context covers the question."""


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        header = f"[Source {i}: TS {c.get('spec_number','?')} §{c.get('section_path','?')} | {c.get('release','?')}]"
        parts.append(f"{header}\n{c['chunk_text']}")
    return "\n\n---\n\n".join(parts)


def generator_node(state: RAGState) -> dict:
    print("  [generator] Generating answer …")
    context  = format_context(state["retrieved_chunks"])
    question = state["user_query"]

    prompt = f"""Context from 3GPP specifications:

{context}

Question: {question}

Answer based strictly on the provided context:"""

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

    # Extract confidence JSON from end of answer
    confidence = 0.5
    needs_more = False
    json_match = re.search(r'\{[^{}]*"confidence"[^{}]*\}', answer_raw)
    if json_match:
        try:
            meta = json.loads(json_match.group())
            confidence  = float(meta.get("confidence", 0.5))
            needs_more  = bool(meta.get("needs_more_context", False))
            answer_raw  = answer_raw[:json_match.start()].strip()
        except (json.JSONDecodeError, ValueError):
            pass

    # Build citation list
    citations = [
        {
            "spec": c.get("spec_number"),
            "section": c.get("section_path"),
            "release": c.get("release"),
            "score": round(c.get("score", 0), 3)
        }
        for c in state["retrieved_chunks"][:5]
    ]

    return {
        "answer":     answer_raw,
        "citations":  citations,
        "confidence": confidence,
        "route":      "retry" if needs_more else "done"
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Router (conditional edge logic)
# ─────────────────────────────────────────────────────────────────────────────
def router_node(state: RAGState) -> str:
    return "done"   # no retries for faster response


# ─────────────────────────────────────────────────────────────────────────────
# Build the LangGraph StateGraph
# ─────────────────────────────────────────────────────────────────────────────
def create_engine(conn=None) -> "CompiledGraph":
    if conn is None:
        conn = get_pg_conn()

    graph = StateGraph(RAGState)

    # Register nodes
    graph.add_node("planner",   planner_node)
    graph.add_node("retriever", lambda s: retriever_node(s, conn))
    graph.add_node("reranker",  reranker_node)
    graph.add_node("generator", generator_node)

    # Entry point
    graph.set_entry_point("planner")

    # Edges
    graph.add_edge("planner",   "retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker",  "generator")

    # Conditional routing after generator
    graph.add_conditional_edges(
        "generator",
        router_node,
        {
            "retry":           "retriever",   # widen search
            "low_confidence":  END,           # surface low-conf answer anyway
            "done":            END
        }
    )

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Run a query and return structured result
# ─────────────────────────────────────────────────────────────────────────────
def run_query(engine, query: str, spec: str = None,
              release: str = None) -> dict:
    initial_state: RAGState = {
        "user_query":       query,
        "spec_filter":      spec,
        "release_filter":   release,
        "sub_queries":      [],
        "extracted_spec":   spec,
        "extracted_rel":    release,
        "retrieved_chunks": [],
        "retrieval_hops":   0,
        "answer":           "",
        "citations":        [],
        "confidence":       0.0,
        "route":            "",
        "hop_count":        0
    }
    final_state = engine.invoke(initial_state)
    return {
        "answer":     final_state["answer"],
        "citations":  final_state["citations"],
        "confidence": final_state["confidence"],
        "hops":       final_state["retrieval_hops"]
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI interface
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
    "Explain the DSON energy saving procedure and which nodes are involved",
    "What is the difference between SA and NSA deployment architectures?",
    "How does the PRB uplink blanking mechanism work for coexistence?",
    "Describe the 5G NR handover procedure and the messages exchanged"
]


def interactive_cli(engine):
    print("\n3GPP RAG — Interactive Query CLI")
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

    print("Initializing engine …")
    engine = create_engine()
    print("✓ LangGraph engine ready\n")

    if args.query:
        result = run_query(engine, args.query, args.spec, args.release)
        print_result(result)
    else:
        interactive_cli(engine)


if __name__ == "__main__":
    main()
