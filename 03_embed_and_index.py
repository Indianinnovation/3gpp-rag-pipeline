"""
03_embed_and_index.py
=====================
Reads chunk JSONL files from S3, generates embeddings via Titan Embed v2,
and indexes to PostgreSQL with pgvector for hybrid vector + keyword search.

Usage:
    python 03_embed_and_index.py
    python 03_embed_and_index.py --spec 38300
    python 03_embed_and_index.py --reindex
"""

import argparse
import json
import time

import boto3
import psycopg2
from psycopg2.extras import execute_values

from config import (
    AWS_REGION, S3_PROCESSED_BUCKET,
    PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD, PG_TABLE,
    EMBED_MODEL_ID, EMBED_DIM
)

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
s3      = boto3.client("s3",             region_name=AWS_REGION)

BATCH_SIZE = 25
INDEX_BATCH = 50  # chunks per DB insert batch


# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL connection
# ─────────────────────────────────────────────────────────────────────────────
def get_pg_conn():
    if not PG_HOST:
        raise ValueError("PG_HOST is empty — update config.py after running 00_create_infra.py")
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DATABASE,
        user=PG_USER, password=PG_PASSWORD
    )


# ─────────────────────────────────────────────────────────────────────────────
# Titan Embed v2
# ─────────────────────────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        # Titan Embed requires non-empty input
        clean = text.strip()[:8000] if text else "empty"
        if not clean:
            clean = "empty"
        try:
            resp = bedrock.invoke_model(
                modelId=EMBED_MODEL_ID,
                body=json.dumps({
                    "inputText": clean
                }),
                contentType="application/json",
                accept="application/json"
            )
            embeddings.append(json.loads(resp["body"].read())["embedding"])
        except Exception as e:
            print(f"    ⚠ Embed error, using zero vector: {e}")
            embeddings.append([0.0] * EMBED_DIM)
        if len(embeddings) % BATCH_SIZE == 0:
            time.sleep(0.3)
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# S3 chunk reader
# ─────────────────────────────────────────────────────────────────────────────
def list_chunk_keys(spec_filter: str = None) -> list[str]:
    prefix = f"chunks/{spec_filter}/" if spec_filter else "chunks/"
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=S3_PROCESSED_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".jsonl"):
                keys.append(obj["Key"])
    return keys


def read_chunks_from_s3(s3_key: str) -> list[dict]:
    obj = s3.get_object(Bucket=S3_PROCESSED_BUCKET, Key=s3_key)
    lines = obj["Body"].read().decode().strip().split("\n")
    return [json.loads(l) for l in lines if l.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# BUG 6 FIX: Metadata completeness validator
# Rejects chunks with missing/invalid clause metadata at index time.
# Chunks with clause="auto-ingested" or empty section_path cannot be
# evaluated by IsREL or CRAG, causing unvalidatable noise in results.
# ─────────────────────────────────────────────────────────────────────────────
REJECTED_CHUNKS_LOG = "rejected_chunks.jsonl"


def validate_chunk_metadata(chunk: dict) -> bool:
    """Returns False if chunk has missing/invalid clause metadata.
    
    BUG 6 FIX: Prevents indexing chunks that CRAG and IsREL cannot evaluate.
    Rejected chunks are logged to rejected_chunks.jsonl for manual review.
    """
    section_path = (chunk.get("section_path") or "").strip()
    
    # Reject if section_path is empty, null, or "auto-ingested"
    if not section_path:
        return False
    if section_path.lower() in ("auto-ingested", "unknown", "none"):
        return False
    # Reject if section_path is just a number with no title
    if section_path.isdigit():
        return False
    return True


def log_rejected_chunk(chunk: dict, reason: str) -> None:
    """Append rejected chunk info to log file for manual review."""
    import os
    entry = {
        "chunk_id": chunk.get("chunk_id", "?"),
        "spec_number": chunk.get("spec_number", "?"),
        "section_path": chunk.get("section_path", "?"),
        "reason": reason,
        "text_preview": chunk.get("chunk_text", "")[:100]
    }
    with open(REJECTED_CHUNKS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL indexing
# ─────────────────────────────────────────────────────────────────────────────
def index_chunks(conn, chunks: list[dict]) -> tuple[int, int]:
    # Filter out empty chunks
    chunks = [c for c in chunks if c.get("chunk_text", "").strip()]
    
    # BUG 6 FIX: Validate metadata completeness before indexing
    valid_chunks = []
    for c in chunks:
        if validate_chunk_metadata(c):
            valid_chunks.append(c)
        else:
            log_rejected_chunk(c, "invalid_section_path")
    
    if valid_chunks != chunks:
        rejected_count = len(chunks) - len(valid_chunks)
        print(f"    ⚠ Rejected {rejected_count} chunks with invalid metadata (logged to {REJECTED_CHUNKS_LOG})")
    chunks = valid_chunks
    
    if not chunks:
        return 0, 0

    total_ok = total_fail = 0
    for batch_start in range(0, len(chunks), INDEX_BATCH):
        batch = chunks[batch_start:batch_start + INDEX_BATCH]
        texts = [c["chunk_text"] for c in batch]
        embeddings = embed_texts(texts)

        rows = []
        for chunk, emb in zip(batch, embeddings):
            rows.append((
                chunk["chunk_id"],
                chunk["doc_id"],
                chunk.get("spec_series", ""),
                chunk.get("spec_number", ""),
                chunk.get("release", ""),
                chunk.get("section_path", ""),
                chunk.get("doc_type", "TS"),
                chunk["chunk_text"],
                chunk.get("summary", ""),
                chunk.get("keywords", []),
                chunk.get("hyp_questions", []),
                chunk.get("token_count", 0),
                chunk.get("source_s3_key", ""),
                str(emb)
            ))

        # Fresh connection per batch to avoid timeout
        try:
            batch_conn = get_pg_conn()
            cur = batch_conn.cursor()
            execute_values(cur, f"""
                INSERT INTO {PG_TABLE}
                (chunk_id, doc_id, spec_series, spec_number, release, section_path,
                 doc_type, chunk_text, summary, keywords, hyp_questions,
                 token_count, source_s3_key, embedding)
                VALUES %s
                ON CONFLICT (chunk_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    chunk_text = EXCLUDED.chunk_text,
                    indexed_at = NOW()
            """, rows)
            batch_conn.commit()
            batch_conn.close()
            total_ok += len(rows)
        except Exception as e:
            print(f"    ⚠ Batch error: {e}")
            total_fail += len(rows)

    return total_ok, total_fail


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid search (vector + keyword) — used by 04_query.py
# ─────────────────────────────────────────────────────────────────────────────
def hybrid_search(
    query: str,
    conn=None,
    top_k: int = 10,
    spec_filter: str = None,
    release_filter: str = None
) -> list[dict]:
    if conn is None:
        conn = get_pg_conn()

    query_emb = embed_texts([query])[0]

    conditions = []
    params = []
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
               COALESCE(ts_rank(to_tsvector('english', chunk_text), plainto_tsquery('english', %s)), 0) AS text_score
        FROM {PG_TABLE}
        {where}
        ORDER BY (0.7 * (1 - (embedding <=> %s::vector)) +
                  0.3 * COALESCE(ts_rank(to_tsvector('english', chunk_text), plainto_tsquery('english', %s)), 0)) DESC
        LIMIT %s
    """
    all_params = [str(query_emb), query] + params + [str(query_emb), query, top_k]

    cur = conn.cursor()
    cur.execute(sql, all_params)
    rows = cur.fetchall()

    return [
        {
            "chunk_id":     r[0],
            "spec_number":  r[1],
            "release":      r[2],
            "section_path": r[3],
            "doc_type":     r[4],
            "chunk_text":   r[5],
            "summary":      r[6],
            "score":        round(0.7 * r[7] + 0.3 * r[8], 4)
        }
        for r in rows
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec",    type=str, default=None)
    parser.add_argument("--reindex", action="store_true")
    args = parser.parse_args()

    conn = get_pg_conn()
    print("✓ Connected to PostgreSQL")

    if args.reindex:
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {PG_TABLE}")
        cur.execute(f"""
            CREATE TABLE {PG_TABLE} (
                chunk_id TEXT PRIMARY KEY, doc_id TEXT, spec_series TEXT,
                spec_number TEXT, release TEXT, section_path TEXT,
                doc_type TEXT DEFAULT 'TS', chunk_text TEXT, summary TEXT,
                keywords TEXT[], hyp_questions TEXT[], token_count INTEGER,
                source_s3_key TEXT, embedding vector({EMBED_DIM}),
                indexed_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        cur.execute(f"""CREATE INDEX chunks_embedding_idx ON {PG_TABLE}
                       USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);""")
        cur.execute(f"""CREATE INDEX chunks_text_search_idx ON {PG_TABLE}
                       USING gin (to_tsvector('english', chunk_text));""")
        conn.commit()
        print("  ✓ Table recreated")

    keys = list_chunk_keys(spec_filter=args.spec)
    print(f"Found {len(keys)} chunk JSONL files to index\n")

    total_ok = total_fail = 0
    for i, key in enumerate(keys, 1):
        chunks = read_chunks_from_s3(key)
        print(f"[{i}/{len(keys)}] {key} — {len(chunks)} chunks")
        try:
            ok, fail = index_chunks(conn, chunks)
        except Exception as e:
            print(f"    ⚠ Connection lost, reconnecting: {e}")
            conn = get_pg_conn()
            ok, fail = index_chunks(conn, chunks)
        total_ok += ok
        total_fail += fail
        print(f"  ✓ indexed {ok}, failed {fail}")

    print(f"\n── Indexing complete ───────────────────────────────────────────")
    print(f"  Total indexed : {total_ok}")
    print(f"  Total failed  : {total_fail}")

    # Sanity search
    print("\n── Sanity search ───────────────────────────────────────────────")
    conn = get_pg_conn()
    hits = hybrid_search("RRC state machine idle connected", conn, top_k=3)
    for h in hits:
        print(f"  [{h['score']:.3f}] §{h['section_path']} | TS{h['spec_number']} {h['release']}")
        print(f"    {h['chunk_text'][:120]}…\n")

    conn.close()
    print("Next step: python 04_query.py")


if __name__ == "__main__":
    main()
