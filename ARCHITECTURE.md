# Architecture — 3GPP RAG Pipeline

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        3GPP RAG Pipeline                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │  INGEST  │───▶│  PARSE   │───▶│  EMBED   │───▶│  QUERY ENGINE    │  │
│  │  (HTTPS) │    │  & CHUNK │    │  & INDEX │    │  (LangGraph)     │  │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────────┬─────────┘  │
│       │               │               │                    │            │
│       ▼               ▼               ▼                    ▼            │
│  ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │   S3    │    │   S3    │    │PostgreSQL│    │ Amazon Bedrock   │   │
│  │Landing  │    │Processed│    │ pgvector │    │ Claude + Titan   │   │
│  └─────────┘    └─────────┘    └──────────┘    └──────────────────┘   │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────┐                                                            │
│  │DynamoDB │  (delta manifest)                                          │
│  └─────────┘                                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Pipeline Stages

### Stage 1: Ingest (`01_ftp_crawler.py`)
- Crawls 3GPP HTTPS endpoints for spec ZIPs
- Sources: TSGR2_129/129bis meeting docs, LS exchanges, Rel-18/19/20 specs
- Delta sync via DynamoDB (SHA-256 tracking, skip already-ingested files)
- Uploads raw ZIPs to S3 landing bucket

### Stage 2: Parse & Chunk (`02_process_docs.py`)
- Extracts .docx from ZIP files
- Parses document structure (headings, clauses, tables)
- Structure-aware chunking (512 tokens, 64 token overlap)
- Tables preserved as markdown within chunks
- Heading breadcrumbs injected for context
- Optional: Claude Haiku generates summary + keywords + hypothetical questions

### Stage 3: Embed & Index (`03_embed_and_index.py`)
- Amazon Titan Embed v2 generates 1024-dim vectors
- Hybrid indexing in PostgreSQL:
  - HNSW index for vector cosine similarity
  - GIN index for full-text search (tsvector)
- Batch processing (50 chunks per DB transaction)
- Auto-reconnect on connection timeout

### Stage 4: Query (`04_query.py`)
- LangGraph StateGraph with 4 nodes:
  1. **Planner**: Decomposes complex queries, extracts filters
  2. **Retriever**: Hybrid search (70% vector + 30% keyword)
  3. **Reranker**: Score-based deduplication, top-10 selection
  4. **Generator**: Claude Sonnet 4.5 with grounded citations

### Stage 5: Evaluate (`05_evaluate.py`)
- LLM-as-Judge scoring (faithfulness, relevancy, completeness)
- Golden Q&A set for regression testing
- JSON report generation

## AWS Services Used

| Service | Purpose | Tier |
|---------|---------|------|
| Amazon S3 | Raw + processed file storage | Standard |
| Amazon DynamoDB | Delta sync manifest | On-demand |
| Amazon RDS PostgreSQL | Vector + keyword search (pgvector) | db.t3.micro |
| Amazon Bedrock - Titan Embed v2 | 1024-dim embeddings | On-demand |
| Amazon Bedrock - Claude Sonnet 4.5 | Answer generation | Inference profile |
| Amazon Bedrock - Claude Haiku 4.5 | Metadata generation | Inference profile |

## Data Flow

```
3GPP HTTPS ──▶ S3 Landing ──▶ Parse ──▶ S3 Processed ──▶ Embed ──▶ PostgreSQL
                    │                                                     │
                    ▼                                                     ▼
               DynamoDB                                            Query Engine
            (delta tracking)                                    (LangGraph + Bedrock)
```

## Key Design Decisions

1. **pgvector over OpenSearch Serverless**: 20x cheaper ($0.50/day vs $11.52/day) for POC scale
2. **Hybrid search**: Combines semantic (vector) and lexical (BM25/tsvector) for better recall
3. **Structure-aware chunking**: Preserves 3GPP clause boundaries and table integrity
4. **Heading breadcrumbs**: Each chunk includes its section path for context
5. **No retry loops**: Single-pass retrieval for predictable latency
6. **Batch DB writes**: Fresh connection per batch prevents timeout on large files
