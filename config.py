"""
3GPP RAG POC — Central Configuration
=====================================
Edit PG_HOST and PG_PASSWORD after running 00_create_infra.py.

To deploy:
    1. Run: python 00_create_infra.py
    2. Copy the printed RDS endpoint here
    3. Set PG_PASSWORD to the value printed
"""

# ── AWS Region ────────────────────────────────────────────────────────────────
AWS_REGION = "us-east-1"

# ── S3 Buckets ────────────────────────────────────────────────────────────────
S3_LANDING_BUCKET   = "3gpp-rag-landing"       # Raw downloaded spec files
S3_PROCESSED_BUCKET = "3gpp-rag-processed"     # Parsed chunk JSONL files

# ── DynamoDB ──────────────────────────────────────────────────────────────────
DYNAMODB_TABLE = "3gpp-rag-manifest"            # Delta-sync tracking table

# ── PostgreSQL + pgvector ─────────────────────────────────────────────────────
# Fill these in after running 00_create_infra.py
PG_HOST     = "tgpp-rag-poc.cc9k6aouw4cy.us-east-1.rds.amazonaws.com"
PG_PORT     = 5432
PG_DATABASE = "ragdb"
PG_USER     = "ragadmin"
PG_PASSWORD = "Change_Me_123!"
PG_TABLE    = "chunks"

# ── Bedrock Models ────────────────────────────────────────────────────────────
# Titan Embed v2 — 1024-dim embeddings (on-demand, no inference profile needed)
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"

# Claude Sonnet 4.5 — answer generation (requires inference profile)
LLM_MODEL_ID   = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# Claude Haiku 4.5 — metadata generation (requires inference profile)
HAIKU_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Embedding dimension (Titan v2 default output)
EMBED_DIM      = 1024

# ── HTTPS Ingest Sources ──────────────────────────────────────────────────────
FTP_HOST      = "ftp.3gpp.org"
FTP_BASE_PATH = "/Specs/latest_agreement/"
TARGET_SERIES = ["38.3", "38.4", "38.5", "38.1"]
MAX_FILES_POC = 20   # Cap for POC — increase or remove for full ingest

# ── Chunking Parameters ───────────────────────────────────────────────────────
MAX_CHUNK_TOKENS     = 512   # Maximum tokens per chunk
CHUNK_OVERLAP_TOKENS = 64    # Overlap between adjacent chunks (context preservation)
MIN_CHUNK_TOKENS     = 50    # Discard fragments smaller than this
