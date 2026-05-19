# 3GPP RAG Pipeline — AWS GenAI Solution

> **Full-pipeline Retrieval-Augmented Generation (RAG) for 3GPP specifications**  
> `HTTPS ingest → parse → chunk → embed → index → LangGraph query`

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![AWS](https://img.shields.io/badge/AWS-Bedrock%20%7C%20RDS%20%7C%20S3-orange.svg)](https://aws.amazon.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---
<img width="1917" height="933" alt="image" src="https://github.com/user-attachments/assets/a24c8350-832a-4ec9-a3ca-1206f2d3c0c7" />

## 🏗️ Architecture

```
┌────────────┐     ┌────────────┐     ┌────────────┐     ┌─────────────────┐
│   INGEST   │────▶│   PARSE    │────▶│   EMBED    │────▶│  QUERY ENGINE   │
│  (HTTPS)   │     │  & CHUNK   │     │  & INDEX   │     │  (LangGraph)    │
└─────┬──────┘     └─────┬──────┘     └─────┬──────┘     └────────┬────────┘
      │                   │                  │                     │
      ▼                   ▼                  ▼                     ▼
┌─────────┐        ┌─────────┐       ┌──────────┐       ┌────────────────┐
│   S3    │        │   S3    │       │PostgreSQL│       │Amazon Bedrock  │
│ Landing │        │Processed│       │ pgvector │       │Claude + Titan  │
└─────────┘        └─────────┘       └──────────┘       └────────────────┘
      │
      ▼
┌─────────┐
│DynamoDB │ (delta tracking)
└─────────┘
```

## 🔧 AWS Services Used

| Service | Purpose | Cost (POC) |
|---------|---------|------------|
| Amazon Bedrock — Titan Embed v2 | 1024-dim vector embeddings | ~$0.15 |
| Amazon Bedrock — Claude Sonnet 4.5 | Answer generation | ~$0.50 |
| Amazon Bedrock — Claude Haiku 4.5 | Metadata generation (optional) | ~$0.30 |
| Amazon RDS PostgreSQL + pgvector | Hybrid vector + keyword search | ~$0.50/day |
| Amazon S3 | Raw + processed file storage | ~$0.05 |
| Amazon DynamoDB | Delta sync manifest | ~$0.01 |
| **Total POC** | | **~$2–5** |

---

## 📋 Prerequisites

### 1. AWS Account Setup

- AWS account with billing enabled
- IAM user or role with permissions for: S3, DynamoDB, RDS, Bedrock, EC2 (security groups)
- **Bedrock model access enabled** in `us-east-1` for:
  - `amazon.titan-embed-text-v2:0`
  - `anthropic.claude-haiku-4-5-20251001-v1:0` (via inference profile)
  - `anthropic.claude-sonnet-4-5-20250929-v1:0` (via inference profile)

> **How to enable Bedrock models**: AWS Console → Amazon Bedrock → Model access → Request access

### 2. Local Environment

```bash
# Python 3.11+ required
python --version

# Install AWS CLI
brew install awscli   # macOS
# or: pip install awscli

# Configure credentials
aws configure
# AWS Access Key ID: <your-key>
# AWS Secret Access Key: <your-secret>
# Default region: us-east-1
# Default output: json

# Verify access
aws sts get-caller-identity
```

---

## 🚀 Quick Start (5 Steps)

### Step 0 — Clone & Install

```bash
git clone https://github.com/<your-username>/3gpp-rag-pipeline.git
cd 3gpp-rag-pipeline

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy config template
cp config.py.example config.py
```

### Step 1 — Create Infrastructure (~10 min)

```bash
python 00_create_infra.py
```

This creates:
- S3 buckets: `3gpp-rag-landing`, `3gpp-rag-processed`
- DynamoDB table: `3gpp-rag-manifest`
- Security group for PostgreSQL access
- RDS PostgreSQL instance with pgvector extension
- Chunks table with HNSW + GIN indexes

**After completion**, update `config.py` with the printed RDS endpoint and password.

### Step 2 — Ingest 3GPP Specs (~2–5 min)

```bash
# Preview what will be downloaded
python 01_ftp_crawler.py --dry-run

# Download specs (default: 20 files for POC)
python 01_ftp_crawler.py --limit 20

# Download only meeting docs or specs
python 01_ftp_crawler.py --source meetings --limit 50
python 01_ftp_crawler.py --source specs --limit 20
```

Sources crawled:
- TSGR2_129 / 129bis meeting documents + LS exchanges
- Rel-18, Rel-19, Rel-20 NR specifications (38-series)

### Step 3 — Parse & Chunk (~1–2 min with --skip-metadata)

```bash
# Fast mode (recommended for POC)
python 02_process_docs.py --limit 10 --skip-metadata

# With AI-generated metadata (slower: ~1-2 sec per chunk)
python 02_process_docs.py --limit 5
```

What happens per spec:
1. Unzip → find .docx inside
2. Extract sections via heading styles + clause number detection
3. Structure-aware chunking (512 tokens, 64 token overlap, table-aware)
4. (Optional) Claude Haiku generates: summary + keywords + hypothetical questions
5. Chunk JSONL uploaded to `s3://3gpp-rag-processed/chunks/`

### Step 4 — Embed & Index (~5–15 min)

```bash
python 03_embed_and_index.py

# Rebuild index from scratch
python 03_embed_and_index.py --reindex
```

- Titan Embed v2 generates 1024-dim vectors
- Hybrid PostgreSQL index: HNSW cosine + GIN full-text
- Batch processing (50 chunks per transaction) with auto-reconnect

### Step 5 — Query

```bash
# Single query
python 04_query.py --query "What are the RRC states in 5G NR?"

# With filters
python 04_query.py --query "Explain HARQ process" --spec 38213

# Interactive CLI
python 04_query.py
```

**LangGraph engine flow:**
```
User query
  → [planner]    extracts spec/release, decomposes complex queries
  → [retriever]  hybrid pgvector search (vector 70% + keyword 30%)
  → [reranker]   deduplicate + top-10 by score
  → [generator]  Claude Sonnet 4.5 with grounded citations
  → Answer + citations + confidence score
```

---

## 📊 Evaluation

```bash
# Run golden Q&A set (5 questions)
python 05_evaluate.py

# Single question evaluation
python 05_evaluate.py --single "What is PRB uplink blanking?"

# Custom golden set
python 05_evaluate.py --golden my_golden_qa.json
```

| Metric | Target | Description |
|--------|--------|-------------|
| Faithfulness | ≥ 0.80 | Answer only contains claims from context |
| Relevancy | ≥ 0.75 | Answer directly addresses the question |
| Completeness | ≥ 0.70 | Answer covers the ground truth |
| Avg confidence | ≥ 0.70 | Model self-assessed confidence |

---

## 📁 Project Structure

```
3gpp-rag-pipeline/
├── config.py.example      ← Template config (copy to config.py)
├── config.py              ← Your config with secrets (git-ignored)
├── requirements.txt       ← Python dependencies
├── 00_create_infra.py     ← One-time AWS infrastructure setup
├── 01_ftp_crawler.py      ← HTTPS → S3 delta sync crawler
├── 02_process_docs.py     ← Parse + chunk + optional Haiku metadata
├── 03_embed_and_index.py  ← Titan Embed v2 + pgvector indexing
├── 04_query.py            ← LangGraph reasoning engine + CLI
├── 05_evaluate.py         ← RAGAS-style evaluation harness
├── cleanup.py             ← Tear down all AWS resources
├── ARCHITECTURE.md        ← Detailed system design
├── DESIGN_PRINCIPLES.md   ← AWS GAAB principles alignment
├── .gitignore             ← Git ignore rules
└── README.md              ← This file
```

---

## 💰 Cost Management

### Stop charges when not using:

```bash
# Stop RDS instance (saves ~$0.50/day)
aws rds stop-db-instance --db-instance-identifier tgpp-rag-poc --region us-east-1

# Restart when needed
aws rds start-db-instance --db-instance-identifier tgpp-rag-poc --region us-east-1
```

### Full cleanup (delete everything):

```bash
python cleanup.py
```

---

## 🔒 Security Notes

- RDS security group allows 0.0.0.0/0 on port 5432 — **restrict to your IP in production**
- Never commit `config.py` with credentials — use `config.py.example` as template
- Use IAM database authentication for production deployments
- Enable RDS encryption at rest for sensitive data
- Consider VPC endpoints for Bedrock to avoid public internet

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `InvalidClientTokenId` | Re-run `aws configure` with valid credentials |
| `Malformed input request` (Titan Embed) | Chunk text is empty — handled automatically |
| `Operation timed out` (PostgreSQL) | Large file — batching handles this automatically |
| `Legacy model` error | Use inference profile IDs (e.g., `us.anthropic.claude-...`) |
| `No .docx found` | ZIP contains PDF/other format — skipped automatically |
| FTP timeout | Pipeline uses HTTPS instead — no FTP needed |

---

## 🚀 Next Steps to Productionize

1. **API Layer**: Wrap `04_query.py` in FastAPI + API Gateway + Lambda
2. **Orchestration**: Step Functions over steps 1–3 triggered by EventBridge cron
3. **Guardrails**: Add Bedrock Guardrails grounding check to generator
4. **Observability**: CloudWatch custom metrics (faithfulness, latency, token cost)
5. **IaC**: CDK/Terraform for all resources in `00_create_infra.py`
6. **Security**: VPC, IAM DB auth, encryption, WAF
7. **Scale**: Aurora Serverless v2 for auto-scaling pgvector
8. **CI/CD**: GitHub Actions running `05_evaluate.py` on every PR

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [3GPP](https://www.3gpp.org/) for publicly available specifications
- [pgvector](https://github.com/pgvector/pgvector) for PostgreSQL vector search
- [LangGraph](https://github.com/langchain-ai/langgraph) for stateful agent orchestration
- [Amazon Bedrock](https://aws.amazon.com/bedrock/) for managed foundation models
