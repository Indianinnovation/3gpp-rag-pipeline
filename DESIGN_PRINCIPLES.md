# Design Principles — AWS Generative AI Application Builder

> This project follows the [AWS Generative AI Application Builder (GAAB)](https://docs.aws.amazon.com/solutions/generative-ai-application-builder-on-aws/) design principles for building production-ready GenAI applications.

---

## 1. Rapid Experimentation

| Principle | Implementation |
|-----------|---------------|
| Compare model outputs | `config.py` allows swapping LLM_MODEL_ID between Claude Sonnet/Haiku without code changes |
| Iterate on prompts | System prompts in `04_query.py` (PLANNER_SYSTEM, GENERATOR_SYSTEM) are externalized constants |
| Test configurations | `--skip-metadata`, `--limit`, `--spec` flags enable fast iteration |
| Evaluate changes | `05_evaluate.py` provides automated RAGAS-style scoring against golden Q&A sets |

**How to experiment:**
```bash
# Try different models
# Edit config.py → LLM_MODEL_ID
python 04_query.py --query "What is carrier aggregation?"

# Compare with/without metadata enrichment
python 02_process_docs.py --limit 5
python 02_process_docs.py --limit 5 --skip-metadata

# Evaluate impact
python 05_evaluate.py
```

---

## 2. Configurability

| Principle | Implementation |
|-----------|---------------|
| Centralized config | Single `config.py` controls all models, endpoints, chunking params |
| Model selection | Swap between Titan/Cohere embeddings, Claude/Llama for generation |
| Chunking tuning | `MAX_CHUNK_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `MIN_CHUNK_TOKENS` |
| Search tuning | Hybrid weight (70% vector / 30% keyword) adjustable in `03_embed_and_index.py` |
| Source selection | `--source meetings|specs|all` controls ingestion scope |

**Configuration hierarchy:**
```
config.py.example  →  config.py  →  CLI flags (--limit, --spec, --skip-metadata)
```

---

## 3. Production-Ready (Well-Architected)

### Security Pillar
| Practice | Implementation |
|----------|---------------|
| Secrets management | `config.py` git-ignored; `config.py.example` as template |
| Network isolation | Security group created per deployment; restrict CIDR in production |
| IAM least privilege | Scoped permissions for S3, DynamoDB, Bedrock, RDS |
| Encryption in transit | RDS SSL connections; HTTPS for all API calls |
| No hardcoded credentials | All secrets via `config.py` or environment variables |

### Reliability Pillar
| Practice | Implementation |
|----------|---------------|
| Auto-reconnect | `03_embed_and_index.py` reconnects on PostgreSQL timeout |
| Batch processing | 50-chunk batches prevent memory/timeout issues |
| Delta sync | DynamoDB manifest tracks ingested files; re-runs skip duplicates |
| Graceful degradation | Empty chunks skipped; embed errors use zero vectors |
| Idempotent operations | `ON CONFLICT DO UPDATE` ensures safe re-indexing |

### Performance Pillar
| Practice | Implementation |
|----------|---------------|
| HNSW index | Approximate nearest neighbor for sub-100ms vector search |
| GIN index | Full-text search acceleration |
| Hybrid retrieval | 70% semantic + 30% lexical for better recall |
| Connection pooling | Fresh connections per batch to avoid idle timeouts |
| Streaming support | LangGraph architecture supports token streaming |

### Cost Optimization Pillar
| Practice | Implementation |
|----------|---------------|
| Right-sized compute | db.t3.micro for POC (~$0.50/day vs $11.52/day AOSS) |
| On-demand pricing | Bedrock pay-per-token; no provisioned capacity |
| Resource lifecycle | `cleanup.py` tears down all resources; RDS stop/start |
| POC caps | `MAX_FILES_POC` prevents runaway costs |
| Skip optional steps | `--skip-metadata` reduces Bedrock calls by 90% |

### Operational Excellence Pillar
| Practice | Implementation |
|----------|---------------|
| Progress logging | Each pipeline step prints status with counts |
| Error handling | Try/except with descriptive messages at every stage |
| Evaluation harness | `05_evaluate.py` for regression testing |
| Cleanup automation | `cleanup.py` for full resource teardown |
| Documentation | README, ARCHITECTURE.md, inline code comments |

---

## 4. Extensible Modular Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Modular Pipeline                           │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│  Ingest  │  Parse   │  Embed   │  Index   │  Query Engine  │
│  Module  │  Module  │  Module  │  Module  │  Module        │
├──────────┼──────────┼──────────┼──────────┼────────────────┤
│ HTTPS    │ python-  │ Titan    │ pgvector │ LangGraph      │
│ crawler  │ docx     │ Embed v2 │ HNSW+GIN │ StateGraph     │
└──────────┴──────────┴──────────┴──────────┴────────────────┘
     ↕           ↕          ↕          ↕           ↕
  Swappable  Swappable  Swappable  Swappable   Swappable
  (FTP/API)  (PDF/HTML) (Cohere)   (AOSS/FAISS) (Bedrock Agents)
```

**Extension points:**
- **Ingest**: Add new sources (APIs, SharePoint, Confluence)
- **Parse**: Support PDF via Textract, HTML via BeautifulSoup
- **Embed**: Swap to Cohere Embed, OpenAI, or custom models
- **Index**: Switch to OpenSearch, FAISS, Pinecone, or Bedrock Knowledge Bases
- **Query**: Replace LangGraph with Bedrock Agents or Strands SDK

---

## 5. Authentication & Authorization (Production Path)

| Component | POC (Current) | Production (Recommended) |
|-----------|---------------|--------------------------|
| API access | AWS CLI credentials | Cognito + API Gateway |
| DB auth | Username/password | IAM Database Authentication |
| Model access | IAM user | IAM role with conditions |
| Network | Public RDS | VPC + private subnets |
| Web UI | CLI only | CloudFront + S3 + Cognito |

**Production deployment path:**
```bash
# Wrap query engine in FastAPI
# Deploy via Lambda + API Gateway
# Add Cognito user pool
# Enable WAF rules
# Add VPC endpoints for Bedrock
```

---

## 6. Observability & Monitoring (Production Path)

| Metric | Source | Implementation |
|--------|--------|---------------|
| Query latency | CloudWatch | Timer around LangGraph invoke |
| Retrieval quality | Custom metric | Confidence score per query |
| Token usage | Bedrock logs | Enable model invocation logging |
| Index health | PostgreSQL | Row count, index size monitoring |
| Pipeline throughput | CloudWatch | Chunks processed per minute |
| Cost tracking | Cost Explorer | Tag-based cost allocation |

**Enable Bedrock logging:**
```python
# Add to production deployment
bedrock.put_model_invocation_logging_configuration(
    loggingConfig={
        "cloudWatchConfig": {"logGroupName": "/aws/bedrock/3gpp-rag"},
        "textDataDeliveryEnabled": True
    }
)
```

---

## 7. Guardrails & Safety

| Guardrail | Implementation |
|-----------|---------------|
| Grounding check | Generator prompt: "Answer ONLY from provided context" |
| Hallucination prevention | Confidence scoring + "needs_more_context" flag |
| Citation requirement | Every answer includes spec/clause citations |
| Input validation | Empty text filtering before embedding |
| Output validation | JSON confidence extraction + threshold checks |
| Content filtering | Bedrock Guardrails (production add-on) |

**Production guardrails:**
```python
# Add Bedrock Guardrails to generator
bedrock.apply_guardrail(
    guardrailIdentifier="3gpp-rag-guardrail",
    guardrailVersion="1",
    source="OUTPUT",
    content=[{"text": {"text": answer}}]
)
```

---

## 8. Feedback & Continuous Improvement

| Practice | Implementation |
|----------|---------------|
| Golden Q&A set | `05_evaluate.py` with 5 domain-specific questions |
| Automated scoring | LLM-as-Judge (faithfulness, relevancy, completeness) |
| Regression testing | Run evaluation on every code change |
| Threshold alerts | Warn if faithfulness < 0.70 |
| Report generation | JSON evaluation reports with timestamps |

**CI/CD integration:**
```yaml
# GitHub Actions example
- name: Run evaluation
  run: python 05_evaluate.py
  env:
    AWS_REGION: us-east-1
- name: Check thresholds
  run: |
    python -c "
    import json
    report = json.load(open('eval_report_*.json'))
    assert report['aggregate']['faithfulness'] >= 0.70
    "
```

---

## Summary: GAAB Principle Mapping

| GAAB Principle | This Pipeline |
|----------------|---------------|
| Rapid experimentation | Config-driven, CLI flags, evaluation harness |
| Configurability | Single config.py, swappable models/indexes |
| Production-ready | Well-Architected across all 5 pillars |
| Extensible architecture | Modular stages, clear interfaces |
| Authentication | IAM-based, Cognito-ready for production |
| Observability | Logging, metrics, evaluation reports |
| Guardrails | Grounding prompts, confidence scoring, citations |
| Feedback loop | Automated evaluation, threshold alerts |

---

## References

- [AWS Generative AI Application Builder](https://docs.aws.amazon.com/solutions/generative-ai-application-builder-on-aws/)
- [AWS Well-Architected Framework — GenAI Lens](https://docs.aws.amazon.com/wellarchitected/latest/generative-ai-lens/)
- [Amazon Bedrock Best Practices](https://docs.aws.amazon.com/bedrock/latest/userguide/best-practices.html)
- [pgvector Performance Tuning](https://github.com/pgvector/pgvector#performance)
