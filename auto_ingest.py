"""
auto_ingest.py
==============
Auto-downloads, parses, embeds, and indexes 3GPP specs on demand
when the query engine detects insufficient context.

Usage (called automatically by 04_query.py):
    from auto_ingest import auto_ingest_spec
    chunks_added = auto_ingest_spec("38300", "Rel-18")
"""

import io
import re
import json
import hashlib
import zipfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
import requests

from config import (
    AWS_REGION, S3_LANDING_BUCKET, S3_PROCESSED_BUCKET,
    DYNAMODB_TABLE, EMBED_MODEL_ID, PG_HOST, PG_PORT,
    PG_DATABASE, PG_USER, PG_PASSWORD, PG_TABLE,
    MAX_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS, MIN_CHUNK_TOKENS
)

s3 = boto3.client("s3", region_name=AWS_REGION)
ddb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = ddb.Table(DYNAMODB_TABLE)
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

BASE_URL = "https://www.3gpp.org/ftp/Specs/latest"

# Release letter mapping: Rel-18='i', Rel-19='j', Rel-20='k', etc.
REL_LETTER = {str(i): c for i, c in zip(range(10, 27), "abcdefghijklmnopqrstuvwxyz")}


def identify_needed_specs(query: str) -> list[dict]:
    """Use LLM to identify which 3GPP specs are needed for a query."""
    from config import LLM_MODEL_ID

    prompt = f"""Given this 3GPP-related question, identify which 3GPP specification(s) would contain the answer.

Question: {query}

Return ONLY valid JSON array of specs needed:
[{{"spec": "23228", "series": "23_series", "release": "Rel-18", "reason": "IMS architecture"}}]

Rules:
- spec: 5-digit number (e.g. 38331, 23228, 36331)
- series: format like "38_series", "23_series", "36_series"
- release: "Rel-18" or "Rel-19" (prefer latest)
- Return max 3 most relevant specs
- No markdown fences"""

    resp = bedrock.invoke_model(
        modelId=LLM_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}]
        }),
        contentType="application/json",
        accept="application/json"
    )
    raw = json.loads(resp["body"].read())["content"][0]["text"].strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def find_spec_url(spec: str, series: str, release: str) -> str:
    """Find the actual download URL for a spec on 3gpp.org."""
    url = f"{BASE_URL}/{release}/{series}/"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return ""
        # Find matching zip file
        pattern = re.compile(rf'href="([^"]*{spec}[^"]*\.zip)"')
        matches = pattern.findall(resp.text)
        if matches:
            return matches[0]  # First match (latest version)
    except Exception:
        pass
    return ""


def download_spec(zip_url: str) -> tuple[bytes, str]:
    """Download a spec ZIP file. Returns (content, s3_key)."""
    filename = zip_url.split("/")[-1]
    s3_key = f"raw/auto_ingest/{filename}"

    # Check if already downloaded
    try:
        resp = table.scan(
            FilterExpression=boto3.dynamodb.conditions.Attr("s3_key").eq(s3_key)
        )
        if resp.get("Items"):
            return None, s3_key  # Already ingested
    except Exception:
        pass

    r = requests.get(zip_url, timeout=120)
    r.raise_for_status()
    data = r.content
    sha256 = hashlib.sha256(data).hexdigest()

    s3.put_object(
        Bucket=S3_LANDING_BUCKET, Key=s3_key, Body=data,
        Metadata={"sha256": sha256, "source_url": zip_url},
        ContentType="application/zip"
    )
    table.put_item(Item={
        "file_path": zip_url, "s3_key": s3_key,
        "ftp_size": len(data), "sha256": sha256,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "processed": False
    })
    return data, s3_key


def process_and_index(zip_data: bytes, s3_key: str) -> int:
    """Parse, chunk, embed, and index a spec file. Returns chunk count."""
    import docx
    import tiktoken
    import psycopg2

    try:
        import fitz
        HAS_PDF = True
    except ImportError:
        HAS_PDF = False

    enc = tiktoken.get_encoding("cl100k_base")
    filename = Path(s3_key).name

    # Parse metadata from filename
    stem = Path(filename).stem
    parts = stem.split("-")
    spec_number = parts[0] if parts else stem
    release = "unknown"
    if len(parts) > 1 and parts[1] and parts[1][0].lower() in REL_LETTER.values():
        rev_map = {v: f"Rel-{k}" for k, v in REL_LETTER.items()}
        release = rev_map.get(parts[1][0].lower(), "unknown")

    doc_id = str(uuid.uuid4())
    all_text_chunks = []

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        docx_names = [n for n in zf.namelist() if n.lower().endswith(".docx")]
        pdf_names = [n for n in zf.namelist() if n.lower().endswith(".pdf")]

        full_text = ""
        if docx_names:
            with zf.open(docx_names[0]) as f:
                doc = docx.Document(io.BytesIO(f.read()))
                for para in doc.paragraphs:
                    if para.text.strip():
                        full_text += para.text.strip() + "\n"
        elif pdf_names and HAS_PDF:
            with zf.open(pdf_names[0]) as f:
                pdf_doc = fitz.open(stream=f.read(), filetype="pdf")
                for page in pdf_doc:
                    full_text += page.get_text()
                pdf_doc.close()
        else:
            return 0

    if not full_text.strip():
        return 0

    # Simple chunking by paragraphs
    paragraphs = [p.strip() for p in full_text.split("\n") if p.strip()]
    current_chunk = ""
    chunks_data = []

    for para in paragraphs:
        if len(enc.encode(current_chunk + "\n" + para)) > MAX_CHUNK_TOKENS:
            if len(enc.encode(current_chunk)) >= MIN_CHUNK_TOKENS:
                chunks_data.append({
                    "chunk_id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "spec_number": spec_number,
                    "spec_series": spec_number[:2] + "series",
                    "release": release,
                    "section_path": "auto-ingested",
                    "doc_type": "TS",
                    "chunk_text": current_chunk,
                    "token_count": len(enc.encode(current_chunk)),
                    "source_s3_key": s3_key
                })
            current_chunk = para
        else:
            current_chunk += "\n" + para

    if current_chunk and len(enc.encode(current_chunk)) >= MIN_CHUNK_TOKENS:
        chunks_data.append({
            "chunk_id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "spec_number": spec_number,
            "spec_series": spec_number[:2] + "series",
            "release": release,
            "section_path": "auto-ingested",
            "doc_type": "TS",
            "chunk_text": current_chunk,
            "token_count": len(enc.encode(current_chunk)),
            "source_s3_key": s3_key
        })

    if not chunks_data:
        return 0

    # Upload chunks to S3
    lines = "\n".join(json.dumps(c) for c in chunks_data)
    chunk_key = f"chunks/{spec_number}/{doc_id}.jsonl"
    s3.put_object(Bucket=S3_PROCESSED_BUCKET, Key=chunk_key, Body=lines.encode())

    # Embed and index
    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DATABASE,
                            user=PG_USER, password=PG_PASSWORD)
    indexed = 0
    for i in range(0, len(chunks_data), 25):
        batch = chunks_data[i:i+25]
        for chunk in batch:
            text = chunk["chunk_text"][:8000] or "empty"
            try:
                resp = bedrock.invoke_model(
                    modelId=EMBED_MODEL_ID,
                    body=json.dumps({"inputText": text}),
                    contentType="application/json",
                    accept="application/json"
                )
                emb = json.loads(resp["body"].read())["embedding"]
            except Exception:
                emb = [0.0] * 1024

            cur = conn.cursor()
            cur.execute("""INSERT INTO chunks
                (chunk_id, doc_id, spec_series, spec_number, release, section_path,
                 doc_type, chunk_text, summary, keywords, hyp_questions,
                 token_count, source_s3_key, embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    embedding=EXCLUDED.embedding, chunk_text=EXCLUDED.chunk_text""",
                (chunk["chunk_id"], chunk["doc_id"], chunk.get("spec_series", ""),
                 chunk["spec_number"], chunk.get("release", ""),
                 chunk.get("section_path", ""), chunk.get("doc_type", "TS"),
                 chunk["chunk_text"], "", [], [],
                 chunk.get("token_count", 0), chunk.get("source_s3_key", ""),
                 str(emb)))
            conn.commit()
            indexed += 1
        time.sleep(0.3)

    conn.close()

    # Mark as processed
    try:
        resp = table.scan(
            FilterExpression=boto3.dynamodb.conditions.Attr("s3_key").eq(s3_key)
        )
        for item in resp.get("Items", []):
            table.update_item(
                Key={"file_path": item["file_path"]},
                UpdateExpression="SET #p = :t",
                ExpressionAttributeNames={"#p": "processed"},
                ExpressionAttributeValues={":t": True}
            )
    except Exception:
        pass

    return indexed


def auto_ingest_spec(query: str) -> tuple[int, list[str]]:
    """
    Auto-identify, download, process, and index specs needed for a query.
    Returns (total_chunks_added, list_of_specs_ingested).
    """
    print("  [auto-ingest] Identifying needed specs …")
    needed = identify_needed_specs(query)

    if not needed:
        return 0, []

    total_chunks = 0
    ingested_specs = []

    for spec_info in needed[:3]:  # Max 3 specs per query
        spec = spec_info.get("spec", "")
        series = spec_info.get("series", "")
        release = spec_info.get("release", "Rel-18")
        reason = spec_info.get("reason", "")

        print(f"  [auto-ingest] Looking for TS {spec} ({series}, {release}) — {reason}")

        zip_url = find_spec_url(spec, series, release)
        if not zip_url:
            # Try Rel-19 if Rel-18 not found
            zip_url = find_spec_url(spec, series, "Rel-19")
        if not zip_url:
            print(f"    ✗ Not found on 3gpp.org")
            continue

        print(f"    Downloading {zip_url.split('/')[-1]} …")
        try:
            data, s3_key = download_spec(zip_url)
            if data is None:
                print(f"    · Already ingested")
                continue

            print(f"    Processing and indexing …")
            n = process_and_index(data, s3_key)
            total_chunks += n
            ingested_specs.append(f"TS {spec} ({release})")
            print(f"    ✓ Indexed {n} chunks from TS {spec}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

    return total_chunks, ingested_specs
