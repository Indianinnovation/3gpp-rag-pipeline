"""
01_ftp_crawler.py
=================
Downloads 3GPP documents via HTTPS from www.3gpp.org:
- TSGR2_129 and 129bis meeting docs + LS exchanges
- Rel-18/19/20 38-series specs

Uploads to S3 landing bucket with DynamoDB delta tracking.

Usage:
    python 01_ftp_crawler.py               # full delta sync
    python 01_ftp_crawler.py --dry-run     # list files only
    python 01_ftp_crawler.py --source meetings   # only meeting docs
    python 01_ftp_crawler.py --source specs      # only Rel-18/19/20 specs
"""

import argparse
import hashlib
import io
import re
import time
from datetime import datetime, timezone

import boto3
import requests
from botocore.exceptions import ClientError

from config import (
    AWS_REGION, S3_LANDING_BUCKET, DYNAMODB_TABLE, MAX_FILES_POC
)

s3 = boto3.client("s3", region_name=AWS_REGION)
ddb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = ddb.Table(DYNAMODB_TABLE)

BASE_URL = "https://www.3gpp.org/ftp"

# Sources to crawl
MEETING_SOURCES = [
    {"url": f"{BASE_URL}/tsg_ran/WG2_RL2/TSGR2_129/Docs/", "s3_prefix": "raw/TSGR2_129/Docs"},
    {"url": f"{BASE_URL}/tsg_ran/WG2_RL2/TSGR2_129/LSin/", "s3_prefix": "raw/TSGR2_129/LSin"},
    {"url": f"{BASE_URL}/tsg_ran/WG2_RL2/TSGR2_129/LSout/", "s3_prefix": "raw/TSGR2_129/LSout"},
    {"url": f"{BASE_URL}/tsg_ran/WG2_RL2/TSGR2_129bis/Docs/", "s3_prefix": "raw/TSGR2_129bis/Docs"},
    {"url": f"{BASE_URL}/tsg_ran/WG2_RL2/TSGR2_129bis/LSin/", "s3_prefix": "raw/TSGR2_129bis/LSin"},
    {"url": f"{BASE_URL}/tsg_ran/WG2_RL2/TSGR2_129bis/LSout/", "s3_prefix": "raw/TSGR2_129bis/LSout"},
]

SPEC_SOURCES = [
    {"url": f"{BASE_URL}/Specs/latest/Rel-18/38_series/", "s3_prefix": "raw/specs/Rel-18/38_series"},
    {"url": f"{BASE_URL}/Specs/latest/Rel-19/38_series/", "s3_prefix": "raw/specs/Rel-19/38_series"},
    {"url": f"{BASE_URL}/Specs/latest/Rel-20/38_series/", "s3_prefix": "raw/specs/Rel-20/38_series"},
]

# CR (Change Request) archive sources — contains per-release change descriptions
# Each CR zip has the change description + affected clauses + release info
CR_SOURCES = [
    {"url": f"{BASE_URL}/Specs/archive/38_series/38.413/", "s3_prefix": "raw/cr/38413", "spec": "38413"},
    {"url": f"{BASE_URL}/Specs/archive/38_series/38.473/", "s3_prefix": "raw/cr/38473", "spec": "38473"},
    {"url": f"{BASE_URL}/Specs/archive/24_series/24.501/", "s3_prefix": "raw/cr/24501", "spec": "24501"},
    {"url": f"{BASE_URL}/Specs/archive/38_series/38.331/", "s3_prefix": "raw/cr/38331", "spec": "38331"},
    {"url": f"{BASE_URL}/Specs/archive/38_series/38.423/", "s3_prefix": "raw/cr/38423", "spec": "38423"},
    {"url": f"{BASE_URL}/Specs/archive/38_series/38.463/", "s3_prefix": "raw/cr/38463", "spec": "38463"},
]


# ─────────────────────────────────────────────────────────────────────────────
# HTTPS directory listing
# ─────────────────────────────────────────────────────────────────────────────
def list_zip_files(url: str) -> list[str]:
    """Scrape directory listing for .zip file URLs."""
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return re.findall(r'href="([^"]+\.zip)"', resp.text)
    except Exception as e:
        print(f"  ⚠ Failed to list {url}: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Delta detection
# ─────────────────────────────────────────────────────────────────────────────
def is_already_ingested(file_url: str) -> bool:
    try:
        resp = table.get_item(Key={"file_path": file_url})
        return "Item" in resp
    except ClientError:
        return False


def record_manifest(file_url: str, s3_key: str, size: int, sha256: str):
    table.put_item(Item={
        "file_path": file_url,
        "s3_key": s3_key,
        "ftp_size": size,
        "sha256": sha256,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "processed": False
    })


# ─────────────────────────────────────────────────────────────────────────────
# Download + upload
# ─────────────────────────────────────────────────────────────────────────────
def download_and_upload(file_url: str, s3_key: str) -> bool:
    try:
        resp = requests.get(file_url, timeout=120)
        resp.raise_for_status()
        data = resp.content
    except Exception as e:
        print(f"    ✗ Download failed: {e}")
        return False

    sha256 = hashlib.sha256(data).hexdigest()
    s3.put_object(
        Bucket=S3_LANDING_BUCKET,
        Key=s3_key,
        Body=data,
        Metadata={"sha256": sha256, "source_url": file_url},
        ContentType="application/zip"
    )
    record_manifest(file_url, s3_key, len(data), sha256)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Collect files from all sources
# ─────────────────────────────────────────────────────────────────────────────
def collect_files(sources: list[dict]) -> list[dict]:
    files = []
    for src in sources:
        print(f"  Scanning: {src['url']}")
        urls = list_zip_files(src["url"])
        for url in urls:
            filename = url.split("/")[-1]
            files.append({
                "url": url,
                "filename": filename,
                "s3_key": f"{src['s3_prefix']}/{filename}"
            })
        print(f"    Found {len(urls)} zip files")
    return files


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source", choices=["meetings", "specs", "cr", "all"], default="all")
    parser.add_argument("--limit", type=int, default=MAX_FILES_POC)
    parser.add_argument("--spec-filter", type=str, default=None, help="Only crawl CRs for this spec (e.g. 38413)")
    args = parser.parse_args()

    sources = []
    if args.source in ("meetings", "all"):
        sources += MEETING_SOURCES
    if args.source in ("specs", "all"):
        sources += SPEC_SOURCES
    if args.source in ("cr", "all"):
        cr_list = CR_SOURCES
        if args.spec_filter:
            cr_list = [s for s in CR_SOURCES if s["spec"] == args.spec_filter]
        sources += cr_list

    print(f"\n3GPP HTTPS Crawler — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Sources: {args.source}")
    print(f"Mode   : {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Limit  : {args.limit}\n")

    print("Scanning directories …")
    all_files = collect_files(sources)
    print(f"\n  Total zip files found: {len(all_files)}")

    to_download = [f for f in all_files if not is_already_ingested(f["url"])]
    print(f"  New files to download: {len(to_download)}")

    if args.dry_run:
        for f in to_download[:20]:
            print(f"    {f['filename']}")
        if len(to_download) > 20:
            print(f"    … and {len(to_download) - 20} more")
        return

    if len(to_download) > args.limit:
        print(f"  ⚠ Capping at {args.limit} files (use --limit to change)")
        to_download = to_download[:args.limit]

    ok = fail = 0
    for i, f in enumerate(to_download, 1):
        print(f"  [{i}/{len(to_download)}] {f['filename']} …")
        if download_and_upload(f["url"], f["s3_key"]):
            print(f"    ✓ s3://{S3_LANDING_BUCKET}/{f['s3_key']}")
            ok += 1
        else:
            fail += 1
        time.sleep(0.3)

    print(f"\n── Summary ─────────────────────────────────")
    print(f"  Downloaded : {ok}")
    print(f"  Failed     : {fail}")
    print(f"  Skipped    : {len(all_files) - len(to_download)} (already ingested)")
    print(f"\nNext step: python 02_process_docs.py")


if __name__ == "__main__":
    main()
