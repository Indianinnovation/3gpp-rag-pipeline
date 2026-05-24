"""
ingest_rel19_rrc.py
===================
One-time script to index TS 38.331 Rel-19 from S3.
Also serves as a template for future corpus gap fixes.

Usage:
    pip install python-docx tiktoken PyMuPDF boto3 psycopg2-binary
    python ingest_rel19_rrc.py

Future Prevention Strategy:
===========================
The Q9 issue (wrong content, high confidence) happens when:
  1. User asks about spec X release Y
  2. Spec X release Y is NOT in the index
  3. Retriever returns tangentially related content from other specs
  4. Generator confidently answers from wrong context

To prevent this permanently, implement these 3 safeguards:

SAFEGUARD 1 — Release-aware retrieval:
  When query mentions a specific release (Rel-15/16/17/18/19/20),
  add a release_filter to the retrieval call. This ensures only
  chunks from that release are returned.

SAFEGUARD 2 — Corpus coverage check:
  Before generating, check if the retrieved chunks actually match
  the requested spec+release. If not, flag as "insufficient context"
  instead of generating a confident wrong answer.

SAFEGUARD 3 — Automated gap detection + ingestion:
  After each query, log which spec+release was requested vs what
  was retrieved. If there's a mismatch, trigger auto-ingest for
  the missing spec+release combination.

Implementation below covers SAFEGUARD 3.
"""

import json
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The specs that SHOULD be indexed for every release
CRITICAL_SPECS = {
    "38331": "RRC protocol",
    "38300": "NR architecture",
    "38304": "Cell selection/reselection",
    "38912": "Study on NR",
    "38321": "MAC protocol",
    "38211": "Physical channels",
    "24501": "NAS protocol",
    "38413": "NGAP",
    "38473": "F1AP",
    "38423": "XnAP",
    "38463": "E1AP",
}

RELEASES = ["Rel-18", "Rel-19", "Rel-20"]


def check_corpus_gaps():
    """Check which critical spec+release combinations are missing from the index."""
    import psycopg2
    
    conn = psycopg2.connect(
        host=os.environ.get("PG_HOST", "tgpp-rag-poc.cc9k6aouw4cy.us-east-1.rds.amazonaws.com"),
        port=5432, dbname="ragdb", user="ragadmin",
        password=os.environ.get("PG_PASSWORD", "Change_Me_123!")
    )
    cur = conn.cursor()
    
    gaps = []
    for spec, desc in CRITICAL_SPECS.items():
        for release in RELEASES:
            cur.execute(
                "SELECT COUNT(*) FROM chunks WHERE spec_number = %s AND release = %s",
                (spec, release)
            )
            count = cur.fetchone()[0]
            if count == 0:
                gaps.append({"spec": spec, "desc": desc, "release": release})
                print(f"  GAP: TS {spec} ({desc}) {release} — 0 chunks")
            else:
                print(f"  OK:  TS {spec} ({desc}) {release} — {count} chunks")
    
    conn.close()
    return gaps


def check_s3_availability(gaps):
    """Check which gaps have source files available in S3."""
    import boto3
    s3 = boto3.client("s3", region_name="us-east-1")
    
    available = []
    for gap in gaps:
        # Map release to letter: Rel-18=i, Rel-19=j, Rel-20=k
        rel_num = int(gap["release"].split("-")[1])
        rel_letter = chr(ord('a') + rel_num - 10)  # 18→i, 19→j, 20→k
        
        prefix = f"raw/specs/{gap['release']}/38_series/{gap['spec']}"
        paginator = s3.get_paginator("list_objects_v2")
        found = False
        for page in paginator.paginate(Bucket="3gpp-rag-landing", Prefix=prefix):
            if page.get("Contents"):
                found = True
                gap["s3_key"] = page["Contents"][0]["Key"]
                break
        
        if not found:
            # Try alternate prefix
            prefix2 = f"raw/specs/{gap['release']}/38_series/"
            for page in paginator.paginate(Bucket="3gpp-rag-landing", Prefix=prefix2):
                for obj in page.get("Contents", []):
                    if gap["spec"] in obj["Key"]:
                        found = True
                        gap["s3_key"] = obj["Key"]
                        break
        
        if found:
            available.append(gap)
            print(f"  AVAILABLE: {gap['s3_key']}")
        else:
            print(f"  MISSING:   TS {gap['spec']} {gap['release']} not in S3 — needs crawling")
    
    return available


if __name__ == "__main__":
    print("=" * 60)
    print("  CORPUS GAP ANALYSIS")
    print("=" * 60)
    print("\nChecking index for critical spec+release coverage...\n")
    
    gaps = check_corpus_gaps()
    
    if not gaps:
        print("\n✓ All critical specs are indexed for all releases!")
        sys.exit(0)
    
    print(f"\n{'=' * 60}")
    print(f"  Found {len(gaps)} gaps. Checking S3 availability...")
    print(f"{'=' * 60}\n")
    
    available = check_s3_availability(gaps)
    
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total gaps: {len(gaps)}")
    print(f"  Available in S3 (can index now): {len(available)}")
    print(f"  Need crawling first: {len(gaps) - len(available)}")
    
    if available:
        print(f"\n  To index available specs, run:")
        for gap in available:
            print(f"    python 02_process_docs.py --s3-key {gap['s3_key']} --skip-metadata")
        print(f"    python 03_embed_and_index.py")
    
    # Write gaps to file for tracking
    with open("corpus_gaps.json", "w") as f:
        json.dump({"gaps": gaps, "available": available}, f, indent=2)
    print(f"\n  Gap report saved to corpus_gaps.json")
