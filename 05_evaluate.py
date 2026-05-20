"""
05_evaluate.py
==============
RAGAS evaluation harness for the 3GPP RAG pipeline.
Measures faithfulness, answer relevancy, and context precision
against a golden Q&A set.

Usage:
    python 05_evaluate.py                    # run default golden set
    python 05_evaluate.py --golden my_qa.json
    python 05_evaluate.py --single "What are RRC states?"
"""

import argparse
import json
import statistics
from datetime import datetime

import boto3

from config import AWS_REGION, LLM_MODEL_ID, HAIKU_MODEL_ID
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

embed_link = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embed_and_index.py")
if not os.path.exists(embed_link):
    os.symlink("03_embed_and_index.py", embed_link)

query_link = os.path.join(os.path.dirname(os.path.abspath(__file__)), "query.py")
if not os.path.exists(query_link):
    os.symlink("04_query.py", query_link)

from query import create_engine, run_query

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# ─────────────────────────────────────────────────────────────────────────────
# Golden Q&A set — 3GPP-domain ground truth
# ─────────────────────────────────────────────────────────────────────────────
GOLDEN_QA = [
    {
        "question": "What are the three RRC states in 5G NR?",
        "ground_truth": "5G NR defines three RRC states: RRC_IDLE, RRC_INACTIVE, and RRC_CONNECTED. RRC_INACTIVE is a new state introduced in NR that was not present in LTE."
    },
    {
        "question": "What protocol does the UE use to request a paging response in RRC_IDLE?",
        "ground_truth": "In RRC_IDLE, the UE monitors the paging channel and responds via a random access procedure on PRACH to initiate RRC connection establishment."
    },
    {
        "question": "What is the role of the AMF in 5G core network?",
        "ground_truth": "The Access and Mobility Management Function (AMF) is responsible for UE registration, connection management, reachability management, and mobility management in the 5G Core network."
    },
    {
        "question": "What is the difference between SA and NSA 5G deployments?",
        "ground_truth": "SA (Standalone) uses a 5G NR air interface with a 5G Core, while NSA (Non-Standalone) uses 5G NR as a secondary cell anchored to an LTE primary cell with an EPC or 5GC core."
    },
    {
        "question": "What is the maximum number of HARQ processes in NR downlink?",
        "ground_truth": "NR supports up to 16 HARQ processes in the downlink, compared to 8 in LTE, providing greater scheduling flexibility."
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# LLM-as-Judge scoring (Claude Haiku for cost efficiency)
# ─────────────────────────────────────────────────────────────────────────────
JUDGE_SYSTEM = """You are an objective evaluator of RAG system answers about 3GPP specifications.
Score the answer on three dimensions, each 0.0–1.0:
- faithfulness: does the answer contain only claims supported by the provided context?
- relevancy: how directly does the answer address the question?
- completeness: how fully does the answer cover the ground truth?
Return ONLY valid JSON: {"faithfulness": 0.0, "relevancy": 0.0, "completeness": 0.0, "rationale": "..."}"""


def judge_answer(question: str, answer: str, context_chunks: list[dict],
                 ground_truth: str) -> dict:
    context_str = "\n\n".join(c["chunk_text"][:500] for c in context_chunks[:4])
    prompt = f"""Question: {question}

Ground truth: {ground_truth}

Retrieved context (first 4 chunks):
{context_str}

System answer: {answer}

Score the answer:"""

    resp = bedrock.invoke_model(
        modelId=HAIKU_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "system": JUDGE_SYSTEM,
            "messages": [{"role": "user", "content": prompt}]
        }),
        contentType="application/json",
        accept="application/json"
    )
    raw = json.loads(resp["body"].read())["content"][0]["text"].strip()
    import re
    raw = re.sub(r"^```json\s*|\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"faithfulness": 0.0, "relevancy": 0.0, "completeness": 0.0,
                "rationale": "parse error"}


# ─────────────────────────────────────────────────────────────────────────────
# Run evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(qa_set: list[dict]) -> dict:
    engine = create_engine()
    print(f"Running evaluation on {len(qa_set)} questions …\n")

    results = []
    for i, qa in enumerate(qa_set, 1):
        print(f"[{i}/{len(qa_set)}] {qa['question'][:70]}")
        result = run_query(engine, qa["question"], spec=qa.get("spec"))
        scores = judge_answer(
            question=qa["question"],
            answer=result["answer"],
            context_chunks=[{"chunk_text": c.get("chunk_text", "")}
                            for c in result.get("citations", [])],
            ground_truth=qa["ground_truth"]
        )
        entry = {
            "question":     qa["question"],
            "answer":       result["answer"][:200] + "…",
            "confidence":   result["confidence"],
            "faithfulness": scores.get("faithfulness", 0),
            "relevancy":    scores.get("relevancy", 0),
            "completeness": scores.get("completeness", 0),
            "rationale":    scores.get("rationale", "")
        }
        results.append(entry)
        print(f"  faith={entry['faithfulness']:.2f}  "
              f"relev={entry['relevancy']:.2f}  "
              f"compl={entry['completeness']:.2f}  "
              f"conf={entry['confidence']:.2f}")

    # Aggregate
    agg = {
        "faithfulness":  statistics.mean(r["faithfulness"] for r in results),
        "relevancy":     statistics.mean(r["relevancy"]    for r in results),
        "completeness":  statistics.mean(r["completeness"] for r in results),
        "confidence":    statistics.mean(r["confidence"]   for r in results)
    }
    return {"results": results, "aggregate": agg,
            "evaluated_at": datetime.utcnow().isoformat()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=str, default=None,
                        help="Path to JSON file with [{question, ground_truth, spec}]")
    parser.add_argument("--single", type=str, default=None)
    args = parser.parse_args()

    if args.single:
        engine = create_engine()
        result = run_query(engine, args.single)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nConfidence: {result['confidence']:.0%}")
        return

    if args.golden:
        with open(args.golden) as f:
            qa_set = json.load(f)
    else:
        qa_set = GOLDEN_QA

    report = run_evaluation(qa_set)
    agg = report["aggregate"]

    print(f"\n{'─'*60}")
    print("EVALUATION REPORT")
    print(f"{'─'*60}")
    print(f"  Faithfulness  : {agg['faithfulness']:.2%}")
    print(f"  Relevancy     : {agg['relevancy']:.2%}")
    print(f"  Completeness  : {agg['completeness']:.2%}")
    print(f"  Avg confidence: {agg['confidence']:.2%}")
    print(f"{'─'*60}")

    # Save report
    report_file = f"eval_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {report_file}")

    # Pass/fail thresholds for CI
    if agg["faithfulness"] < 0.7:
        print("⚠ ALERT: Faithfulness below 0.70 threshold")
    if agg["relevancy"] < 0.7:
        print("⚠ ALERT: Relevancy below 0.70 threshold")


if __name__ == "__main__":
    main()
