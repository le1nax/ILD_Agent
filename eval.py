"""
Evaluate the RAG pipeline against test_cases.yaml.

Modes:
  python eval.py                  # full eval: RAG + diagnosis judge + citation judge
  python eval.py --cases 1 3 5    # run specific case IDs
  python eval.py --cite-only      # re-judge citations from cached eval_results.json
"""

import argparse
import json
import os
import re
import time

import yaml
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv(override=True)

from graph import graph, State

# ── Judge prompts ──

DIAGNOSIS_JUDGE_PROMPT = """\
You are an expert ILD (interstitial lung disease) clinician evaluating a RAG system's diagnostic output.

You will receive:
- A clinical case (history + radiology)
- The expected diagnosis
- The RAG system's answer

Judge whether the RAG system's PRIMARY diagnosis matches the expected diagnosis.

Rules:
- "match": The RAG answer's primary/leading diagnosis is essentially the same as expected.
  Minor wording differences are fine (e.g., "IPF" vs "idiopathic pulmonary fibrosis").
- "partial": The expected diagnosis is mentioned but not as the primary suggestion,
  OR the answer is close but missing a key qualifier (e.g., says "NSIP" but misses "CTD-associated").
- "mismatch": The RAG answer's primary diagnosis is fundamentally different from expected.

Respond with ONLY a JSON object:
{
  "verdict": "match" | "partial" | "mismatch",
  "rag_diagnosis": "<the primary diagnosis the RAG system gave>",
  "reasoning": "<1-2 sentences explaining your judgement>"
}"""

CITATION_JUDGE_PROMPT = """\
You are a medical citation auditor. You will receive:
1. A RAG system's answer that contains inline [Source N] citations.
2. The actual source texts that were retrieved (numbered [Source 1], [Source 2], etc.).

Evaluate EACH [Source N] citation in the answer. For each one, check whether the claim
it is attached to is actually supported by the corresponding source text.

Also check for uncited claims: factual medical statements in the answer that have no citation
but should have one.

Respond with ONLY a JSON object:
{
  "citations_checked": <total number of [Source N] citations in the answer>,
  "supported": <number that are correctly supported by their source>,
  "unsupported": <number where the source does NOT support the claim>,
  "uncited_claims": <number of factual medical claims with no citation that should have one>,
  "faithfulness_score": <supported / citations_checked, as a float 0.0-1.0>,
  "details": [
    {
      "citation": "[Source N]",
      "claim": "<the claim it is attached to, quoted briefly>",
      "verdict": "supported" | "unsupported",
      "reason": "<brief explanation>"
    }
  ],
  "uncited_examples": ["<example uncited claim 1>", "<example uncited claim 2>"]
}"""


def make_llm():
    return AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name="gpt-5",
        api_version="2025-03-01-preview",
        temperature=1,
    )


def build_question(case: dict) -> str:
    return (
        f"Clinical History:\n{case['clinical_history'].strip()}\n\n"
        f"Radiology Report:\n{case['radiology_report'].strip()}\n\n"
        f"What is the most likely diagnosis and recommended treatment plan?"
    )


def judge_diagnosis(llm, case: dict, answer: str) -> dict:
    messages = [
        SystemMessage(content=DIAGNOSIS_JUDGE_PROMPT),
        HumanMessage(content=(
            f"## Clinical Case\n{build_question(case)}\n\n"
            f"## Expected Diagnosis\n{case['expected_diagnosis']}\n\n"
            f"## RAG System Answer\n{answer}"
        )),
    ]
    response = llm.invoke(messages)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"verdict": "error", "rag_diagnosis": "parse error", "reasoning": response.content}


def judge_citations(llm, answer: str, sources: list[dict]) -> dict:
    sources_text = "\n\n".join(
        f"[Source {i+1}] {s['text']}" for i, s in enumerate(sources)
    )
    messages = [
        SystemMessage(content=CITATION_JUDGE_PROMPT),
        HumanMessage(content=(
            f"## RAG System Answer\n{answer}\n\n"
            f"## Retrieved Sources\n{sources_text}"
        )),
    ]
    response = llm.invoke(messages)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"error": "parse error", "raw": response.content}


def run_rag(case: dict) -> dict:
    """Run the RAG pipeline on a case, return state dict with answer + docs."""
    question = build_question(case)
    t0 = time.time()
    state = graph.invoke(State(question=question))
    elapsed = time.time() - t0

    sources = []
    for doc in state.get("docs", []):
        sources.append({
            "document": doc.metadata.get("document", "Unknown"),
            "page": doc.metadata.get("page", "Unknown"),
            "text": doc.page_content,
        })

    return {
        "answer": state["answer"],
        "sources": sources,
        "time_s": round(elapsed, 1),
    }


def run_eval(case_ids: list[int] | None = None, cite_only: bool = False):
    with open("test_cases.yaml") as f:
        data = yaml.safe_load(f)

    cases = data["cases"]
    if case_ids:
        cases = [c for c in cases if c["id"] in case_ids]

    # If cite_only, load cached results
    cached = {}
    if cite_only:
        try:
            with open("eval_results.json") as f:
                for r in json.load(f):
                    cached[r["id"]] = r
        except FileNotFoundError:
            print("ERROR: eval_results.json not found. Run full eval first.")
            return

    llm = make_llm()

    results = []
    for case in cases:
        cid = case["id"]
        name = case["name"]
        print(f"\n{'='*70}")
        print(f"Case {cid}: {name}")
        print(f"Expected: {case['expected_diagnosis']}")
        print(f"{'='*70}")

        if cite_only and cid in cached:
            # Reuse cached RAG output
            answer = cached[cid]["answer"]
            sources = cached[cid]["sources"]
            rag_time = cached[cid].get("time_s", 0)
            diag = {
                "verdict": cached[cid].get("verdict", "unknown"),
                "rag_diagnosis": cached[cid].get("rag_diagnosis", "N/A"),
                "reasoning": cached[cid].get("reasoning", "N/A"),
            }
            print(f"  (using cached RAG output)")
        else:
            # Run RAG pipeline
            rag_out = run_rag(case)
            answer = rag_out["answer"]
            sources = rag_out["sources"]
            rag_time = rag_out["time_s"]

            print(f"Retrieved {len(sources)} chunks in {rag_time}s")
            print(f"\nAnswer (first 500 chars):\n{answer[:500]}...")

            # Diagnosis judge
            diag = judge_diagnosis(llm, case, answer)
            print(f"\nDiagnosis verdict: {diag['verdict'].upper()}")
            print(f"RAG diagnosis: {diag.get('rag_diagnosis', 'N/A')}")

        # Citation judge
        print(f"\nJudging citations...")
        cite = judge_citations(llm, answer, sources)

        if "error" not in cite:
            checked = cite.get("citations_checked", 0)
            supported = cite.get("supported", 0)
            unsupported = cite.get("unsupported", 0)
            uncited = cite.get("uncited_claims", 0)
            faith = cite.get("faithfulness_score", 0)
            print(f"  Citations checked: {checked}")
            print(f"  Supported: {supported}  Unsupported: {unsupported}")
            print(f"  Faithfulness: {faith:.0%}")
            print(f"  Uncited claims: {uncited}")
            if cite.get("details"):
                for d in cite["details"]:
                    if d["verdict"] == "unsupported":
                        print(f"    BAD: {d['citation']} — {d['claim'][:80]}...")
                        print(f"         Reason: {d['reason']}")
            if cite.get("uncited_examples"):
                for ex in cite["uncited_examples"][:3]:
                    print(f"    UNCITED: {ex[:100]}")
        else:
            print(f"  Citation judge parse error: {cite.get('raw', '')[:200]}")
            cite = {}

        results.append({
            "id": cid,
            "name": name,
            "expected": case["expected_diagnosis"],
            "answer": answer,
            "sources": sources,
            "time_s": rag_time,
            "n_docs": len(sources),
            # Diagnosis
            "verdict": diag.get("verdict"),
            "rag_diagnosis": diag.get("rag_diagnosis"),
            "reasoning": diag.get("reasoning"),
            # Citations
            "citations_checked": cite.get("citations_checked", 0),
            "citations_supported": cite.get("supported", 0),
            "citations_unsupported": cite.get("unsupported", 0),
            "uncited_claims": cite.get("uncited_claims", 0),
            "faithfulness_score": cite.get("faithfulness_score", 0),
            "citation_details": cite.get("details", []),
            "uncited_examples": cite.get("uncited_examples", []),
        })

    # ── Summary ──
    print(f"\n\n{'='*70}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*70}")
    print(f"{'ID':<4} {'Verdict':<10} {'Expected':<45} {'RAG Diagnosis'}")
    print("-" * 120)
    for r in results:
        v = (r.get("verdict") or "?").upper()
        print(f"{r['id']:<4} {v:<10} {r['expected']:<45} {r.get('rag_diagnosis', 'N/A')}")

    total = len(results)
    match = sum(1 for r in results if r.get("verdict") == "match")
    partial = sum(1 for r in results if r.get("verdict") == "partial")
    mismatch = sum(1 for r in results if r.get("verdict") == "mismatch")
    print(f"\nMatch: {match}/{total}  Partial: {partial}/{total}  Mismatch: {mismatch}/{total}")

    print(f"\n{'='*70}")
    print("CITATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'ID':<4} {'Checked':<9} {'OK':<5} {'Bad':<5} {'Uncited':<9} {'Faithfulness'}")
    print("-" * 60)
    total_checked = 0
    total_supported = 0
    total_unsupported = 0
    total_uncited = 0
    for r in results:
        c = r.get("citations_checked", 0)
        s = r.get("citations_supported", 0)
        u = r.get("citations_unsupported", 0)
        uc = r.get("uncited_claims", 0)
        f = r.get("faithfulness_score", 0)
        total_checked += c
        total_supported += s
        total_unsupported += u
        total_uncited += uc
        print(f"{r['id']:<4} {c:<9} {s:<5} {u:<5} {uc:<9} {f:.0%}")

    if total_checked > 0:
        avg_faith = total_supported / total_checked
        print(f"\nOverall: {total_supported}/{total_checked} supported ({avg_faith:.0%}), "
              f"{total_unsupported} unsupported, {total_uncited} uncited claims")

    out_path = "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, nargs="+", help="Case IDs to run (default: all)")
    parser.add_argument("--cite-only", action="store_true",
                        help="Only re-judge citations using cached RAG output from eval_results.json")
    args = parser.parse_args()
    run_eval(args.cases, cite_only=args.cite_only)
