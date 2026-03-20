"""
Generate a human-readable citation audit document from eval_results.json.

For each case, prints the full answer with each [Source N] citation followed
by the actual source text, so a human reviewer can verify faithfulness.

Usage:
    python citation_audit.py                # all cases
    python citation_audit.py --cases 1 10   # specific cases
    python citation_audit.py --out audit.md # write to file
"""

import argparse
import json
import re
import sys


def generate_audit(case_ids=None, out_file=None):
    with open("eval_results.json") as f:
        results = json.load(f)

    if case_ids:
        results = [r for r in results if r["id"] in case_ids]

    lines = []

    for r in results:
        answer = r.get("answer", "")
        sources = r.get("sources", [])
        if not answer or not sources:
            continue

        lines.append(f"{'='*80}")
        lines.append(f"CASE {r['id']}: {r['name']}")
        lines.append(f"Expected: {r['expected']}")
        lines.append(f"RAG diagnosis: {r.get('rag_diagnosis', 'N/A')}")
        lines.append(f"Diagnosis verdict: {r.get('verdict', 'N/A').upper()}")
        lines.append(f"{'='*80}")
        lines.append("")

        # Print the full answer
        lines.append("FULL ANSWER:")
        lines.append("-" * 40)
        lines.append(answer)
        lines.append("")

        # Print each source
        lines.append("RETRIEVED SOURCES:")
        lines.append("-" * 40)
        for i, src in enumerate(sources):
            lines.append(f"\n[Source {i+1}] {src['document']} — Page {src['page']}")
            lines.append("~" * 60)
            # Trim the metadata prefix from the text
            text = src["text"]
            text = re.sub(r'^\[Document:.*?\]\n', '', text)
            # Truncate very long sources for readability
            if len(text) > 1500:
                text = text[:1500] + "\n[... truncated ...]"
            lines.append(text)
        lines.append("")

        # Print citation-by-citation breakdown
        lines.append("CITATION-BY-CITATION AUDIT:")
        lines.append("-" * 40)

        # Find all [Source N] references in the answer
        # Split answer into segments around citations
        parts = re.split(r'(\[Source[s]?\s+[\d,\s]+\])', answer)

        citation_num = 0
        for i, part in enumerate(parts):
            # Check if this part is a citation reference
            source_match = re.match(r'\[Sources?\s+([\d,\s]+)\]', part)
            if source_match:
                citation_num += 1
                nums = [int(n.strip()) for n in source_match.group(1).split(",")]

                # Get the claim (text before this citation)
                claim = parts[i-1].strip() if i > 0 else ""
                # Take last sentence/clause as the claim
                claim_sentences = re.split(r'(?<=[.;])\s+', claim)
                claim_short = claim_sentences[-1] if claim_sentences else claim
                if len(claim_short) > 200:
                    claim_short = "..." + claim_short[-200:]

                lines.append(f"\n--- Citation #{citation_num}: {part} ---")
                lines.append(f"CLAIM: {claim_short}")
                lines.append("")
                for n in nums:
                    if 0 < n <= len(sources):
                        src = sources[n-1]
                        src_text = src["text"]
                        src_text = re.sub(r'^\[Document:.*?\]\n', '', src_text)
                        if len(src_text) > 800:
                            src_text = src_text[:800] + "\n[... truncated ...]"
                        lines.append(f"SOURCE {n} ({src['document']}, p.{src['page']}):")
                        lines.append(src_text)
                    else:
                        lines.append(f"SOURCE {n}: NOT FOUND (only {len(sources)} sources)")
                    lines.append("")
                lines.append(f"YOUR VERDICT: [ ] Supported  [ ] Unsupported  [ ] Partially supported")
                lines.append("")

        lines.append("")
        lines.append(f"SUMMARY FOR CASE {r['id']}:")
        lines.append(f"Total citations: {citation_num}")
        lines.append(f"Auto-judge said: {r.get('citations_supported', '?')} supported, "
                     f"{r.get('citations_unsupported', '?')} unsupported, "
                     f"faithfulness {r.get('faithfulness_score', 0):.0%}")
        lines.append("")
        lines.append("")

    output = "\n".join(lines)

    if out_file:
        with open(out_file, "w") as f:
            f.write(output)
        print(f"Audit written to {out_file}")
    else:
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, nargs="+")
    parser.add_argument("--out", type=str, default="citation_audit.txt")
    args = parser.parse_args()
    generate_audit(args.cases, args.out)
