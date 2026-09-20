"""
rag_eval.py — RAG retrieval quantitative evaluation for LUMA (Ch.7/8)

Computes hit-rate@k for k in {1, 2, 3, 5} across the 30 grounded questions
in rag_eval_questions.json, using the SAME retrieval setup as production
(LumaRAG from luma_rag.py, same embedding model, same KB files).

HOW TO RUN:
1. Copy this file and rag_eval_questions.json into the same folder as your
   luma_rag.py, coping_strategies.json, and first_aid.json (i.e. your
   actual LUMA project folder — this reuses your real KBs and real index,
   not a copy).
2. Run: python rag_eval.py
3. Results print to console and are also written to rag_eval_results.csv
   and rag_eval_summary.md (a ready-to-paste table for the thesis).

WHY hit-rate@k this way:
A "hit" for question i at cutoff k means at least one of that question's
expected_ids appears in the top-k retrieved results for its own KB
(coping or first_aid — queries are only ever run against the KB they're
labelled for, matching how LUMA itself scopes retrieval by concern tier).
hit-rate@k = (number of questions with a hit at that k) / (total questions).
"""

import json
import csv
from luma_rag import LumaRAG, COPING_KB_PATH, FIRST_AID_KB_PATH

K_VALUES = [1, 2, 3, 5]


def load_questions(path="rag_eval_questions.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate():
    print("Loading knowledge bases and building indices (same as production)...")
    coping_rag = LumaRAG(COPING_KB_PATH)
    first_aid_rag = LumaRAG(FIRST_AID_KB_PATH)
    rags = {"coping": coping_rag, "first_aid": first_aid_rag}

    questions = load_questions()
    max_k = max(K_VALUES)

    rows = []
    for q in questions:
        rag = rags[q["kb"]]
        results = rag.retrieve(q["query"], k=max_k)
        retrieved_ids = [r["id"] for r in results]
        retrieved_titles = [r["title"] for r in results]

        # For each cutoff k, was any expected id within the first k results?
        hits_at_k = {}
        for k in K_VALUES:
            top_k_ids = retrieved_ids[:k]
            hit = any(exp_id in top_k_ids for exp_id in q["expected_ids"])
            hits_at_k[k] = hit

        # Rank of the first correct hit (1-indexed), or None if never found
        first_hit_rank = None
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in q["expected_ids"]:
                first_hit_rank = rank
                break

        row = {
            "id": q["id"],
            "kb": q["kb"],
            "difficulty": q["difficulty"],
            "query": q["query"],
            "expected_ids": ";".join(q["expected_ids"]),
            "retrieved_top5": " | ".join(retrieved_titles[:5]),
            "first_hit_rank": first_hit_rank if first_hit_rank else "NOT FOUND",
        }
        for k in K_VALUES:
            row[f"hit@{k}"] = hits_at_k[k]
        rows.append(row)

    # --- Write per-question CSV ---
    csv_path = "rag_eval_results.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-question results written to {csv_path}")

    # --- Aggregate hit-rate@k, overall and by difficulty ---
    def hit_rate(subset, k):
        if not subset:
            return None
        return sum(1 for r in subset if r[f"hit@{k}"]) / len(subset)

    overall = rows
    easy = [r for r in rows if r["difficulty"] == "easy"]
    hard = [r for r in rows if r["difficulty"] == "hard"]

    summary_lines = []
    summary_lines.append("# LUMA RAG Retrieval Evaluation — Hit-Rate@k\n")
    summary_lines.append(f"Total questions: {len(rows)} ({len(easy)} easy, {len(hard)} hard)\n")
    summary_lines.append("| k | Overall hit-rate | Easy (direct phrasing) | Hard (paraphrased) |")
    summary_lines.append("|---|---|---|---|")
    for k in K_VALUES:
        o = hit_rate(overall, k)
        e = hit_rate(easy, k)
        h = hit_rate(hard, k)
        summary_lines.append(f"| {k} | {o:.2f} | {e:.2f} | {h:.2f} |")

    not_found = [r for r in rows if r["first_hit_rank"] == "NOT FOUND"]
    if not_found:
        summary_lines.append(f"\n**Never retrieved within k={max_k}:** " +
                              ", ".join(r["id"] for r in not_found))

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    with open("rag_eval_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print("\nSummary table written to rag_eval_summary.md — paste directly into Ch.7.")


if __name__ == "__main__":
    evaluate()
