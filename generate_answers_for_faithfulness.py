"""
generate_answers_for_faithfulness.py — Step 2 of the RAG quantitative eval.

Runs each of the 30 questions through the REAL retrieval + generation
path (detect_concern_tier -> _build_retrieval_block -> get_response, all
imported directly from your existing llm.py) and saves query, retrieved
context, and generated answer to a CSV for manual faithfulness scoring.

This calls the real Groq LLM once per question (30 calls total), so it
needs your existing environment/API key set up exactly as it is for
main.py to run normally.

HOW TO RUN:
1. Copy this file and rag_eval_questions.json into your LUMA project
   folder (same place as llm.py, luma_rag.py, safety.py, etc.).
2. Run: python generate_answers_for_faithfulness.py
3. Open the resulting rag_faithfulness_scoring.csv and fill in the
   'faithfulness_score' column by hand for each row (see the rubric in
   README_rag_eval.md) before reporting the results in Ch.7/8.
"""

import json
import csv
import time
from llm import get_response, detect_concern_tier, _build_retrieval_block


def load_questions(path="rag_eval_questions.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run():
    questions = load_questions()

    out_path = "rag_faithfulness_scoring.csv"
    fieldnames = ["id", "kb", "difficulty", "query", "detected_concern_tier",
                  "expected_ids", "retrieved_context", "generated_answer",
                  "faithfulness_score", "notes"]

    # utf-8 explicitly — Windows' default (cp1252) can't encode characters
    # some LLM answers contain (e.g. U+2011 non-breaking hyphen), and
    # writing row-by-row means a failure partway through doesn't lose
    # every call made before it.
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, q in enumerate(questions, start=1):
            print(f"[{i}/{len(questions)}] {q['id']}: {q['query'][:60]}...")

            concern = detect_concern_tier(q["query"])
            retrieved_context = _build_retrieval_block(q["query"], concern)

            # emotion/history/long_term_summary left neutral/empty — this
            # isolates the RAG contribution rather than mixing in emotion-
            # dependent tone or conversational memory effects.
            answer = get_response(
                q["query"],
                emotion=None,
                history=[],
                long_term_summary=None,
            )

            row = {
                "id": q["id"],
                "kb": q["kb"],
                "difficulty": q["difficulty"],
                "query": q["query"],
                "detected_concern_tier": concern,
                "expected_ids": ";".join(q["expected_ids"]),
                "retrieved_context": retrieved_context,
                "generated_answer": answer,
                "faithfulness_score": "",   # fill in by hand: Faithful / Partial / Unfaithful
                "notes": "",
            }
            writer.writerow(row)
            f.flush()  # ensure this row is safely on disk before the next call

            time.sleep(0.5)  # small pause to be gentle on the Groq API

    print(f"\nDone. {len(questions)} rows written to {out_path}.")
    print("Open it and fill in 'faithfulness_score' for each row (see README_rag_eval.md rubric).")


if __name__ == "__main__":
    run()
