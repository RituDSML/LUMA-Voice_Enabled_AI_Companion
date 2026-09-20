# LUMA RAG Quantitative Evaluation — How to Run and Write Up

Replaces the qualitative RAG assessment with a fixed set of 30 grounded
questions scored for **retrieval hit-rate@k** and **answer faithfulness**,
per supervisor feedback.

## Files

- `rag_eval_questions.json` — 30 questions (12 first-aid, 18 coping),
  each with the KB it targets and the `expected_ids` a correct retrieval
  should surface. Half are "easy" (near the entry's own phrasing), half
  are "hard" (paraphrased/indirect — testing whether retrieval only works
  on keyword overlap or genuinely captures meaning).
- `rag_eval.py` — Part 1: retrieval evaluation. Runs your real `LumaRAG`
  (same embedding model, same KBs) against all 30 questions and computes
  hit-rate@1/2/3/5, overall and split by easy/hard.
- `generate_answers_for_faithfulness.py` — Part 2: generates real
  answers. Runs each question through your real `detect_concern_tier` →
  `_build_retrieval_block` → `get_response` path and saves
  query/context/answer triples to a CSV for you to score by hand.

## Step-by-step

1. Copy all three files above into your actual LUMA project folder
   (same directory as `luma_rag.py`, `llm.py`, `coping_strategies.json`,
   `first_aid.json`) — the scripts import your real modules directly, so
   they need to sit alongside them.
2. Run `python rag_eval.py` → produces `rag_eval_results.csv` (per-question
   detail) and `rag_eval_summary.md` (the table to paste into Ch.7).
3. Run `python generate_answers_for_faithfulness.py` → produces
   `rag_faithfulness_scoring.csv` (this one calls the real Groq LLM 30
   times, so it'll take a minute or two and uses your API quota).
4. Open `rag_faithfulness_scoring.csv` and fill in `faithfulness_score`
   for each of the 30 rows using the rubric below. Do this yourself,
   reading the `retrieved_context` and `generated_answer` columns side by
   side — this is a judgment call, and a human-scored rubric is more
   defensible in a thesis than an LLM grading its own output, unless you
   want to explicitly report an LLM-as-judge as a *secondary*, faster
   check (see note at the end).

## Faithfulness rubric (score each row)

For each question, read the `retrieved_context` and `generated_answer`
side by side and ask: **is every claim in the answer actually supported
by the retrieved context (or by extremely common-sense filler, like "I
hope that helps"), or does the answer add specifics that aren't in the
context at all?**

- **Faithful** — every substantive claim in the answer traces back to
  the retrieved context. Paraphrasing/rewording is fine; adding new
  *facts* not present in the context is not.
- **Partial** — the core advice matches the retrieved context, but the
  answer adds at least one specific detail (a number, a claim, a step)
  that isn't actually in the retrieved text.
- **Unfaithful** — the answer's core advice contradicts the retrieved
  context, or ignores it and answers from the model's general knowledge
  instead, or the context was empty/irrelevant but the answer confidently
  gave specific advice anyway.

Report the **count/percentage in each of the three categories** as your
faithfulness result — don't collapse it to a single number, since the
three-way breakdown itself is informative (e.g. "24/30 Faithful, 5/30
Partial, 1/30 Unfaithful" tells a reader more than a bare average would).

## Writing this up in Ch.7/8

**Ch.7 (Results):**
> A fixed set of 30 grounded questions (Appendix X) — 12 targeting the
> first-aid knowledge base and 18 targeting the coping-strategies
> knowledge base, split evenly between direct and paraphrased phrasing —
> was used to evaluate retrieval quality quantitatively. Hit-rate@k
> results are shown in Table X [paste rag_eval_summary.md's table here].
> Faithfulness scoring of the corresponding generated answers found
> [X/30] Faithful, [X/30] Partial, and [X/30] Unfaithful (Table Y).

**Ch.8 (Discussion):** this is where the easy-vs-hard split becomes
useful — if hit-rate@k is much lower on the "hard" (paraphrased)
questions than "easy" ones, that's a direct, citable finding about
retrieval robustness to phrasing, parallel to the crisis-detection
paraphrase-recall gap your supervisor already flagged — worth explicitly
drawing that parallel, since both stem from the same underlying issue
(surface-form matching vs. semantic matching), even though the RAG layer
already uses embeddings while crisis detection currently doesn't.

## Optional: LLM-as-judge as a secondary check

If you want extra rigor (not required, but strengthens the section), you
could additionally have an LLM score the same 30 rows using the identical
rubric above as a prompt, and report **agreement between your manual
scores and the LLM's** (e.g. "manual and LLM-judge scoring agreed on
27/30 faithfulness ratings"). This is a nice-to-have, not a replacement —
report your own manual scores as the primary result either way.
