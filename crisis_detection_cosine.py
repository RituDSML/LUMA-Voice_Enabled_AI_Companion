"""
crisis_detection_cosine.py — embedding-based (cosine-similarity) crisis
detection, proposed as a semantic upgrade to safety.py's substring-based
check_crisis().

Design principle: this reuses the EXACT SAME 40 tracked phrases already
in safety.py (CRISIS_KEYWORDS) as the reference set. It is deliberately
NOT a new, separately-curated or expanded phrase list — that constraint
means any recall improvement measured against the baseline can be
attributed to the matching METHOD (semantic embedding similarity vs.
exact substring presence), not to simply adding more tracked phrases.
This keeps the before/after comparison in Chapter 7 methodologically
clean.

Uses the same embedding model already used elsewhere in LUMA's stack
(all-MiniLM-L6-v2 — see luma_rag.py) so no new dependency is introduced.

Dependencies (already installed, per luma_rag.py):
    pip install sentence-transformers --break-system-packages
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from safety import CRISIS_KEYWORDS

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_phrase_embeddings = None  # L2-normalized, computed once and cached


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _get_phrase_embeddings():
    global _phrase_embeddings
    if _phrase_embeddings is None:
        model = _get_model()
        embeddings = model.encode(CRISIS_KEYWORDS, convert_to_numpy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        _phrase_embeddings = embeddings / norms
    return _phrase_embeddings


def max_similarity(text: str):
    """
    Returns (best_score, best_phrase): the highest cosine similarity
    between `text` and any of the 40 tracked crisis phrases, and which
    phrase produced it (useful for the per-utterance detail report and
    for manually sanity-checking borderline cases).
    """
    if not text:
        return 0.0, None

    model = _get_model()
    phrase_embeddings = _get_phrase_embeddings()

    query_embedding = model.encode([text], convert_to_numpy=True)[0]
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return 0.0, None
    query_embedding = query_embedding / query_norm

    similarities = phrase_embeddings @ query_embedding
    best_idx = int(np.argmax(similarities))
    return float(similarities[best_idx]), CRISIS_KEYWORDS[best_idx]


def check_crisis_cosine(text: str, threshold: float = 0.5) -> bool:
    """
    Drop-in alternative to safety.check_crisis(): returns True if the
    text's maximum cosine similarity to any tracked crisis phrase meets
    or exceeds `threshold`. The default threshold of 0.5 is a starting
    point only — run evaluate_crisis_cosine.py to pick a threshold
    empirically justified against the labelled test set, then update
    this default (or pass the chosen value explicitly wherever this is
    called from graph.py's crisis_detection_node).
    """
    score, _ = max_similarity(text)
    return score >= threshold


if __name__ == "__main__":
    # Quick manual sanity check against a couple of example utterances
    examples = [
        "I want to kill myself",
        "Everyone would be so much happier if I just wasn't around",
        "I'm anxious about my job interview tomorrow",
    ]
    print("Loading embedding model and encoding the 40 tracked phrases...")
    for ex in examples:
        score, phrase = max_similarity(ex)
        print(f"\n  \"{ex}\"")
        print(f"  max similarity: {score:.3f}  (closest tracked phrase: \"{phrase}\")")
