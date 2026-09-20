"""
luma_rag.py

A minimal, dependency-light RAG module for LUMA.

What this does:
1. Loads a JSON knowledge base of coping-strategy entries.
2. Embeds each entry using a local sentence-transformer model (no API calls).
3. Builds a FAISS index for fast similarity search.
4. Exposes a single function, `retrieve(query, k)`, that returns the top-k
   most relevant entries for a given user message.

How this plugs into your existing LUMA architecture:
- Call `retrieve()` as a new node in your LangGraph pipeline, positioned
  AFTER the crisis-check (crisis-check should always run first, untouched)
  and BEFORE the final LLM call.
- Feed the retrieved text into your LLM's system/context prompt, e.g.:

    relevant_strategies = retrieve(user_message, k=2)
    context_block = "\n\n".join(r["content"] for r in relevant_strategies)
    # then include context_block in the prompt sent to your Groq LLM call

Dependencies (already installed in this environment):
    pip install sentence-transformers faiss-cpu --break-system-packages
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- Configuration ---
COPING_KB_PATH = os.path.join(os.path.dirname(__file__), "coping_strategies.json")
FIRST_AID_KB_PATH = os.path.join(os.path.dirname(__file__), "first_aid.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, free, runs locally


class LumaRAG:
    """
    Generic RAG wrapper — pass any kb_path (a JSON list of {id, title,
    content} entries) to build a retriever over that specific knowledge
    base. Create one instance per knowledge base (e.g. one for coping
    strategies, one for first aid) so retrieval stays scoped and doesn't
    mix unrelated content.
    """
    def __init__(self, kb_path: str):
        self.kb_path = kb_path
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.entries = self._load_knowledge_base()
        self.index = self._build_index()

    def _load_knowledge_base(self):
        with open(self.kb_path, "r") as f:
            entries = json.load(f)
        return entries

    def _build_index(self):
        # Embed each entry's content (title + content combined gives the
        # embedding a bit more signal about what the entry is about)
        texts = [f"{e['title']}: {e['content']}" for e in self.entries]
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        # Normalize so we can use inner product as cosine similarity
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # IP = inner product (cosine, since normalized)
        index.add(embeddings)
        return index

    def retrieve(self, query: str, k: int = 2):
        """
        Returns the top-k most relevant knowledge base entries for a query.

        Args:
            query: the user's message (or a summarized version of it)
            k: number of results to return

        Returns:
            List of dicts, each with 'id', 'title', 'content', and 'score'
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.entries[idx])
            entry["score"] = float(score)
            results.append(entry)
        return results


# --- Simple demo / sanity check ---
if __name__ == "__main__":
    coping_rag = LumaRAG(COPING_KB_PATH)
    first_aid_rag = LumaRAG(FIRST_AID_KB_PATH)

    coping_queries = [
        "I keep spiraling into panic before my exams",
        "I can't stop thinking the same worries over and over",
        "I feel like giving up and just isolating from everyone",
    ]
    first_aid_queries = [
        "I cut my hand and it's bleeding a lot",
        "my kid just burned their hand on the stove",
        "someone is choking at dinner",
    ]

    print("=== Coping strategies ===")
    for q in coping_queries:
        print(f"\nQuery: {q}")
        for r in coping_rag.retrieve(q, k=2):
            print(f"  [{r['score']:.3f}] {r['title']}")

    print("\n=== First aid ===")
    for q in first_aid_queries:
        print(f"\nQuery: {q}")
        for r in first_aid_rag.retrieve(q, k=2):
            print(f"  [{r['score']:.3f}] {r['title']}")
