"""
test_llm_wiring.py — verifies the RAG gating/wiring logic in llm.py WITHOUT
needing real Groq API access or a real embedding model download (this
sandbox can't reach either). This only tests: does retrieval fire for the
right concern tiers, and does it stay silent for everything else?

Run this for real (with actual Groq/embeddings) on your own machine once
you've confirmed the logic here looks right.
"""

import sys
import types
import numpy as np

# --- Stub out sentence_transformers before luma_rag imports it ---
class FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, convert_to_numpy=True):
        # Deterministic fake embeddings based on text hash, just so
        # retrieve() runs end-to-end without a real model download.
        rng = np.random.RandomState(abs(hash(tuple(texts))) % (2**31))
        return rng.rand(len(texts), 384).astype("float32")

fake_st_module = types.ModuleType("sentence_transformers")
fake_st_module.SentenceTransformer = FakeSentenceTransformer
sys.modules["sentence_transformers"] = fake_st_module

# --- Stub out groq before llm.py imports it ---
class FakeMessage:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None

class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)

class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]

class FakeCompletions:
    def create(self, model, messages, tools=None, tool_choice=None):
        # Just echo back confirmation of what system prompt it received,
        # so we can inspect gating behavior without a real API call.
        system_prompt = messages[0]["content"]
        return FakeResponse(f"[STUBBED RESPONSE] system_prompt_length={len(system_prompt)}")

class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()

class FakeGroq:
    def __init__(self, api_key=None):
        self.chat = FakeChat()

fake_groq_module = types.ModuleType("groq")
fake_groq_module.Groq = FakeGroq
sys.modules["groq"] = fake_groq_module

# --- Stub out tools.py (not relevant to this test) ---
fake_tools_module = types.ModuleType("tools")
fake_tools_module.TOOLS = []
fake_tools_module.TOOL_FUNCTIONS = {}
sys.modules["tools"] = fake_tools_module

# --- Stub out dotenv (not relevant to this test) ---
fake_dotenv_module = types.ModuleType("dotenv")
fake_dotenv_module.load_dotenv = lambda: None
sys.modules["dotenv"] = fake_dotenv_module

# Now safe to import the real llm.py — it will use our stubs above
sys.path.insert(0, "/home/claude")
import llm


def check(label, user_message, expected_concern, expect_retrieval):
    concern = llm.detect_concern_tier(user_message)
    retrieval_block = llm._build_retrieval_block(user_message, concern)
    has_retrieval = bool(retrieval_block.strip())

    concern_ok = (concern == expected_concern)
    retrieval_ok = (has_retrieval == expect_retrieval)
    status = "PASS" if (concern_ok and retrieval_ok) else "FAIL"

    print(f"[{status}] {label}")
    print(f"  message: {user_message!r}")
    print(f"  detected concern: {concern} (expected: {expected_concern})")
    print(f"  retrieval fired: {has_retrieval} (expected: {expect_retrieval})")
    if has_retrieval:
        preview = retrieval_block[:80].replace(chr(10), " ")
        print(f"  retrieval preview: {preview}...")
    print()
    return status == "PASS"


results = []

results.append(check(
    "Unrelated message should NOT trigger any retrieval",
    "What time is it right now?",
    expected_concern=None,
    expect_retrieval=False,
))

results.append(check(
    "Book recommendation should NOT trigger any retrieval",
    "Can you recommend a good book to read this week?",
    expected_concern=None,
    expect_retrieval=False,
))

results.append(check(
    "Mental health language SHOULD trigger coping-strategy retrieval",
    "I feel so hopeless and overwhelmed today",
    expected_concern="mental_health",
    expect_retrieval=True,
))

results.append(check(
    "Physical health language SHOULD trigger first-aid retrieval",
    "I cut my finger and it's bleeding a lot, not feeling well",
    expected_concern="physical_health",
    expect_retrieval=True,
))

results.append(check(
    "Habit tracking should NOT trigger RAG retrieval (different tier, no KB for this)",
    "I smoked 2 cigarettes after lunch",
    expected_concern="habit_tracking",
    expect_retrieval=False,
))

print("=" * 50)
print(f"{sum(results)}/{len(results)} checks passed")
print("=" * 50)

# --- Also verify end-to-end get_response() runs without crashing ---
print("\nRunning get_response() end-to-end with stubbed Groq/embeddings:\n")
for msg, emo in [
    ("What time is it right now?", "neutral"),
    ("I feel so hopeless today", "negative"),
    ("I cut my finger and it's bleeding a lot", "neutral"),
]:
    try:
        result = llm.get_response(msg, emotion=emo)
        print(f"OK  | {msg!r} -> {result}")
    except Exception as e:
        print(f"FAIL | {msg!r} raised: {e}")
