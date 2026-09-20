"""
test_tool_chaining.py — exercises the tool-calling loop in llm.get_response()
using a stubbed Groq client, WITHOUT needing real Groq API access. Also
stubs luma_rag (RAG behavior is not what this test targets — that's already
covered by your own test_llm_wiring.py).

Covers:
  - FR-14: chaining of at least one tool call within a single turn —
    tests that when the model requests MULTIPLE tool calls in one response,
    all of them are actually executed and their results fed back correctly.
  - The MAX_TOOL_ROUNDS safety cap (does not loop forever if the model
    keeps requesting tools without ever giving a final answer).
"""

import sys
import os
import types
import tempfile
import shutil

TEST_DIR = tempfile.mkdtemp(prefix="luma_test_")
sys.path.insert(0, "/home/claude/luma_test")

# --- Real database, isolated test file ---
import database
database.DB_PATH = os.path.join(TEST_DIR, "luma_test.db")
database.init_db()

# --- Stub luma_rag (not under test here) ---
class FakeLumaRAG:
    def __init__(self, *args, **kwargs):
        pass

    def retrieve(self, query, k=2):
        return []  # no retrieval — keeps this test focused on tool-calling

fake_rag_module = types.ModuleType("luma_rag")
fake_rag_module.LumaRAG = FakeLumaRAG
fake_rag_module.COPING_KB_PATH = "unused"
fake_rag_module.FIRST_AID_KB_PATH = "unused"
sys.modules["luma_rag"] = fake_rag_module

# --- Stub dotenv ---
fake_dotenv_module = types.ModuleType("dotenv")
fake_dotenv_module.load_dotenv = lambda: None
sys.modules["dotenv"] = fake_dotenv_module

# --- Stub groq with a scripted, multi-call sequence ---
class FakeToolCallFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = FakeToolCallFunction(name, arguments)


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeResponse:
    def __init__(self, message):
        self.choices = [FakeChoice(message)]


class ScriptedCompletions:
    """Returns a pre-scripted sequence of responses, one per call, so we
    can simulate the model requesting multiple tools then finishing."""
    def __init__(self, script):
        self.script = script
        self.call_count = 0
        self.calls_received = []  # record what messages/tools were passed each call

    def create(self, model, messages, tools=None, tool_choice=None):
        self.calls_received.append({"messages": list(messages), "tools": tools})
        if self.call_count >= len(self.script):
            # Fallback call (no tools param) at the end, if script exhausted
            response = FakeResponse(FakeMessage(content="[fallback] no more script"))
        else:
            response = self.script[self.call_count]
        self.call_count += 1
        return response


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeGroqClient:
    def __init__(self, completions):
        self.chat = FakeChat(completions)


import tools as real_tools  # real tool functions + schemas, real DB underneath


def run_chaining_test():
    print("=" * 60)
    print("FR-14: TOOL CHAINING TEST")
    print("=" * 60)

    # Script: turn 1 -> model requests TWO tool calls at once
    #         turn 2 -> model gives final text answer
    multi_tool_response = FakeMessage(
        content=None,
        tool_calls=[
            FakeToolCall("call_1", "get_current_time", "{}"),
            FakeToolCall("call_2", "schedule_reminder",
                         '{"label": "take BP medication", "hour": 9, "minute": 0}'),
        ]
    )
    final_response = FakeMessage(content="Done — I checked the time and set your reminder.")

    completions = ScriptedCompletions([
        FakeResponse(multi_tool_response),
        FakeResponse(final_response),
    ])

    fake_groq_module = types.ModuleType("groq")
    fake_groq_module.Groq = lambda api_key=None: FakeGroqClient(completions)
    sys.modules["groq"] = fake_groq_module

    # Fresh import of llm.py with our stubs in place
    if "llm" in sys.modules:
        del sys.modules["llm"]
    import llm

    result = llm.get_response("Can you tell me the time and set a 9am reminder for my BP meds?",
                               emotion="neutral")

    results = []

    def check(label, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
        results.append(condition)

    check("get_response() completed without raising", True)
    check("Final response text returned correctly",
          result == "Done — I checked the time and set your reminder.", result)
    check("Exactly 2 model calls made (1 tool-request round + 1 final)", completions.call_count == 2,
          f"got {completions.call_count}")

    # Verify BOTH tools actually executed (check real side effects in the DB)
    reminders_created = database.get_all_reminders()
    check("schedule_reminder tool call actually created a reminder in the DB",
          len(reminders_created) == 1 and reminders_created[0]["label"] == "take BP medication")

    # Verify the second API call's messages included both tool results
    second_call_messages = completions.calls_received[1]["messages"]
    tool_result_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    check("Both tool results were fed back to the model in the next call",
          len(tool_result_messages) == 2, f"got {len(tool_result_messages)}")

    tool_call_ids_responded = {m["tool_call_id"] for m in tool_result_messages}
    check("Tool results correctly matched back to their original call_ids by id",
          tool_call_ids_responded == {"call_1", "call_2"}, tool_call_ids_responded)

    return results


def run_max_rounds_test():
    print()
    print("=" * 60)
    print("MAX_TOOL_ROUNDS SAFETY CAP TEST")
    print("=" * 60)

    # Script: model NEVER gives a final answer, always requests a tool call.
    # Should stop after MAX_TOOL_ROUNDS and force a plain fallback call.
    infinite_tool_response = FakeMessage(
        content=None,
        tool_calls=[FakeToolCall("call_x", "get_current_time", "{}")]
    )
    completions = ScriptedCompletions([FakeResponse(infinite_tool_response)] * 10)
    # Make the fallback call (called without `tools=`) return a distinct message
    original_create = completions.create
    def create_with_fallback_detection(model, messages, tools=None, tool_choice=None):
        if tools is None:
            completions.call_count += 1
            completions.calls_received.append({"messages": list(messages), "tools": tools})
            return FakeResponse(FakeMessage(content="[forced fallback response]"))
        return original_create(model, messages, tools=tools, tool_choice=tool_choice)
    completions.create = create_with_fallback_detection

    fake_groq_module = types.ModuleType("groq")
    fake_groq_module.Groq = lambda api_key=None: FakeGroqClient(completions)
    sys.modules["groq"] = fake_groq_module

    del sys.modules["llm"]
    import llm

    result = llm.get_response("keep calling tools forever", emotion="neutral")

    results = []

    def check(label, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
        results.append(condition)

    check("Loop terminates (does not hang) when model never stops requesting tools", True)
    check(f"Stopped after exactly MAX_TOOL_ROUNDS ({llm.MAX_TOOL_ROUNDS}) + 1 fallback call",
          completions.call_count == llm.MAX_TOOL_ROUNDS + 1, f"got {completions.call_count}")
    check("Forced fallback call returned its response correctly",
          result == "[forced fallback response]", result)

    return results


all_results = run_chaining_test() + run_max_rounds_test()

print()
print("=" * 60)
print(f"{sum(all_results)}/{len(all_results)} checks passed")
print("=" * 60)

shutil.rmtree(TEST_DIR, ignore_errors=True)
