"""
test_tools.py — exercises tools.py against a real, isolated SQLite DB.
Covers:
  - FR-12: tool schemas are well-formed for all five tools
  - FR-13: TOOL_FUNCTIONS correctly executes and returns tool output
  - FR-19: schedule_reminder tool creates a reminder correctly
  - FR-21: list_reminders tool reflects created reminders
  - log_habit tool writes through to the DB correctly

NOTE: get_weather is excluded from live execution here — it calls
Open-Meteo over the network, and this sandbox's network allowlist does
not include that domain. Schema validation for get_weather IS covered
below; run test_tools.py on your own machine to also exercise the live
network call.
"""

import sys
import os
import tempfile
import shutil

TEST_DIR = tempfile.mkdtemp(prefix="luma_test_")
sys.path.insert(0, "/home/claude/luma_test")

import database
database.DB_PATH = os.path.join(TEST_DIR, "luma_test.db")
database.init_db()

import tools

results = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
    results.append(condition)
    return condition


print("=" * 60)
print("TOOL SCHEMA VALIDATION (FR-12)")
print("=" * 60)

EXPECTED_TOOLS = {"get_current_time", "get_weather", "log_habit", "schedule_reminder", "list_reminders"}
schema_names = {t["function"]["name"] for t in tools.TOOLS}
check("All 5 required tools present in TOOLS schema", schema_names == EXPECTED_TOOLS,
      f"got {schema_names}")

for t in tools.TOOLS:
    fn = t["function"]
    name = fn["name"]
    has_desc = bool(fn.get("description"))
    has_params = "parameters" in fn and "type" in fn["parameters"]
    check(f"{name}: has non-empty description", has_desc)
    check(f"{name}: has valid parameters schema", has_params)

check("TOOL_FUNCTIONS has an entry for every schema'd tool",
      set(tools.TOOL_FUNCTIONS.keys()) == EXPECTED_TOOLS)

print()
print("=" * 60)
print("TOOL EXECUTION (FR-13) — get_current_time")
print("=" * 60)

time_result = tools.TOOL_FUNCTIONS["get_current_time"]()
check("get_current_time() returns a non-empty string", isinstance(time_result, str) and len(time_result) > 0, time_result)

print()
print("=" * 60)
print("TOOL EXECUTION (FR-13) — log_habit")
print("=" * 60)

habit_result = tools.TOOL_FUNCTIONS["log_habit"](habit_type="smoking", quantity="1 cigarette", notes="after lunch")
check("log_habit tool returns confirmation", "Logged" in habit_result, habit_result)

logged = database.get_habit_logs()
check("log_habit tool call actually wrote to the database", len(logged) == 1 and logged[0]["habit_type"] == "smoking")

# Required-arg validation: habit_type is required, should raise or handle missing arg
try:
    tools.TOOL_FUNCTIONS["log_habit"](quantity="no habit type given")
    check("log_habit without required habit_type raises an error (as schema requires)", False,
          "call succeeded silently — schema says habit_type is required but function has no default enforcement")
except TypeError:
    check("log_habit without required habit_type raises an error (as schema requires)", True)

print()
print("=" * 60)
print("TOOL EXECUTION (FR-13, FR-19) — schedule_reminder")
print("=" * 60)

sched_result = tools.TOOL_FUNCTIONS["schedule_reminder"](label="take BP medication", hour=9, minute=0)
check("schedule_reminder tool returns confirmation with time", "9:00" in sched_result.replace("09:00", "9:00") or "09:00" in sched_result, sched_result)

invalid_result = tools.TOOL_FUNCTIONS["schedule_reminder"](label="bad time", hour=25, minute=0)
check("schedule_reminder rejects invalid hour (25) gracefully, not a crash",
      "Invalid" in invalid_result, invalid_result)

invalid_result2 = tools.TOOL_FUNCTIONS["schedule_reminder"](label="bad minute", hour=10, minute=99)
check("schedule_reminder rejects invalid minute (99) gracefully", "Invalid" in invalid_result2, invalid_result2)

print()
print("=" * 60)
print("TOOL EXECUTION (FR-13, FR-21) — list_reminders")
print("=" * 60)

list_result = tools.TOOL_FUNCTIONS["list_reminders"]()
check("list_reminders tool includes the reminder we just scheduled", "take BP medication" in list_result, list_result)
check("list_reminders tool does NOT include the rejected invalid reminders",
      "bad time" not in list_result and "bad minute" not in list_result)

print()
print("=" * 60)
print(f"{sum(results)}/{len(results)} checks passed")
print("=" * 60)

shutil.rmtree(TEST_DIR, ignore_errors=True)
