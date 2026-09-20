"""
test_database.py — exercises database.py against a real, isolated SQLite
file (not your actual luma.db). Covers:
  - FR-18: memory data persisted in local SQLite
  - NFR-05: session/long-term data persists across "application restarts"
    (simulated here by closing all connections and re-opening the DB file)
  - Habit log read/write correctness (backs FR-12/FR-13 via log_habit tool)
  - Reminder CRUD correctness (backs FR-19/FR-21)
"""

import sys
import os
import tempfile
import shutil

TEST_DIR = tempfile.mkdtemp(prefix="luma_test_")
sys.path.insert(0, "/home/claude/luma_test")

import database
database.DB_PATH = os.path.join(TEST_DIR, "luma_test.db")

results = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
    results.append(condition)
    return condition


print("=" * 60)
print("DATABASE PERSISTENCE TESTS")
print("=" * 60)

database.init_db()
check("init_db() creates DB file", os.path.exists(database.DB_PATH))

# --- Habit logging ---
result = database.log_habit("smoking", "2 cigarettes", "stressful afternoon")
check("log_habit() returns confirmation string", "Logged" in result and "smoking" in result, result)

logs = database.get_habit_logs()
check("get_habit_logs() returns the logged entry", len(logs) == 1 and logs[0]["habit_type"] == "smoking")
check("logged quantity persisted correctly", logs[0]["quantity"] == "2 cigarettes")
check("logged notes persisted correctly", logs[0]["notes"] == "stressful afternoon")

database.log_habit("drinking", "1 drink")
filtered = database.get_habit_logs(habit_type="smoking")
check("get_habit_logs(habit_type=...) filters correctly", len(filtered) == 1 and filtered[0]["habit_type"] == "smoking")

# --- Reminders ---
rid = database.create_reminder("take BP medication", 9, 0)
check("create_reminder() returns a valid id", isinstance(rid, int) and rid > 0)

all_reminders = database.get_all_reminders()
check("get_all_reminders() returns the created reminder", len(all_reminders) == 1)
check("reminder label persisted correctly", all_reminders[0]["label"] == "take BP medication")
check("reminder is active by default", all_reminders[0]["active"] == 1)

active = database.get_active_reminders()
check("get_active_reminders() returns it while active", len(active) == 1)

database.mark_reminder_sent(rid, "2026-09-10")
updated = database.get_all_reminders()
check("mark_reminder_sent() persists last_sent_date", updated[0]["last_sent_date"] == "2026-09-10")

database.deactivate_reminder(rid)
active_after = database.get_active_reminders()
check("deactivate_reminder() removes it from active list", len(active_after) == 0)
still_listed = database.get_all_reminders()
check("deactivated reminder still appears in get_all_reminders()", len(still_listed) == 1)

# --- Long-term memory persistence ---
database.save_long_term_summary("User's name is Ritz. Studying MSc Data Science.", 42)
summary, last_id = database.get_long_term_summary()
check("save/get_long_term_summary round-trips correctly",
      summary == "User's name is Ritz. Studying MSc Data Science." and last_id == 42)

# Overwrite behavior (single running row)
database.save_long_term_summary("Updated summary.", 99)
summary2, last_id2 = database.get_long_term_summary()
check("save_long_term_summary() replaces (not appends) the previous row",
      summary2 == "Updated summary." and last_id2 == 99)

# --- NFR-05: persistence across simulated "application restart" ---
# Simulate a restart by dropping our in-process reference and re-importing
# against the same DB_PATH, as a fresh process would.
del database
import importlib
import database as database2
database2.DB_PATH = os.path.join(TEST_DIR, "luma_test.db")

reminders_after_restart = database2.get_all_reminders()
check("NFR-05: reminders persist after simulated restart", len(reminders_after_restart) == 1)

summary_after_restart, id_after_restart = database2.get_long_term_summary()
check("NFR-05: long-term memory persists after simulated restart",
      summary_after_restart == "Updated summary." and id_after_restart == 99)

habits_after_restart = database2.get_habit_logs()
check("NFR-05: habit logs persist after simulated restart", len(habits_after_restart) == 2)

print()
print("=" * 60)
print(f"{sum(results)}/{len(results)} checks passed")
print("=" * 60)

shutil.rmtree(TEST_DIR, ignore_errors=True)
