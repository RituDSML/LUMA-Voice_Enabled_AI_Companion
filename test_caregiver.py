"""
test_caregiver.py — exercises save_caregiver_info() / get_caregiver_info()
against a real, isolated SQLite file (not your actual luma.db).

This closes a gap in the earlier test_database.py suite, which covered
habit logs, reminders, and long-term memory persistence but never
actually tested the caregiver_info data layer.

Scope note: this tests ONLY the database.py functions themselves. It does
NOT test (and cannot test, since they don't exist yet) any WebSocket
handler, frontend button, or end-to-end retrieval flow — see Chapter 6,
Section 6.5.1 for that distinction.
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

results = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
    results.append(condition)
    return condition


print("=" * 60)
print("CAREGIVER_INFO DATA LAYER TESTS")
print("=" * 60)

# --- No record saved yet ---
result = database.get_caregiver_info()
check("get_caregiver_info() returns None when nothing has been saved", result is None, result)

# --- Save a record ---
database.save_caregiver_info("Nikhil", "+91 9999999999", "friend")
result = database.get_caregiver_info()
check("get_caregiver_info() returns a dict after saving", isinstance(result, dict), result)
check("saved name persisted correctly", result["name"] == "Nikhil" if result else False)
check("saved phone persisted correctly", result["phone"] == "+91 9999999999" if result else False)
check("saved relationship persisted correctly", result["relationship"] == "friend" if result else False)

# --- Single-record behavior: saving again REPLACES, doesn't add a second row ---
database.save_caregiver_info("Mom", "+91 8606174275", "mother")
result2 = database.get_caregiver_info()
check("saving a second time replaces the record (single-record design)",
      result2["name"] == "Mom" and result2["phone"] == "+91 8606174275")

conn = database.get_connection()
count = conn.execute("SELECT COUNT(*) as c FROM caregiver_info").fetchone()["c"]
conn.close()
check("only one row ever exists in caregiver_info, even after multiple saves", count == 1, f"found {count} rows")

# --- Persistence across simulated restart ---
del database
import database as database2
database2.DB_PATH = os.path.join(TEST_DIR, "luma_test.db")
result3 = database2.get_caregiver_info()
check("caregiver record persists after simulated application restart",
      result3 is not None and result3["name"] == "Mom")

print()
print("=" * 60)
print(f"{sum(results)}/{len(results)} checks passed")
print("=" * 60)
print("\nNote: this confirms the DATA LAYER only. save_caregiver_info() is not")
print("currently called anywhere by any WebSocket handler or frontend code —")
print("see Chapter 6, Section 6.5.1 for that gap, which this test does not")
print("and cannot close.")

shutil.rmtree(TEST_DIR, ignore_errors=True)
