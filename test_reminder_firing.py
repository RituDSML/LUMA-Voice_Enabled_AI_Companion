"""
test_reminder_firing.py — exercises reminders._check_and_fire_reminders()
directly, WITHOUT waiting on the real APScheduler 1-minute interval (that
would make tests slow and non-deterministic). We call the internal check
function directly and control "now" via monkeypatching datetime, which
tests the actual firing/dedup logic that the real scheduler calls every
minute — this is the part that matters, not the scheduling wrapper itself.

Covers:
  - FR-20: reminder fires at the intended time
  - Correct suppression of duplicate same-day firing
  - Delivery callback receives the right reminder_id/message/log_id
"""

import sys
import os
import tempfile
import shutil
from datetime import datetime as real_datetime

TEST_DIR = tempfile.mkdtemp(prefix="luma_test_")
sys.path.insert(0, "/home/claude/luma_test")

import database
database.DB_PATH = os.path.join(TEST_DIR, "luma_test.db")
database.init_db()

import reminders

results = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
    results.append(condition)
    return condition


class FrozenDateTime(real_datetime):
    """Lets us control what reminders._check_and_fire_reminders() sees as 'now'."""
    _frozen = None

    @classmethod
    def now(cls, tz=None):
        return cls._frozen


print("=" * 60)
print("REMINDER FIRING LOGIC TESTS")
print("=" * 60)

# Set up one reminder at 09:00
rid = database.create_reminder("take BP medication", 9, 0)

delivered_calls = []


def fake_delivery_callback(reminder_id, message, log_id):
    delivered_calls.append((reminder_id, message, log_id))


reminders.set_delivery_callback(fake_delivery_callback)

# --- Case 1: current time does NOT match reminder time -> should not fire ---
reminders.datetime = FrozenDateTime
FrozenDateTime._frozen = real_datetime(2026, 9, 10, 8, 59)
reminders._check_and_fire_reminders()
check("Reminder does NOT fire before its scheduled time (08:59 vs 09:00)",
      len(delivered_calls) == 0)

# --- Case 2: current time MATCHES reminder time -> should fire exactly once ---
FrozenDateTime._frozen = real_datetime(2026, 9, 10, 9, 0)
reminders._check_and_fire_reminders()
check("Reminder fires at its scheduled time (09:00)", len(delivered_calls) == 1)

if delivered_calls:
    fired_rid, fired_msg, fired_log_id = delivered_calls[0]
    check("Delivered callback received correct reminder_id", fired_rid == rid)
    check("Delivered message includes the reminder label", "take BP medication" in fired_msg)
    check("Delivered callback received a valid log_id", isinstance(fired_log_id, int) and fired_log_id > 0)

# --- Case 3: same minute fires again (e.g. scheduler tick runs twice) -> should NOT double-fire ---
reminders._check_and_fire_reminders()
check("Reminder does NOT fire twice on the same day (dedup via last_sent_date)",
      len(delivered_calls) == 1)

# --- Case 4: next day, same time -> SHOULD fire again ---
FrozenDateTime._frozen = real_datetime(2026, 9, 11, 9, 0)
reminders._check_and_fire_reminders()
check("Reminder fires again the next day at the same time", len(delivered_calls) == 2)

# --- Case 5: deactivated reminder should never fire ---
database.deactivate_reminder(rid)
FrozenDateTime._frozen = real_datetime(2026, 9, 12, 9, 0)
reminders._check_and_fire_reminders()
check("Deactivated reminder does not fire", len(delivered_calls) == 2)

# --- Case 6: no delivery callback set -> should not crash, just log ---
reminders._delivery_callback = None
rid2 = database.create_reminder("drink water", 14, 30)
FrozenDateTime._frozen = real_datetime(2026, 9, 12, 14, 30)
try:
    reminders._check_and_fire_reminders()
    check("Firing with no delivery callback set does not raise an exception", True)
except Exception as e:
    check("Firing with no delivery callback set does not raise an exception", False, str(e))

# Verify it was still logged even with no callback (so it can be caught up later)
undelivered = database.get_undelivered_reminders()
check("Reminder fired with no callback is still logged as undelivered for catch-up",
      len(undelivered) >= 1)

print()
print("=" * 60)
print(f"{sum(results)}/{len(results)} checks passed")
print("=" * 60)

shutil.rmtree(TEST_DIR, ignore_errors=True)
