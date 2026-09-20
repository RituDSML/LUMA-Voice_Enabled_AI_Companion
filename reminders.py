# reminders.py — LUMA's proactive reminder scheduler
#
# Runs a background job (APScheduler) that checks every minute for
# reminders due to fire. When one fires, it calls back into main.py to
# actually deliver it — either live (if the browser is connected) or
# logged as undelivered, to be caught up on next connect.
#
# Requires: pip install apscheduler

from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

from database import (
    get_active_reminders,
    mark_reminder_sent,
    log_reminder_fired,
)

_scheduler = None
_delivery_callback = None  # set via set_delivery_callback() from main.py


def set_delivery_callback(callback):
    """
    Register the function that actually delivers a fired reminder.
    main.py provides this — it knows whether a WebSocket is currently
    connected and can push the message live, or otherwise leave it
    logged as undelivered for catch-up on next connect.

    callback signature: callback(reminder_id: int, message: str, log_id: int)
    """
    global _delivery_callback
    _delivery_callback = callback


def _check_and_fire_reminders():
    """Runs every minute: checks if any active reminder matches the
    current hour/minute and hasn't already fired today."""
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    for reminder in get_active_reminders():
        if reminder["hour"] == now.hour and reminder["minute"] == now.minute:
            if reminder["last_sent_date"] == today_str:
                continue  # already fired today, skip

            message = f"Just a gentle reminder: {reminder['label']}"
            log_id = log_reminder_fired(reminder["id"], message)
            mark_reminder_sent(reminder["id"], today_str)

            if _delivery_callback:
                _delivery_callback(reminder["id"], message, log_id)
            else:
                print(f"[reminders] Fired but no delivery callback set: {message}")


def start_scheduler():
    """Start the background scheduler. Call once, on app startup."""
    global _scheduler
    if _scheduler is not None:
        return  # already running
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(_check_and_fire_reminders, "interval", minutes=1, id="reminder_check")
    _scheduler.start()
    print("[reminders] Scheduler started — checking every minute.")


def stop_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown()
        _scheduler = None
