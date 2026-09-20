# database.py — LUMA SQLite storage
# Handles conversation history, mood logs, caregiver info, habit logs,
# and long-term memory. Single-user (LUMA is a personal companion, not a
# multi-tenant product) — multi-user/login support is proposed as a
# future extension in the thesis rather than implemented here.

import sqlite3
import os
from datetime import datetime, timedelta
from collections import defaultdict

# Overridable via LUMA_DB_PATH so the container can point this at a
# mounted volume (see docker-compose.yml) for persistence across
# restarts (NFR-05). Falls back to the original same-directory behavior
# for local/non-container runs — nothing changes when you run this
# outside Docker.
DB_PATH = os.environ.get(
    "LUMA_DB_PATH",
    os.path.join(os.path.dirname(__file__), "luma.db")
)


def get_connection():
    """New connection per call — safe across async requests too."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't already exist. Safe to call on every app startup."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            role TEXT NOT NULL,            -- 'user' or 'luma'
            message TEXT NOT NULL,
            emotion TEXT                   -- positive/negative/ambiguous/neutral, NULL for luma rows
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mood_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE DEFAULT CURRENT_DATE,
            mood TEXT NOT NULL,
            note TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS caregiver_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            phone TEXT,
            relationship TEXT
        )
    """)

    # Habit tracker (used by the log_habit tool in tools.py)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS habit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            habit_type TEXT NOT NULL,
            quantity TEXT,
            notes TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Long-term memory: a single running summary of what LUMA knows about
    # the person, updated periodically from conversation history. Scope
    # is deliberately narrow (see memory.py) — name/age/job/schooling/
    # health readings/daily check-ins only, nothing emotional/interpretive.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS long_term_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            summary TEXT NOT NULL,
            last_summarized_message_id INTEGER DEFAULT 0,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Recurring reminders (e.g. "take BP medication", "exercise") — either
    # set up by the person via the schedule_reminder tool in conversation,
    # or added directly. hour/minute are 24hr local time; last_sent_date
    # prevents firing more than once per day.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL,
            hour INTEGER NOT NULL,
            minute INTEGER NOT NULL,
            active INTEGER DEFAULT 1,
            last_sent_date DATE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Log of fired reminders, so an undelivered one (person wasn't
    # connected when it fired) can be caught up on next connect instead
    # of silently disappearing.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reminder_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reminder_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            fired_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            delivered INTEGER DEFAULT 0,
            FOREIGN KEY (reminder_id) REFERENCES reminders(id)
        )
    """)

    conn.commit()
    conn.close()


# ---------- Conversations ----------

def log_message(role, message, emotion=None):
    """Save one conversation turn ('user' or 'luma')."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO conversations (role, message, emotion) VALUES (?, ?, ?)",
        (role, message, emotion)
    )
    conn.commit()
    conn.close()


def get_recent_conversations(limit=50):
    """Most recent conversation turns, returned oldest-first for chat display."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM conversations ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return list(reversed(rows))


# ---------- Mood logs ----------

def log_mood(mood, note=""):
    """Save a mood log entry for today."""
    conn = get_connection()
    conn.execute("INSERT INTO mood_logs (mood, note) VALUES (?, ?)", (mood, note))
    conn.commit()
    conn.close()


def get_mood_logs(days=7):
    """Mood logs from the last N days, oldest first (for the weekly chart)."""
    conn = get_connection()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = conn.execute(
        "SELECT * FROM mood_logs WHERE date >= ? ORDER BY date ASC", (cutoff,)
    ).fetchall()
    conn.close()
    return rows


# ---------- Insights ----------

def get_session_count(days=7):
    """Distinct days with at least one user message, in the last N days."""
    conn = get_connection()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    row = conn.execute(
        """SELECT COUNT(DISTINCT DATE(timestamp)) as cnt
           FROM conversations
           WHERE role = 'user' AND DATE(timestamp) >= ?""",
        (cutoff,)
    ).fetchone()
    conn.close()
    return row["cnt"] if row else 0


def get_emotion_distribution(days=7):
    """{emotion: percentage} across all user messages in the last N days."""
    conn = get_connection()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = conn.execute(
        """SELECT emotion, COUNT(*) as cnt
           FROM conversations
           WHERE role = 'user' AND emotion IS NOT NULL AND DATE(timestamp) >= ?
           GROUP BY emotion""",
        (cutoff,)
    ).fetchall()
    conn.close()

    total = sum(r["cnt"] for r in rows)
    if total == 0:
        return {}
    return {r["emotion"]: round(r["cnt"] / total * 100) for r in rows}


def get_dominant_emotion_days(days=7):
    """Count of days where each emotion was the most common that day.
    e.g. {'positive': 3, 'negative': 2, 'neutral': 1}"""
    conn = get_connection()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = conn.execute(
        """SELECT DATE(timestamp) as day, emotion, COUNT(*) as cnt
           FROM conversations
           WHERE role = 'user' AND emotion IS NOT NULL AND DATE(timestamp) >= ?
           GROUP BY day, emotion""",
        (cutoff,)
    ).fetchall()
    conn.close()

    day_emotions = defaultdict(dict)
    for r in rows:
        day_emotions[r["day"]][r["emotion"]] = r["cnt"]

    dominant_counts = defaultdict(int)
    for day, emotions in day_emotions.items():
        top_emotion = max(emotions, key=emotions.get)
        dominant_counts[top_emotion] += 1

    return dict(dominant_counts)


# ---------- Caregiver info ----------

def save_caregiver_info(name, phone, relationship):
    """Store caregiver contact info (single record — replaces any existing one)."""
    conn = get_connection()
    conn.execute("DELETE FROM caregiver_info")
    conn.execute(
        "INSERT INTO caregiver_info (name, phone, relationship) VALUES (?, ?, ?)",
        (name, phone, relationship)
    )
    conn.commit()
    conn.close()


def get_caregiver_info():
    """Saved caregiver info, or None if not set."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM caregiver_info LIMIT 1").fetchone()
    conn.close()
    return dict(row) if row else None


# ---------- Habit logs (used by the log_habit tool in tools.py) ----------

def log_habit(habit_type, quantity=None, notes=None):
    """Save a habit-tracking entry (e.g. smoking, drinking)."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO habit_logs (habit_type, quantity, notes) VALUES (?, ?, ?)",
        (habit_type, quantity, notes)
    )
    conn.commit()
    conn.close()
    return f"Logged: {habit_type}" + (f" ({quantity})" if quantity else "") + "."


def get_habit_logs(habit_type=None, limit=10):
    """Recent habit log entries, most recent first. Optionally filter by habit_type."""
    conn = get_connection()
    if habit_type:
        rows = conn.execute(
            """SELECT * FROM habit_logs WHERE habit_type = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (habit_type, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM habit_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return rows


# ---------- Long-term memory ----------

def get_long_term_summary():
    """Return (summary_text, last_summarized_message_id), or (None, 0) if none saved yet."""
    conn = get_connection()
    row = conn.execute(
        "SELECT summary, last_summarized_message_id FROM long_term_memory "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row:
        return row["summary"], row["last_summarized_message_id"]
    return None, 0


def save_long_term_summary(summary, last_summarized_message_id):
    """Save an updated long-term summary (single running row — replaces the previous one)."""
    conn = get_connection()
    conn.execute("DELETE FROM long_term_memory")
    conn.execute(
        "INSERT INTO long_term_memory (summary, last_summarized_message_id) VALUES (?, ?)",
        (summary, last_summarized_message_id)
    )
    conn.commit()
    conn.close()


def get_conversations_after(message_id, limit=200):
    """Conversation turns with id > message_id, oldest first — used to feed
    the summarizer only the messages it hasn't already seen."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM conversations WHERE id > ? ORDER BY id ASC LIMIT ?",
        (message_id, limit)
    ).fetchall()
    conn.close()
    return rows


def get_latest_message_id():
    """Highest conversation id currently in the DB (0 if empty)."""
    conn = get_connection()
    row = conn.execute("SELECT MAX(id) as max_id FROM conversations").fetchone()
    conn.close()
    return row["max_id"] or 0


# ---------- Reminders ----------

def create_reminder(label, hour, minute):
    """Create a new daily recurring reminder. Returns the new reminder's id."""
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO reminders (label, hour, minute) VALUES (?, ?, ?)",
        (label, hour, minute)
    )
    conn.commit()
    reminder_id = cursor.lastrowid
    conn.close()
    return reminder_id


def get_active_reminders():
    """All active reminders, for the scheduler to check each minute."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM reminders WHERE active = 1").fetchall()
    conn.close()
    return rows


def get_all_reminders():
    """All reminders (active and inactive), for display/management."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM reminders ORDER BY hour, minute").fetchall()
    conn.close()
    return rows


def mark_reminder_sent(reminder_id, date_str):
    """Record that a reminder fired today, so it doesn't fire again same day."""
    conn = get_connection()
    conn.execute("UPDATE reminders SET last_sent_date = ? WHERE id = ?", (date_str, reminder_id))
    conn.commit()
    conn.close()


def deactivate_reminder(reminder_id):
    conn = get_connection()
    conn.execute("UPDATE reminders SET active = 0 WHERE id = ?", (reminder_id,))
    conn.commit()
    conn.close()


def delete_reminder(reminder_id):
    conn = get_connection()
    conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
    conn.commit()
    conn.close()


# ---------- Reminder delivery log ----------

def log_reminder_fired(reminder_id, message):
    """Record that a reminder fired, initially undelivered. Returns log id."""
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO reminder_log (reminder_id, message, delivered) VALUES (?, ?, 0)",
        (reminder_id, message)
    )
    conn.commit()
    log_id = cursor.lastrowid
    conn.close()
    return log_id


def mark_reminder_delivered(log_id):
    conn = get_connection()
    conn.execute("UPDATE reminder_log SET delivered = 1 WHERE id = ?", (log_id,))
    conn.commit()
    conn.close()


def get_undelivered_reminders():
    """Reminders that fired but were never delivered (person wasn't
    connected at the time) — used to catch up on next connect."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM reminder_log WHERE delivered = 0 ORDER BY fired_at ASC"
    ).fetchall()
    conn.close()
    return rows


# Test
if __name__ == "__main__":
    init_db()
    print("Database initialized at:", DB_PATH)

    log_message("user", "I am doing okay", emotion="positive")
    log_message("luma", "That's great to hear!")
    log_mood("😊", "Feeling good today")
    log_habit("smoking", "2 cigarettes", "stressful afternoon")

    print("Recent conversations:", [dict(r) for r in get_recent_conversations(5)])
    print("Mood logs:", [dict(r) for r in get_mood_logs(7)])
    print("Session count:", get_session_count())
    print("Emotion distribution:", get_emotion_distribution())
    print("Habit logs:", [dict(r) for r in get_habit_logs()])
