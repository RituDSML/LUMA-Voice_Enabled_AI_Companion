# tools.py — LUMA's callable tools for agentic (function-calling) use
#
# These are real tools the LLM can invoke via Groq's native function-calling
# API — the model decides *when* to call them based on the conversation,
# rather than us keyword-matching and stuffing instructions into the prompt.
#
# Three tools in this pass:
#   1. get_current_time   — simple, no external dependency
#   2. get_weather         — free, no-API-key weather via Open-Meteo
#   3. log_habit           — writes a habit entry (smoking/drinking/etc.)
#                             to LUMA's existing SQLite DB (database.py)
#
# NOTE ON CRISIS DETECTION: check_crisis is deliberately NOT included here.
# Your safety.py is designed to run as a mandatory pre-check BEFORE the LLM
# is ever called ("predictable, not dependent on model behavior under
# pressure" — per its own comment). Making it an LLM-callable tool would
# turn a safety-critical check into something the model could simply choose
# not to call. Keep safety.check_crisis() wired into your message-handling
# flow (app.py) exactly as it already is — outside the agent's discretion.

import requests
from datetime import datetime
from database import (
    log_habit as _db_log_habit,
    get_habit_logs as _db_get_habit_logs,
    create_reminder as _db_create_reminder,
    get_all_reminders as _db_get_all_reminders,
)

# ---------------------------------------------------------------------------
# 1. TIME
# ---------------------------------------------------------------------------

def get_current_time():
    """Return the current date and time as a readable string."""
    now = datetime.now()
    return now.strftime("%A, %d %B %Y, %I:%M %p")


# ---------------------------------------------------------------------------
# 2. WEATHER (Open-Meteo — free, no API key required)
# ---------------------------------------------------------------------------

def get_weather(location: str):
    """
    Look up current weather for a place name using Open-Meteo (free, keyless).
    Two-step: geocode the place name to lat/lon, then fetch current weather.
    """
    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1},
            timeout=5
        )
        geo_data = geo_resp.json()
        results = geo_data.get("results")
        if not results:
            return f"Sorry, I couldn't find a location called '{location}'."

        place = results[0]
        lat, lon = place["latitude"], place["longitude"]
        resolved_name = f"{place.get('name')}, {place.get('country', '')}".strip(", ")

        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": "true"},
            timeout=5
        )
        weather_data = weather_resp.json().get("current_weather")
        if not weather_data:
            return f"Sorry, I couldn't fetch weather for {resolved_name} right now."

        temp = weather_data["temperature"]
        windspeed = weather_data["windspeed"]
        return (f"It's currently {temp}°C in {resolved_name}, "
                f"with wind speeds around {windspeed} km/h.")

    except requests.RequestException:
        return "Sorry, I couldn't reach the weather service right now."


# ---------------------------------------------------------------------------
# 3. HABIT TRACKER — now backed by database.py's shared luma.db connection,
#    not a separate file, so all of LUMA's data lives in one place.
# ---------------------------------------------------------------------------

def log_habit(habit_type: str, quantity: str = None, notes: str = None):
    """
    Log a habit-tracking entry (e.g. smoking, drinking) with an optional
    quantity (e.g. '3 cigarettes') and optional notes (e.g. a trigger).
    """
    return _db_log_habit(habit_type, quantity, notes)


def get_habit_summary(habit_type: str = None, limit: int = 10):
    """Retrieve recent habit log entries, optionally filtered by habit_type."""
    rows = _db_get_habit_logs(habit_type, limit)
    if not rows:
        return "No habit entries logged yet."

    lines = []
    for r in rows:
        line = f"{r['habit_type']} — {r['quantity'] or 'n/a'} ({str(r['timestamp'])[:16]})"
        if r["notes"]:
            line += f" — {r['notes']}"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. REMINDERS — daily recurring reminders (e.g. medication, exercise),
#    delivered proactively by the background scheduler in reminders.py.
# ---------------------------------------------------------------------------

def schedule_reminder(label: str, hour: int, minute: int = 0):
    """
    Set up a daily recurring reminder (e.g. 'take BP medication' at 9:00).
    hour is 24-hour format (0-23).
    """
    if not (0 <= hour <= 23) or not (0 <= minute <= 59):
        return "Invalid time — hour must be 0-23 and minute 0-59."
    _db_create_reminder(label, hour, minute)
    time_str = f"{hour:02d}:{minute:02d}"
    return f"Reminder set: \"{label}\" every day at {time_str}."


def list_reminders():
    """List all currently set reminders."""
    rows = _db_get_all_reminders()
    if not rows:
        return "No reminders set yet."
    lines = []
    for r in rows:
        status = "active" if r["active"] else "inactive"
        lines.append(f"{r['hour']:02d}:{r['minute']:02d} — {r['label']} ({status})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TOOL SCHEMAS — Groq/OpenAI-compatible function-calling format
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given place name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or place name, e.g. 'Wayanad' or 'Kochi'"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {  
            "name": "log_habit",
            "description": ("Log a habit-tracking entry when the person mentions something like "
                            "smoking or drinking, so it can be tracked over time. Only call this "
                            "when the person has clearly stated they did the habit (not when just "
                            "discussing it hypothetically)."),
            "parameters": {
                "type": "object",
                "properties": {
                    "habit_type": {
                        "type": "string",
                        "description": "The habit being logged, e.g. 'smoking', 'drinking'"
                    },
                    "quantity": {
                        "type": "string",
                        "description": "Amount if mentioned, e.g. '3 cigarettes', '2 drinks'"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Any trigger or context mentioned, e.g. 'stressful day at work'"
                    }
                },
                "required": ["habit_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_reminder",
            "description": ("Set up a daily recurring reminder for the person, e.g. taking "
                            "medication, exercising, or a routine check-in. Only call this "
                            "when the person clearly asks to be reminded about something at "
                            "a specific time."),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "What to remind them about, e.g. 'take BP medication'"
                    },
                    "hour": {
                        "type": "integer",
                        "description": "Hour of day, 24-hour format (0-23)"
                    },
                    "minute": {
                        "type": "integer",
                        "description": "Minute of the hour (0-59), default 0"
                    }
                },
                "required": ["label", "hour"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "List all reminders currently set, when the person asks what reminders they have.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]

# Maps tool name -> actual Python function, used by llm.py to execute
# whatever the model decides to call.
TOOL_FUNCTIONS = {
    "get_current_time": lambda **kwargs: get_current_time(),
    "get_weather": lambda **kwargs: get_weather(**kwargs),
    "log_habit": lambda **kwargs: log_habit(**kwargs),
    "schedule_reminder": lambda **kwargs: schedule_reminder(**kwargs),
    "list_reminders": lambda **kwargs: list_reminders(),
}
