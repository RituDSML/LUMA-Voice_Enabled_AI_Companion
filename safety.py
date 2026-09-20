# safety.py — LUMA Tier 3 Crisis Detection
# Runs as an independent check BEFORE emotion detection / LLM call.
# Deliberately keyword-based, not LLM-based — predictable, not dependent
# on model behavior under pressure.

CRISIS_KEYWORDS = [
    "kill myself", "killing myself",
    "end it all", "ending it all", "ended it all",
    "end my life", "ending my life", "ended my life",
    "want to die", "wanted to die", "wanting to die",
    "don't want to live", "do not want to live",
    "don't want to be here", "do not want to be here",
    "don't want to be alive", "do not want to be alive",
    "no reason to live", "no reason to live anymore",
    "better off without me", "better off without",
    "can't go on", "cannot go on", "can't go on anymore",
    "suicide", "suicidal", "self harm", "self-harm",
    "hurt myself", "hurting myself",
    "harm myself", "harming myself",
    "cutting myself", "overdose",
    "no point in living", "no point living",
    "tired of living", "give up on life", "giving up on life", "ready to give up",
]

CRISIS_RESPONSE = (
    "I'm really concerned about what you just shared, and I want you to know "
    "you don't have to go through this alone. Please reach out to one of these "
    "right now — trained counselors are available:\n\n"
    "Tele MANAS (Govt of India, 24x7): 14416 or 1-800-891-4416\n"
    "KIRAN Mental Health Helpline (Govt of India, 24x7): 1800-599-0019\n"
    "Vandrevala Foundation (24x7, also on WhatsApp): 1860-266-2345 / 9999 666 555\n\n"
    "If you're in immediate danger, please call 112 or go to your nearest "
    "emergency room. I'm still here — would you like to keep talking?"
)

def check_crisis(text):
    """Returns True if text contains crisis-level language."""
    if not text:
        return False
    lower = text.lower()
    return any(kw in lower for kw in CRISIS_KEYWORDS)