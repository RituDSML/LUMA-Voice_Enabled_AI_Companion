# llm.py - LUMA's Language Model Connection
# Connects to Groq API and generates emotion-aware empathetic responses.
# Now agentic: the model can call real tools (time, weather, habit logging)
# via Groq's native function-calling, instead of us keyword-matching and
# stuffing behavior instructions into the prompt. Crisis detection stays
# in safety.py as a mandatory pre-check outside the model's discretion —
# see the note in tools.py for why.
#
# RAG: two gated knowledge bases (coping strategies, first aid) retrieved
# ONLY when detect_concern_tier() flags 'mental_health' or 'physical_health'
# respectively — this avoids leaking retrieved content into unrelated
# conversation (e.g. "what time is it" never touches either KB).

import json
import re
from groq import Groq
from dotenv import load_dotenv
import os

from tools import TOOLS, TOOL_FUNCTIONS
from luma_rag import LumaRAG, COPING_KB_PATH, FIRST_AID_KB_PATH

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Loaded once at import time (not per-message) — building the FAISS index
# on every message would be wasteful and slow.
coping_rag = LumaRAG(COPING_KB_PATH)
first_aid_rag = LumaRAG(FIRST_AID_KB_PATH)

# Tier 1: general physical health concerns
HEALTH_KEYWORDS = [
    'blood pressure', 'bp ', 'sick', 'unwell', 'fever', 'pain',
    'symptom', 'medication', 'not feeling well', 'doctor'
]

# Tier 2: mental health language
MENTAL_HEALTH_KEYWORDS = [
    'depress', 'hopeless', 'anxious', 'anxiety', "can't cope", "cannot cope",
    'overwhelmed', 'panic', 'worthless', 'lonely', 'isolated', 'burnt out', 'burnout'
]

# Habit-tracking tier: still used to shape TONE/nudging in the prompt.
# Actual logging now happens via the log_habit TOOL (model-invoked), not
# just prompt text — this keyword list just helps set conversational tone.
HABIT_KEYWORDS = [
    'smoke', 'smoking', 'cigarette', 'cigarettes', 'vape', 'vaping',
    'alcohol', 'drinking', 'drink alcohol'
]

MAX_TOOL_ROUNDS = 3  # safety cap so a tool-call loop can't run forever

# Safety net: some models occasionally output a hallucinated tool-call as
# plain text (e.g. "<function=log_habit={...}></function>") instead of
# using the real structured tool_calls mechanism. If that leaks into the
# final reply, this strips it out before it ever reaches the person.
_LEAKED_FUNCTION_CALL_PATTERN = re.compile(r"<function=.*?(</function>|$)", re.DOTALL)


def _strip_leaked_function_syntax(text):
    if not text:
        return text
    cleaned = _LEAKED_FUNCTION_CALL_PATTERN.sub("", text).strip()
    return cleaned if cleaned else text


def detect_concern_tier(text):
    """Lightweight keyword check for health/mental-health/habit nudges (not crisis detection)"""
    lower = text.lower()
    if any(kw in lower for kw in MENTAL_HEALTH_KEYWORDS):
        return 'mental_health'
    if any(kw in lower for kw in HABIT_KEYWORDS):
        return 'habit_tracking'
    if any(kw in lower for kw in HEALTH_KEYWORDS):
        return 'physical_health'
    return None


def _build_retrieval_block(user_message, concern):
    """
    Gated RAG retrieval — only fires for 'mental_health' (coping strategies)
    or 'physical_health' (first aid) concern tiers. Returns "" for every
    other case, so unrelated conversation never touches either knowledge
    base (no leakage).
    """
    if concern == 'mental_health':
        results = coping_rag.retrieve(user_message, k=2)
        if not results:
            return ""
        context = "\n\n".join(r["content"] for r in results)
        return (f"Relevant coping strategies you may draw from if genuinely helpful:\n"
                f"{context}\n"
                f"Only bring these up if they naturally fit what the person needs right "
                f"now — don't force a technique into the conversation. Explain it briefly "
                f"in your own words, like a knowledgeable friend, not a pamphlet.")

    elif concern == 'physical_health':
        results = first_aid_rag.retrieve(user_message, k=2)
        if not results:
            return ""
        context = "\n\n".join(r["content"] for r in results)
        return (f"Standard first-aid steps that may be relevant:\n"
                f"{context}\n"
                f"Only share this for minor, common situations. If anything suggests a "
                f"serious or urgent situation (severe bleeding, chest pain, difficulty "
                f"breathing, unconsciousness, choking that isn't resolving), don't try to "
                f"talk them through it — clearly and immediately tell them to call "
                f"emergency services (112 or 108 in India). Never suggest medications, "
                f"dosages, or diagnose what's wrong.")

    return ""


def _execute_tool_call(tool_call):
    """Run the actual Python function behind a model-requested tool call."""
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}

    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return f"Error: unknown tool '{name}'"

    try:
        return func(**args)
    except Exception as e:
        return f"Error running tool '{name}': {e}"


def get_response(user_message, emotion=None, history=None, long_term_summary=None):
    """Send message to Groq and get emotion-aware response.

    history: optional list of {'role': 'user'|'luma', 'content': str} dicts,
    most-recent-last, representing the last few turns of conversation.

    long_term_summary: optional string — a compact summary of what LUMA
    knows about this person from past sessions (see memory.py). Lets LUMA
    remember things across conversations, not just the current one.

    Now supports tool-calling: if the model requests a tool (time, weather,
    habit logging, crisis check), we run it, feed the result back, and let
    the model produce its final natural-language reply.

    Also supports gated RAG retrieval (coping strategies / first aid) — see
    _build_retrieval_block for the gating logic.
    """

    # Tone mapping aligned to actual 4-group model output
    if emotion == 'positive':
        tone = "Be warm, positive and forward-looking. Celebrate with them."
    elif emotion == 'negative':
        tone = "Be gentle, empathetic and supportive. Validate the feeling before offering anything else."
    elif emotion == 'ambiguous':
        tone = "Be curious and gently exploratory — invite them to share more before assuming how they feel."
    else:  # neutral
        tone = "Be warm and supportive."

    # Engagement style — only push hard into curiosity/suggestions for non-negative tone
    if emotion == 'negative':
        engagement = ("Once you've validated how they feel, you can gently ask one specific "
                      "follow-up question — but don't rush to suggestions or advice.")
    else:
        engagement = ("Be a genuinely curious, present conversational partner, not just a "
                      "reflective listener. Ask specific follow-up questions about details they "
                      "mention, rather than generic ones like 'would you like to talk about it?'. "
                      "Feel free to share a light opinion, suggestion, or recommendation when it "
                      "fits naturally (an activity idea, a book or music suggestion) — the way an "
                      "engaged friend would. Avoid repeating stock phrases like 'I'm here to listen' "
                      "or 'what's on your mind' in every response.")

    # When LUMA makes a concrete recommendation (book, show, activity, etc.),
    # give it somewhere real to go, then close the loop instead of lingering —
    # a good friend doesn't keep the conversation open just to keep it open.
    suggestion_closure = ("If you suggest something concrete like a book, show, or resource: "
                          "point to legitimate ways to access it — a public library app "
                          "(e.g. Libby/OverDrive), a public-domain source for older or classic "
                          "works (e.g. Project Gutenberg, LibriVox, Standard Ebooks), an official "
                          "free trial (e.g. Audible), or a normal retailer/bookstore. Never mention "
                          "or imply free PDF downloads, file-sharing sites, or any other unofficial "
                          "source for in-copyright work. After giving that, close warmly and let "
                          "the conversation rest — something like inviting them to let you know how "
                          "it goes, and that you'll be here whenever they want to check back in. "
                          "Don't keep prompting with more questions once you've made a suggestion "
                          "and pointed them to it.")

    # Tier 1 / Tier 2 / habit-tracking nudges
    concern = detect_concern_tier(user_message)
    if concern == 'mental_health':
        nudge = ("The person may be describing ongoing mental health difficulty. "
                 "Validate the feeling first, without judgment. If it sounds persistent "
                 "rather than a passing moment, gently suggest that talking to a counselor "
                 "or therapist could help — frame it as an option, not an instruction.")
    elif concern == 'habit_tracking':
        nudge = ("The person mentioned a habit worth gently tracking (e.g. smoking, "
                 "drinking). As a health-monitoring companion, show curiosity about the "
                 "specifics — frequency, amount, or triggers (e.g. 'how many a day', "
                 "'what usually brings it on') — the way a caring check-in would, not "
                 "an interrogation. Ask at most one such question per reply. Never lecture, "
                 "moralize, or push them to quit; stay supportive and non-judgmental, and "
                 "only mention professional support if they express wanting to change. "
                 "If they've clearly told you they did the habit (not just discussing it), "
                 "log it using the log_habit tool.")
    elif concern == 'physical_health':
        nudge = ("The person may be raising a physical health concern. Show you're listening, "
                 "ask one gentle follow-up if natural, and if it sounds ongoing or unaddressed, "
                 "gently suggest checking in with a doctor.")
    else:
        nudge = ""

    # NEW: gated RAG retrieval block (empty string if concern tier doesn't match)
    retrieval_block = _build_retrieval_block(user_message, concern)

    tool_guidance = ("You have access to tools: get_current_time, get_weather, log_habit, "
                     "schedule_reminder, and list_reminders. Call them only when genuinely "
                     "relevant to what the person said — don't force tool use into unrelated "
                     "conversation. IMPORTANT for log_habit: only call it when the person has "
                     "just stated, as fact, that they did the habit (e.g. 'I smoked earlier'). "
                     "Never call it because YOU are asking them a question about a habit — "
                     "asking is not the same as them confirming it. For schedule_reminder: only "
                     "call it when they clearly ask to be reminded about something at a specific "
                     "time.")

    if long_term_summary:
        memory_block = (f"What you remember about this person from past conversations:\n"
                        f"{long_term_summary}\n"
                        f"Use this naturally where relevant — don't recite it or announce that "
                        f"you're recalling something, just let it inform how you respond, the "
                        f"way a friend would remember things without saying 'I recall that...'.")
    else:
        memory_block = ""

    system_prompt = f"""You are LUMA, a warm and empathetic AI mental health
companion. You support people through compassionate conversation.
You never replace professional care. You never diagnose or give medical advice.
Stay fully present as a companion in every reply — never break the moment to say
things like "I'm not human," "I don't have feelings like humans do," "I'm just an
AI," or similar disclaimers about your own nature. The person already knows what
you are; restating it adds nothing and undercuts the support you're giving. If
asked directly what you are, answer honestly and briefly, then move on — but
never volunteer it unprompted, and never use it to soften or hedge a caring
response.
{tone}
{engagement}
{suggestion_closure}
{nudge}
{retrieval_block}
{tool_guidance}
{memory_block}
Keep responses concise and natural (2-4 sentences). Never assume the person is
struggling if they say they are fine or okay."""

    messages = [{"role": "system", "content": system_prompt}]

    # Include recent turns so the model has short-term memory of the
    # conversation. Groq's API (like OpenAI's) expects 'user'/'assistant',
    # so map our stored 'luma' role to 'assistant' here.
    if history:
        for turn in history:
            role = "assistant" if turn["role"] == "luma" else "user"
            messages.append({"role": role, "content": turn["content"]})

    messages.append({"role": "user", "content": user_message})

    # --- Tool-calling loop ---
    # The model may respond with tool_calls instead of (or before) a final
    # text answer. We execute whatever it asks for, feed results back as
    # 'tool' messages, and re-call until we get a plain text response or
    # hit MAX_TOOL_ROUNDS (safety cap against runaway loops).
    #
    # NOTE: llama-3.3-70b-versatile is deprecated by Groq, shutting down
    # 08/16/2026. Switched to openai/gpt-oss-120b, Groq's recommended
    # replacement for reasoning/tool-use workloads — also more reliable
    # at actually using the structured tool_calls mechanism instead of
    # leaking pseudo-function-call text into the reply.
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return _strip_leaked_function_syntax(msg.content)

        # Record the assistant's tool-call request, then run each tool
        # and append its result so the model can use it on the next turn.
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                } for tc in msg.tool_calls
            ]
        })

        for tc in msg.tool_calls:
            result = _execute_tool_call(tc)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result)
            })

    # If we hit MAX_TOOL_ROUNDS without a final text answer, ask once more
    # without tools available, forcing a plain response instead of looping.
    fallback = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages
    )
    return fallback.choices[0].message.content


# Test
if __name__ == "__main__":
    tests = [
        ("What time is it right now?", "neutral"),
        ("What's the weather like in Wayanad today?", "neutral"),
        ("I smoked 2 cigarettes after lunch, stress got to me", "neutral"),
        ("I feel so hopeless today", "negative"),
        ("Can you recommend a good book to read this week?", "neutral"),
        ("I cut my finger and it's bleeding a lot", "neutral"),
    ]
    for msg, emo in tests:
        print(f"You: {msg}")
        print("LUMA:", get_response(msg, emotion=emo))
        print("-" * 40)
