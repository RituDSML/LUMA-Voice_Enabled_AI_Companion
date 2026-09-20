# graph.py — LUMA's LangGraph state graph
#
# Wraps the existing crisis / emotion / response / memory functions as
# LangGraph nodes, replacing the hand-written sequential logic previously
# in main.py's handle_user_text().

from typing import TypedDict, Optional, List, Dict
from langgraph.graph import StateGraph, END

from emotion_model import get_emotion_label
from llm import get_response
from safety import check_crisis, CRISIS_RESPONSE
from memory import maybe_update_summary


class LumaState(TypedDict):
    user_input: str
    history: List[Dict]
    long_term_summary: Optional[str]
    crisis_tier: str          # "none" | "elevated"
    emotion: Optional[str]
    response: Optional[str]


# --- Nodes ---------------------------------------------------------------

def crisis_detection_node(state: LumaState) -> LumaState:
    state["crisis_tier"] = "elevated" if check_crisis(state["user_input"]) else "none"
    return state


def crisis_response_node(state: LumaState) -> LumaState:
    state["response"] = CRISIS_RESPONSE
    state["emotion"] = None
    return state


def emotion_detection_node(state: LumaState) -> LumaState:
    state["emotion"] = get_emotion_label(state["user_input"])
    return state


def response_generation_node(state: LumaState) -> LumaState:
    state["response"] = get_response(
        state["user_input"],
        emotion=state["emotion"],
        history=state["history"],
        long_term_summary=state["long_term_summary"],
    )
    return state


def memory_update_node(state: LumaState) -> LumaState:
    maybe_update_summary()
    return state


# --- Conditional routing ---------------------------------------------------

def route_after_crisis_check(state: LumaState) -> str:
    return "crisis_response" if state["crisis_tier"] == "elevated" else "emotion_detection"


# --- Build + compile -------------------------------------------------------

def build_graph():
    graph = StateGraph(LumaState)

    graph.add_node("crisis_detection", crisis_detection_node)
    graph.add_node("crisis_response", crisis_response_node)
    graph.add_node("emotion_detection", emotion_detection_node)
    graph.add_node("response_generation", response_generation_node)
    graph.add_node("memory_update", memory_update_node)

    graph.set_entry_point("crisis_detection")
    graph.add_conditional_edges(
        "crisis_detection",
        route_after_crisis_check,
        {"crisis_response": "crisis_response", "emotion_detection": "emotion_detection"},
    )
    graph.add_edge("crisis_response", END)
    graph.add_edge("emotion_detection", "response_generation")
    graph.add_edge("response_generation", "memory_update")
    graph.add_edge("memory_update", END)

    return graph.compile()


luma_graph = build_graph()