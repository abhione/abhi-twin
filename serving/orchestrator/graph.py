"""LangGraph spine: INTAKE -> ROUTER -> RETRIEVER -> PERSONA -> TOOLS -> VOICE
(+ video sink). RUN ON SPARK inside the orchestrator container (spec §11,
forked from the Hermes Agent playbook shape)."""

from __future__ import annotations

import os
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

from serving.orchestrator.router import classify
from serving.rag.search import retrieve

SPARK = os.environ.get("TWIN_SPARK_HOST", "localhost")
LLM_URL = os.environ.get("TWIN_LLM_URL", f"http://{SPARK}:8000/v1")
TTS_URL = os.environ.get("TWIN_TTS_URL", f"http://{SPARK}:8001/v1")
FRONTIER_URL = os.environ.get("TWIN_FRONTIER_URL", "")  # optional cloud escalation
FRONTIER_KEY = os.environ.get("TWIN_FRONTIER_KEY", "")

PERSONA_SYSTEM = (
    "You are AbhiTwin — reply exactly as Abhi Bhattacharya writes: direct, warm, "
    "technically precise, no corporate filler. Use the retrieved context when relevant."
)


class TwinState(TypedDict, total=False):
    user_text: str
    session: str
    route: str
    local_only: bool
    context_chunks: list[str]
    reply_text: str
    tool_results: list[dict]
    audio: bytes | None
    want_voice: bool
    want_video: bool


def intake(state: TwinState) -> TwinState:
    state["user_text"] = state["user_text"].strip()
    return state


def route_node(state: TwinState) -> TwinState:
    decision = classify(state["user_text"])
    state["route"] = decision.route
    state["local_only"] = decision.local_only
    return state


def retriever(state: TwinState) -> TwinState:
    state["context_chunks"] = retrieve(state["user_text"], top_k=8)
    return state


def persona(state: TwinState) -> TwinState:
    if state["route"] == "research" and FRONTIER_URL:
        client = OpenAI(base_url=FRONTIER_URL, api_key=FRONTIER_KEY)
        model = os.environ.get("TWIN_FRONTIER_MODEL", "claude-fable-5")
    else:
        client = OpenAI(base_url=LLM_URL, api_key="local")
        model = os.environ.get("TWIN_LLM_MODEL", "persona-v1")
    context = "\n\n".join(state.get("context_chunks", []))
    system = PERSONA_SYSTEM + (f"\n\n<context>\n{context}\n</context>" if context else "")
    extra: dict[str, Any] = {}
    if state.get("want_voice"):
        # voice roundtrip latency is decode + TTS, both linear in reply length —
        # spoken replies must stay conversational (spec §12 <3 s e2e gate)
        system += "\n\nThis reply will be spoken aloud: answer in one or two short sentences."
        extra["max_tokens"] = 60
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": state["user_text"]},
        ],
        **extra,
    )
    state["reply_text"] = resp.choices[0].message.content
    return state


def tools(state: TwinState) -> TwinState:
    # v1 is advisory-only (spec open question #1): connectors draft, never send.
    state["tool_results"] = []
    return state


def voice(state: TwinState) -> TwinState:
    if not state.get("want_voice"):
        state["audio"] = None
        return state
    import httpx

    resp = httpx.post(
        f"{TTS_URL}/audio/speech",
        json={"model": "voice-v1", "input": state["reply_text"], "voice": "abhi"},
        timeout=120,
    )
    resp.raise_for_status()
    state["audio"] = resp.content
    # video sink: when a WebRTC video session is open, the orchestrator forwards
    # this audio to the MuseTalk streamer (serving/video/server.py) over WS.
    return state


def build_graph() -> Any:
    g = StateGraph(TwinState)
    g.add_node("INTAKE", intake)
    g.add_node("ROUTER", route_node)
    g.add_node("RETRIEVER", retriever)
    g.add_node("PERSONA", persona)
    g.add_node("TOOLS", tools)
    g.add_node("VOICE", voice)
    g.set_entry_point("INTAKE")
    g.add_edge("INTAKE", "ROUTER")
    g.add_edge("ROUTER", "RETRIEVER")
    g.add_edge("RETRIEVER", "PERSONA")
    g.add_conditional_edges(
        "PERSONA",
        lambda s: "TOOLS" if s["route"] == "tool" else "VOICE",
        {"TOOLS": "TOOLS", "VOICE": "VOICE"},
    )
    g.add_edge("TOOLS", "VOICE")
    g.add_edge("VOICE", END)
    return g.compile()
