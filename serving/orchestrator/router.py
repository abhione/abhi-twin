"""Intent router policy (spec §11, mirrors the Reachy Mini pattern). Host: any —
pure logic, no ML deps, unit-tested locally. The LangGraph ROUTER node calls this
first; an optional LLM classifier can refine `casual` only. Routes:

  sensitive  contains email/iMessage/finance/health material -> LOCAL persona LLM ONLY
  tool       actionable ask (send/schedule/create) -> local LLM + connector calls
  research   deep research -> escalate to the frontier stack via API
  casual     everything else -> local persona LLM
"""

from __future__ import annotations

import dataclasses
import re

# privacy first: anything smelling of private comms or money stays local, always
_SENSITIVE = re.compile(
    r"\b(email|inbox|gmail|imessage|text message|dm|slack message"
    r"|bank|finance|financial|salary|comp(?:ensation)?|invoice|tax(?:es)?"
    r"|ssn|passport|password|credentials?"
    r"|medical|health record|diagnosis|prescription)\b",
    re.IGNORECASE,
)
_RESEARCH = re.compile(
    r"\b(deep[- ]?research|research|investigate|literature|survey the|state of the art"
    r"|compare (?:papers|approaches|frameworks)|comprehensive (?:report|analysis))\b",
    re.IGNORECASE,
)
_TOOL = re.compile(
    r"\b(send|schedule|book|remind me|create (?:a )?(?:pr|issue|ticket|event)"
    r"|add to (?:my )?calendar|set up a meeting|draft and send)\b",
    re.IGNORECASE,
)


@dataclasses.dataclass
class RouteDecision:
    route: str  # sensitive | tool | research | casual
    reason: str
    local_only: bool


def classify(text: str) -> RouteDecision:
    if m := _SENSITIVE.search(text):
        return RouteDecision("sensitive", f"matched {m.group(0)!r}", local_only=True)
    if m := _TOOL.search(text):
        return RouteDecision("tool", f"matched {m.group(0)!r}", local_only=True)
    if m := _RESEARCH.search(text):
        # research may leave the box — but never with sensitive content (checked above)
        return RouteDecision("research", f"matched {m.group(0)!r}", local_only=False)
    return RouteDecision("casual", "default", local_only=True)
