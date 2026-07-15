"""Format records as {"messages": [...]} chat pairs (spec §6 step 5). Host: mac | spark.

Records with a real counterpart message keep it as the user turn. Promptless
records (notes, PR bodies, sent-only email) get a synthetic prompt reconstructed
by Qwen2.5-72B: "given this reply, what was likely asked?"
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable

from corpus.pipeline.records import Record

RECONSTRUCT_INSTRUCTION = (
    "You will be shown a message someone wrote. Reconstruct the most likely prompt, "
    "question, or situation it was written in response to. Reply with the "
    "reconstructed prompt only — one to three sentences, no preamble."
)


class EndpointReconstructor:
    """Synthetic-prompt reconstruction against an OpenAI-compatible /v1 endpoint
    (the Spark's vLLM at TWIN_SPARK_HOST:8000)."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "qwen2.5-72b",
        timeout: float = 120.0,
    ) -> None:
        host = os.environ.get("TWIN_SPARK_HOST", "spark")
        self.base_url = (base_url or f"http://{host}:8000/v1").rstrip("/")
        self.model = model
        self.timeout = timeout

    def __call__(self, reply: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": RECONSTRUCT_INSTRUCTION},
                {"role": "user", "content": reply},
            ],
            "temperature": 0.7,
            "max_tokens": 200,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.load(resp)
        return data["choices"][0]["message"]["content"].strip()


def to_messages(
    rec: Record,
    reconstructor: Callable[[str], str] | None = None,
    system: str | None = None,
) -> dict | None:
    """Return the chat-format sample, or None if promptless and no reconstructor."""
    prompt = rec.prompt
    if prompt is None:
        if reconstructor is None:
            return None
        prompt = reconstructor(rec.reply)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": rec.reply})
    return {"messages": messages, "id": rec.id, "source": rec.source}
