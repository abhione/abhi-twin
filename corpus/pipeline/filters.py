"""Length + perplexity quality filters (spec §6 steps 3-4). Host: mac | spark.

The perplexity scorer needs a GPU (Qwen2.5-7B); on the Mac pass --skip-ppl to
build.py, or inject any callable scorer (unit tests use a fake).
"""

from __future__ import annotations

import re
from typing import Callable, Iterable

from corpus.pipeline.records import Record

_WORD = re.compile(r"\S+")


def approx_token_count(text: str, tokenizer: Callable[[str], list] | None = None) -> int:
    """Real tokenizer when provided (Spark); otherwise a conservative estimate:
    max(word count, chars/4) — close enough for a 30..4096 band."""
    if tokenizer is not None:
        return len(tokenizer(text))
    return max(len(_WORD.findall(text)), len(text) // 4)


def length_filter(
    records: Iterable[Record],
    min_tokens: int = 30,
    max_tokens: int = 4096,
    tokenizer: Callable[[str], list] | None = None,
) -> tuple[list[Record], list[Record]]:
    kept, dropped = [], []
    for rec in records:
        n = approx_token_count(rec.reply, tokenizer)
        (kept if min_tokens <= n <= max_tokens else dropped).append(rec)
    return kept, dropped


def ppl_filter(
    records: list[Record],
    scorer: Callable[[str], float],
    drop_frac: float = 0.05,
) -> tuple[list[Record], list[Record]]:
    """Score reply perplexity with a base model; drop the top `drop_frac` weirdest."""
    if not records:
        return [], []
    scored = sorted(((scorer(r.reply), i) for i, r in enumerate(records)), reverse=True)
    n_drop = int(len(records) * drop_frac)
    drop_idx = {i for _, i in scored[:n_drop]}
    kept = [r for i, r in enumerate(records) if i not in drop_idx]
    dropped = [r for i, r in enumerate(records) if i in drop_idx]
    return kept, dropped


class QwenPerplexityScorer:
    """Perplexity of Qwen2.5-7B over the text. RUN ON SPARK (or the cloud trainer);
    requires the model pre-downloaded to /twin/models/qwen2.5-7b."""

    def __init__(self, model_dir: str = "/twin/models/qwen2.5-7b") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda",
        )
        self._model.eval()

    def __call__(self, text: str) -> float:
        torch = self._torch
        ids = self._tok(text, return_tensors="pt", truncation=True, max_length=2048)
        ids = {k: v.to("cuda") for k, v in ids.items()}
        with torch.no_grad():
            loss = self._model(**ids, labels=ids["input_ids"]).loss
        return float(torch.exp(loss))
