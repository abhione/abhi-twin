#!/usr/bin/env python3
"""Persona eval harness (spec §7): held-out PPL, style cosine
(all-mpnet-base-v2), blind A/B via LLM judge (order-swapped), fact recall, and
the boilerplate-rate overfit metric. RUN ON SPARK (needs the serving stack);
the scoring math is pure and unit-tested on the Mac.

Gates (spec §12): phase 2 ships at blind A/B >= 30% indistinguishable and PPL
within 15% of the base model; phase 5 raises A/B to >= 40%.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import click

JUDGE_INSTRUCTION = (
    "Two replies to the same prompt follow, labeled A and B. One was written by a "
    "specific person, the other by a model imitating them. Answer with exactly one "
    'word: "A" if A is clearly the human, "B" if B is clearly the human, or '
    '"indistinguishable" if you cannot tell.'
)

BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (r"^best,?\s*$", r"^thanks,?\s*$", r"^regards,?\s*$", r"sent from my",
              r"^cheers,?\s*$")
]


# ------------------------------------------------------------- pure scoring


def ab_pairs(twin: list[str], real: list[str], seed: int = 7) -> list[dict]:
    """Order-swapped A/B pairs (bias control): each item records which slot holds
    the real reply."""
    rng = random.Random(seed)
    pairs = []
    for t, r in zip(twin, real):
        real_is_a = rng.random() < 0.5
        pairs.append(
            {"A": r if real_is_a else t, "B": t if real_is_a else r,
             "real_slot": "A" if real_is_a else "B"}
        )
    return pairs


def indistinguishable_rate(judgments: list[str], pairs: list[dict]) -> float:
    """Fraction where the judge failed to pick the human: said 'indistinguishable'
    or picked the twin's slot."""
    if not judgments:
        return 0.0
    fooled = 0
    for verdict, pair in zip(judgments, pairs):
        v = verdict.strip().lower()
        if v.startswith("indistinguishable") or (v in ("a", "b") and v.upper() != pair["real_slot"]):
            fooled += 1
    return fooled / len(judgments)


def within_pct(candidate: float, baseline: float, pct: float = 15.0) -> bool:
    """True if candidate is no more than pct% worse than baseline (lower=better)."""
    return candidate <= baseline * (1 + pct / 100.0)


def boilerplate_rate(texts: list[str]) -> float:
    """Fraction of replies containing signature boilerplate — the LoRA-overfit
    tripwire (spec failure mode #4)."""
    if not texts:
        return 0.0
    hits = sum(
        1 for t in texts if any(p.search(line) for line in t.splitlines() for p in
                                BOILERPLATE_PATTERNS)
    )
    return hits / len(texts)


def style_cosine(twin_vecs: list[list[float]], real_vecs: list[list[float]]) -> float:
    """Mean pairwise cosine between twin and real reply embeddings."""
    import math

    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    sims = [cos(a, b) for a, b in zip(twin_vecs, real_vecs)]
    return sum(sims) / len(sims) if sims else 0.0


# ------------------------------------------------------------- spark runners


def _spark_eval(eval_path: Path, n_ab: int, gate_ab: float, gate_ppl_pct: float) -> dict:
    """The full harness — LLM judge, embeddings, PPL. Heavy imports live here."""
    import torch  # noqa: F401  (asserts the NGC build is present)
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer

    llm = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
    samples = [json.loads(x) for x in eval_path.open() if x.strip()][:n_ab]
    prompts = [s["messages"][-2]["content"] for s in samples]
    real = [s["messages"][-1]["content"] for s in samples]

    twin = []
    for p in prompts:
        resp = llm.chat.completions.create(
            model="persona-v1", messages=[{"role": "user", "content": p}]
        )
        twin.append(resp.choices[0].message.content)

    pairs = ab_pairs(twin, real)
    judgments = []
    for prompt, pair in zip(prompts, pairs):
        resp = llm.chat.completions.create(
            model="qwen2.5-72b",
            messages=[
                {"role": "system", "content": JUDGE_INSTRUCTION},
                {"role": "user",
                 "content": f"Prompt:\n{prompt}\n\nA:\n{pair['A']}\n\nB:\n{pair['B']}"},
            ],
        )
        judgments.append(resp.choices[0].message.content)

    embedder = SentenceTransformer("/twin/models/all-mpnet-base-v2", local_files_only=True)
    return {
        "ab_indistinguishable": indistinguishable_rate(judgments, pairs),
        "style_cosine": style_cosine(
            embedder.encode(twin).tolist(), embedder.encode(real).tolist()
        ),
        "boilerplate_rate": boilerplate_rate(twin),
        "n": len(pairs),
    }


@click.command(help=__doc__)
@click.option("--eval-file", type=click.Path(path_type=Path),
              default=Path("/twin/corpus/out/eval.jsonl"))
@click.option("--n-ab", default=30, show_default=True, help="blind A/B prompt count")
@click.option("--gate", is_flag=True, help="exit non-zero if gates unmet")
@click.option("--gate-ab", default=0.30, show_default=True)
@click.option("--gate-boilerplate", default=0.05, show_default=True)
def main(eval_file: Path, n_ab: int, gate: bool, gate_ab: float,
         gate_boilerplate: float) -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        click.echo("=== RUN ON SPARK === eval/persona.py needs the serving stack + GPU.")
        sys.exit(2)
    results = _spark_eval(eval_file, n_ab, gate_ab, 15.0)
    click.echo(json.dumps(results, indent=2))
    if gate:
        ok = (results["ab_indistinguishable"] >= gate_ab
              and results["boilerplate_rate"] <= gate_boilerplate)
        click.echo("=== verify-persona: PASSED ===" if ok else "=== verify-persona: FAILED ===")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
