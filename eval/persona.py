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


def _verdict_token(verdict: str) -> str:
    """First alphanumeric token of a judge reply, lowercased — verbose judges
    answer 'B.' / '**A**' / 'B — the phrasing…'; scoring must not depend on the
    judge's formatting discipline (a dropped verdict silently counts not-fooled)."""
    m = re.search(r"[a-z]+", verdict.lower())
    return m.group(0) if m else ""


def indistinguishable_rate(judgments: list[str], pairs: list[dict]) -> float:
    """Fraction where the judge failed to pick the human: said 'indistinguishable'
    or picked the twin's slot."""
    if not judgments:
        return 0.0
    fooled = 0
    for verdict, pair in zip(judgments, pairs):
        v = _verdict_token(verdict)
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


# ------------------------------------------------------------- external judge


def judge_claude_cli(prompts: list[str], pairs: list[dict], claude_bin: str,
                     claude_model: str) -> list[str]:
    """Re-judge persisted A/B pairs with a non-Qwen judge via the local claude
    CLI (one call per pair). Runs on the Mac — no GPU/serving stack needed.
    `env -u ANTHROPIC_API_KEY` is load-bearing: the CLI must bill the Max
    subscription, never the API key that may be exported in the shell."""
    import subprocess

    judgments = []
    for i, (prompt, pair) in enumerate(zip(prompts, pairs), 1):
        content = f"{JUDGE_INSTRUCTION}\n\nPrompt:\n{prompt}\n\nA:\n{pair['A']}\n\nB:\n{pair['B']}"
        out = subprocess.run(
            ["env", "-u", "ANTHROPIC_API_KEY", claude_bin, "-p", "--model", claude_model],
            input=content, capture_output=True, text=True, timeout=600, check=True,
        ).stdout.strip()
        judgments.append(out)
        click.echo(f"  [{i}/{len(pairs)}] {out.splitlines()[0][:40] if out else '(empty)'}",
                   err=True)
    return judgments


# ------------------------------------------------------------- spark runners


def _held_out_ppl(llm, model: str, prompt_texts: list[str], reply_texts: list[str]) -> float:
    """Reply-only perplexity over the holdout via vLLM's prompt_logprobs extension
    (no logits access needed — works against the OpenAI-compatible endpoint, LoRA
    adapters included). Two calls per sample: the prompt-only call gives the token
    boundary, the full call gives per-token logprobs for the gold reply."""
    import math

    total_lp, total_tok = 0.0, 0
    for prompt, reply in zip(prompt_texts, reply_texts):
        n_prompt = len(
            llm.completions.create(
                model=model, prompt=prompt, max_tokens=1, temperature=0,
                extra_body={"prompt_logprobs": 0},
            ).choices[0].prompt_logprobs
        )
        full = llm.completions.create(
            model=model, prompt=prompt + reply, max_tokens=1, temperature=0,
            extra_body={"prompt_logprobs": 0},
        )
        for entry in full.choices[0].prompt_logprobs[n_prompt:]:
            if not entry:  # first token of a sequence has no logprob
                continue
            total_lp += next(iter(entry.values()))["logprob"]
            total_tok += 1
    return math.exp(-total_lp / total_tok) if total_tok else float("inf")


def _spark_eval(eval_path: Path, n_ab: int, gate_ab: float, gate_ppl_pct: float,
                pairs_out: Path | None = None) -> dict:
    """The full harness — LLM judge, embeddings, PPL. Heavy imports live here."""
    import os

    import torch  # noqa: F401  (asserts the NGC build is present)
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    llm = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
    samples = [json.loads(x) for x in eval_path.open() if x.strip()][:n_ab]
    prompts = [s["messages"][-2]["content"] for s in samples]
    real = [s["messages"][-1]["content"] for s in samples]

    # PPL is measured on chat-templated text — the distribution the LoRA was
    # trained on — using the serving tokenizer for exact token boundaries.
    model_dir = os.environ.get("TWIN_LLM_MODEL_DIR", "/twin/models/qwen2.5-72b-instruct-awq")
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    prompt_texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    reply_texts = [r + tok.eos_token for r in real]
    ppl_base = _held_out_ppl(llm, "qwen2.5-72b", prompt_texts, reply_texts)
    ppl_persona = _held_out_ppl(llm, "persona-v1", prompt_texts, reply_texts)

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

    if pairs_out:
        # persist the exact pairs + judgments so an external judge (--judge
        # claude-cli on the Mac) can re-score the SAME order-swapped pairs
        pairs_out.parent.mkdir(parents=True, exist_ok=True)
        pairs_out.write_text(json.dumps(
            {"seed": 7, "n": len(pairs), "prompts": prompts, "pairs": pairs,
             "judge": "qwen2.5-72b", "judgments": judgments}, indent=2))

    embedder = SentenceTransformer("/twin/models/all-mpnet-base-v2", local_files_only=True)
    return {
        "ab_indistinguishable": indistinguishable_rate(judgments, pairs),
        "style_cosine": style_cosine(
            embedder.encode(twin).tolist(), embedder.encode(real).tolist()
        ),
        "boilerplate_rate": boilerplate_rate(twin),
        "ppl_base": ppl_base,
        "ppl_persona": ppl_persona,
        "ppl_within_pct": within_pct(ppl_persona, ppl_base, gate_ppl_pct),
        "n": len(pairs),
    }


@click.command(help=__doc__)
@click.option("--eval-file", type=click.Path(path_type=Path),
              default=Path("/twin/corpus/out/eval.jsonl"))
@click.option("--n-ab", default=30, show_default=True, help="blind A/B prompt count")
@click.option("--gate", is_flag=True, help="exit non-zero if gates unmet")
@click.option("--gate-ab", default=0.30, show_default=True)
@click.option("--gate-boilerplate", default=0.05, show_default=True)
@click.option("--judge", type=click.Choice(["qwen", "claude-cli"]), default="qwen",
              show_default=True, help="qwen = local vLLM base model (spark); "
              "claude-cli = external non-Qwen judge over persisted pairs (mac)")
@click.option("--pairs-out", type=click.Path(path_type=Path), default=None,
              help="[qwen judge] persist A/B pairs + judgments here for re-judging")
@click.option("--pairs-file", type=click.Path(exists=True, path_type=Path), default=None,
              help="[claude-cli judge] pairs JSON persisted by a --pairs-out run")
@click.option("--claude-bin", default="claude", show_default=True)
@click.option("--claude-model", default="claude-haiku-4-5-20251001", show_default=True)
def main(eval_file: Path, n_ab: int, gate: bool, gate_ab: float,
         gate_boilerplate: float, judge: str, pairs_out: Path | None,
         pairs_file: Path | None, claude_bin: str, claude_model: str) -> None:
    if judge == "claude-cli":
        if pairs_file is None:
            raise click.UsageError("--judge claude-cli needs --pairs-file "
                                   "(persisted by a spark run with --pairs-out)")
        data = json.loads(pairs_file.read_text())
        judgments = judge_claude_cli(data["prompts"], data["pairs"], claude_bin, claude_model)
        results = {
            "judge": f"claude-cli:{claude_model}",
            "ab_indistinguishable": indistinguishable_rate(judgments, data["pairs"]),
            "n": len(data["pairs"]),
            "self_judge_ab": indistinguishable_rate(data.get("judgments", []), data["pairs"]),
            "judgments": judgments,  # persisted for audit — scoring bugs hide here
        }
        click.echo(json.dumps(results, indent=2))
        if gate:
            ok = results["ab_indistinguishable"] >= gate_ab
            click.echo("=== verify-persona (external judge): PASSED ===" if ok
                       else "=== verify-persona (external judge): FAILED ===")
            sys.exit(0 if ok else 1)
        return
    try:
        import torch  # noqa: F401
    except ImportError:
        click.echo("=== RUN ON SPARK === eval/persona.py needs the serving stack + GPU.")
        sys.exit(2)
    results = _spark_eval(eval_file, n_ab, gate_ab, 15.0, pairs_out)
    click.echo(json.dumps(results, indent=2))
    if gate:
        ok = (results["ab_indistinguishable"] >= gate_ab
              and results["boilerplate_rate"] <= gate_boilerplate
              and results["ppl_within_pct"])
        click.echo("=== verify-persona: PASSED ===" if ok else "=== verify-persona: FAILED ===")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
