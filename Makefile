# AbhiTwin harness — phase targets + verify gates.
# Host legend:  [mac]   runs on this Mac (starbase)
#               [spark] RUN ON SPARK — scripts self-guard and print a marker elsewhere
#               [cloud] runs on the rented Brev/Vast trainer
SHELL := /bin/bash
VENV  := .venv
PY    := $(VENV)/bin/python
PIP   := $(VENV)/bin/pip
RUFF  := $(VENV)/bin/ruff
# --project-directory . makes compose read the repo-root .env (otherwise it
# looks for docker/.env and TWIN_* overrides are silently ignored)
COMPOSE := docker compose --project-directory . -f docker/compose.yaml

.DEFAULT_GOAL := help

help: ## Show targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- local dev
# Spec pins Python 3.12; on macs where `python3` is older, prefer python3.12.
PYTHON ?= $(shell command -v python3.12 2>/dev/null || command -v python3)

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip

setup: $(VENV)/bin/activate ## [mac] venv + editable install with dev deps
	$(PIP) install -e ".[dev]"

lint: setup ## [mac] ruff
	$(RUFF) check .

test-local: setup lint ## [mac] everything runnable on macOS: unit tests + preflight local checks
	$(PY) -m pytest
	$(PY) ci/preflight.py --local-only

# ---------------------------------------------------------------- phase 0
phase0: ## [spark] bring-up: dirs, nvrtc symlink
	bash phase0/setup_dirs.sh
	bash phase0/symlink_nvrtc.sh

verify-phase0: ## [spark] gate: nvidia-smi clean, capability (12,1), tailscale, dirs
	bash phase0/verify_spark.sh

# ---------------------------------------------------------------- phase 1: corpus
corpus: setup ## [mac] extract all sources -> clean -> holdout -> corpus/data/out
	$(PY) -m cli.twin corpus --local

verify-corpus: setup ## [mac|spark] gate: >=8k pairs, PII-scrubbed, holdout frozen
	$(PY) corpus/verify.py --out corpus/data/out

corpus-sync: ## [mac] rsync cleaned corpus to the Spark
	$(PY) -m cli.twin corpus --sync

corpus-upload: ## [mac|spark] tar + push corpus to private HF repo for burst training
	bash training/burst/sync_corpus.sh

# ---------------------------------------------------------------- phases 2/3: training (cloud burst)
train-persona: ## [mac] launch persona LoRA burst (Brev first, Vast fallback)
	bash training/burst/launch_brev.sh training/configs/persona-lora.yaml persona-v1

train-voice: ## [mac] launch Milo voice full-SFT burst
	bash training/burst/launch_brev.sh training/configs/voice-sft.yaml voice-v1

train-musetalk: ## [mac] launch MuseTalk identity LoRA burst (v1.5, optional)
	bash training/burst/launch_brev.sh training/configs/musetalk-identity.yaml musetalk-identity-v1

fetch-adapters: ## [spark] hf download trained adapters/checkpoints into /twin
	bash training/burst/fetch_adapter.sh

verify-persona: ## [spark] gate: blind A/B >=30% indistinguishable, PPL within 15%
	$(PY) eval/persona.py --gate

verify-voice: ## [spark] gate: RTF < 1.5x, 10s synth with no NaN in mel
	$(PY) eval/voice.py --gate

# ---------------------------------------------------------------- phase 4: serving
serve: ## [spark] core stack: LLM + Qdrant + orchestrator
	$(COMPOSE) --profile core up -d

serve-voice: ## [spark] + TTS/STT
	$(COMPOSE) --profile core --profile voice up -d

serve-all: ## [spark] + MuseTalk video (lazy-loads weights on first session)
	$(COMPOSE) --profile core --profile voice --profile video up -d

serve-down: ## [spark] stop the stack
	$(COMPOSE) --profile core --profile voice --profile video down

verify-e2e: ## [spark] gate: voice round trip < 3 s
	$(PY) scripts/e2e_roundtrip.py --gate

# ---------------------------------------------------------------- v1.5: video
video: serve-all ## [spark] alias: full stack incl. MuseTalk

video-demo: ## [spark] reproducibility gate: clean clone -> working MuseTalk demo < 30 min
	bash serving/video/demo.sh

verify-video: ## [spark] gate: FPS >= 10 @256, latency + memory within budget
	$(PY) eval/video.py --gate

# ---------------------------------------------------------------- misc
preflight: ## [spark] full gotcha checklist (CUDA, capability, nvrtc, ORT providers)
	$(PY) ci/preflight.py

clean: ## [mac] remove venv + caches (never touches corpus data)
	rm -rf $(VENV) .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: help setup lint test-local phase0 verify-phase0 corpus verify-corpus corpus-sync \
        corpus-upload train-persona train-voice train-musetalk fetch-adapters verify-persona \
        verify-voice serve serve-voice serve-all serve-down verify-e2e video video-demo \
        verify-video preflight clean
