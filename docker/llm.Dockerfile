# Persona LLM: vLLM on cu130 aarch64. RUN ON SPARK.
# HARD RULE: torch comes from the NGC image — never `pip install torch`
# (silent CPU fallback). Preflight asserts capability (12,1) at container start.
FROM nvcr.io/nvidia/pytorch:25.11-py3

# vLLM nightly cu130 aarch64 wheels only; if serving fails, build from source
# (spec §4). --no-deps-torch: never let pip touch the NGC torch.
RUN pip install --no-cache-dir --pre vllm \
      --extra-index-url https://wheels.vllm.ai/nightly \
    && pip check | grep -v torch || true

# fail the BUILD (not first serve) if pip replaced the NGC torch with a CPU wheel
RUN python -c "import torch; assert torch.version.cuda, 'pip clobbered the NGC torch with a CPU build'"

WORKDIR /app
COPY ci/ ci/
COPY serving/ serving/
COPY training/configs/ training/configs/

# serve the NVFP4-quantized merge with the persona LoRA hot-loadable
# (Hermes Agent playbook pattern). Model + adapter live on the /twin volume.
ENV TWIN_LLM_MODEL_DIR=/twin/models/qwen2.5-72b-nvfp4 \
    TWIN_PERSONA_ADAPTER=/twin/adapters/persona-v1

CMD python ci/preflight.py --check torch && \
    vllm serve "$TWIN_LLM_MODEL_DIR" \
      --served-model-name persona-v1 qwen2.5-72b \
      --enable-lora --lora-modules persona-v1="$TWIN_PERSONA_ADAPTER" \
      --max-model-len 8192 \
      --port 8000
