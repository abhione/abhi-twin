# Milo voice TTS: Qwen3-TTS serving, OpenAI /v1/audio/speech on :8001
# (streaming endpoint also served; compose maps :8002 to it). RUN ON SPARK.
FROM nvcr.io/nvidia/pytorch:25.11-py3

# torch stays the NGC build; everything else is CPU-side plumbing
RUN pip install --no-cache-dir fastapi uvicorn[standard] soundfile transformers accelerate

WORKDIR /app
COPY ci/ ci/
COPY serving/ serving/
COPY training/configs/ training/configs/

ENV TWIN_VOICE_CHECKPOINT=/twin/checkpoints/voice-v1 \
    PYTHONPATH=/app

# preflight enforces: capability (12,1), epsilon clamp present, checkpoint has
# no hub-shaped _name_or_path (the load itself uses local_files_only=True)
CMD python ci/preflight.py --check torch && \
    python ci/preflight.py --check epsilon-clamp && \
    python ci/preflight.py --check checkpoints && \
    uvicorn serving.tts.server:app --host 0.0.0.0 --port 8001
