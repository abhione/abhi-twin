# Milo voice TTS: Qwen3-TTS serving, OpenAI /v1/audio/speech on :8001
# (streaming endpoint also served; compose maps :8002 to it). RUN ON SPARK.
FROM nvcr.io/nvidia/pytorch:25.11-py3

# torch stays the NGC build; everything else is CPU-side plumbing
RUN pip install --no-cache-dir fastapi uvicorn[standard] soundfile transformers accelerate

# Qwen3-TTS runtime (base-voice placeholder until voice-v1 is trained; the spec's
# Qwen/Qwen3-TTS-12Hz-1.7B repo id does not exist — the released family is
# -Base/-CustomVoice/-VoiceDesign, served via the qwen-tts package, not AutoModel).
# qwen-tts goes in --no-deps so pip never touches the NGC torch; transformers/
# accelerate use its exact metadata pins. Its 25Hz tokenizer imports torchaudio/
# onnxruntime/sox at module scope (the served 12Hz path never calls them) — a pip
# torchaudio would drag in a CPU torch, so torchaudio is stubbed (burst lesson);
# sox/onnxruntime are import-only and CPU-side here.
RUN pip install --no-cache-dir --no-deps qwen-tts==0.1.1 \
    && pip install --no-cache-dir "transformers==4.57.3" "accelerate==1.12.0" \
       sox onnxruntime librosa einops \
    && SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && mkdir -p "$SITE/torchaudio/compliance" \
    && touch "$SITE/torchaudio/__init__.py" "$SITE/torchaudio/compliance/__init__.py" \
    && printf 'def __getattr__(name):\n    raise RuntimeError("torchaudio is stubbed in the tts image; the served 12Hz path never uses kaldi")\n' \
       > "$SITE/torchaudio/compliance/kaldi.py" \
    && python -c "import qwen_tts; print('qwen_tts import ok')"

# fail the BUILD (not first serve) if pip replaced the NGC torch with a CPU wheel
RUN python -c "import torch; assert torch.version.cuda, 'pip clobbered the NGC torch with a CPU build'"

WORKDIR /app
COPY ci/ ci/
COPY serving/ serving/
COPY training/configs/ training/configs/
# verify-voice runs inside this container against the live :8001 endpoint
COPY eval/ eval/

ENV TWIN_VOICE_CHECKPOINT=/twin/checkpoints/voice-v1 \
    PYTHONPATH=/app

# preflight enforces: capability (12,1), epsilon clamp present, checkpoint has
# no hub-shaped _name_or_path (the load itself uses local_files_only=True)
CMD python ci/preflight.py --check torch && \
    python ci/preflight.py --check epsilon-clamp && \
    python ci/preflight.py --check checkpoints && \
    uvicorn serving.tts.server:app --host 0.0.0.0 --port 8001
