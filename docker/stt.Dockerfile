# STT: faster-whisper large-v3-turbo on CTranslate2 BUILT FROM SOURCE for sm_121
# (per tsuru_mitsu — no stock aarch64 sm_121 wheels). Port 8003. RUN ON SPARK.
FROM nvcr.io/nvidia/pytorch:25.11-py3

# --- CTranslate2 v4.7.2+ source build targeting sm_121 -----------------------
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --recursive --branch v4.7.2 --depth 1 \
      https://github.com/OpenNMT/CTranslate2 /opt/ctranslate2 \
    && cd /opt/ctranslate2 \
    && cmake -S . -B build -GNinja \
         -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=OFF -DOPENMP_RUNTIME=COMP \
         -DCMAKE_CUDA_ARCHITECTURES=121 \
    && cmake --build build --target install -j"$(nproc)" \
    && ldconfig \
    && cd python && pip install --no-cache-dir -r install_requirements.txt \
    && pip install --no-cache-dir . \
    && cd / && rm -rf /opt/ctranslate2/build

# faster-whisper on top of the source-built ctranslate2 (never overwrite it)
RUN pip install --no-cache-dir --no-deps faster-whisper \
    && pip install --no-cache-dir "tokenizers>=0.13" "onnxruntime<2" av tqdm \
       fastapi uvicorn[standard] python-multipart

WORKDIR /app
COPY ci/ ci/
COPY serving/ serving/
COPY training/configs/ training/configs/
ENV TWIN_STT_MODEL=/twin/models/whisper-large-v3-turbo PYTHONPATH=/app

CMD python ci/preflight.py --check torch && \
    uvicorn serving.stt.server:app --host 0.0.0.0 --port 8003
