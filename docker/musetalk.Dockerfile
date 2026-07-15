# MuseTalk V1.5 sm_121 aarch64 port (video addendum Steps 1-2). RUN ON SPARK.
# The three port gotchas are build steps here, not comments:
#   1. mmcv/mmengine from source (no aarch64 wheels)
#   2. onnxruntime-gpu from source for sm_121 (stock PyPI = silent CPU fallback)
#   3. libnvrtc.so.13 -> 12.8 symlink for CUDA-12.8-pinned deps
FROM nvcr.io/nvidia/pytorch:25.11-py3

# --- gotcha 3: nvrtc symlink --------------------------------------------------
RUN ln -sf /usr/local/cuda/lib64/libnvrtc.so.13 /usr/local/cuda/lib64/libnvrtc.so.12.8

# --- gotcha 1: mmcv from source targeting sm_121 ------------------------------
RUN pip install --no-cache-dir mmengine \
    && TORCH_CUDA_ARCH_LIST="12.1" MMCV_WITH_OPS=1 \
       pip install --no-cache-dir mmcv --no-binary mmcv

# --- gotcha 2: onnxruntime-gpu from source for sm_121 (~30 min build) ----------
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --branch v1.20.1 --depth 1 --recursive \
         https://github.com/microsoft/onnxruntime /opt/onnxruntime \
    && cd /opt/onnxruntime \
    && ./build.sh --config Release --build_wheel --parallel --skip_tests \
         --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda \
         --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=121 \
    && pip install --no-cache-dir build/Linux/Release/dist/onnxruntime_gpu-*.whl \
    && rm -rf /opt/onnxruntime

# --- MuseTalk V1.5 + deps (openmim model pull happens at first run, ~4 GB,
#     into the /twin/models/musetalk volume so rebuilds stay fast) -------------
RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk /opt/musetalk \
    && pip install --no-cache-dir -r /opt/musetalk/requirements.txt \
         --no-deps --ignore-installed || true
RUN pip install --no-cache-dir openmim fastapi uvicorn[standard] websockets \
       insightface opencv-python-headless soundfile

WORKDIR /app
COPY ci/ ci/
COPY serving/ serving/
COPY training/configs/ training/configs/
# streaming wrapper: upstream ships batch scripts; this exposes stream_frames()
COPY docker/patches/musetalk_pipeline.py /opt/musetalk/musetalk/pipeline.py
ENV PYTHONPATH=/app:/opt/musetalk TWIN_MUSETALK_DIR=/twin/models/musetalk

# preflight asserts capability (12,1), nvrtc symlink, AND CUDAExecutionProvider
# (the addendum's "GPU not being hit" failure = CPU fallback caught here)
CMD python ci/preflight.py --check torch && \
    python ci/preflight.py --check nvrtc && \
    python ci/preflight.py --check onnxruntime && \
    uvicorn serving.video.server:app --host 0.0.0.0 --port 8004
