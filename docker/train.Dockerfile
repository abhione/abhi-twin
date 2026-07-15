# Cloud-burst training image (Brev H100 / Vast). Pins MATCH the local Spark
# stack (spec failure mode: "cloud burst environment drift") — same NGC line,
# same torch/CUDA family; sanity-check nvidia-smi + torch.version.cuda on both.
FROM nvcr.io/nvidia/pytorch:25.11-py3

RUN pip install --no-cache-dir \
      "llamafactory[metrics]" \
      "huggingface_hub[cli]" \
      pyyaml click \
    && pip check | grep -v torch || true

WORKDIR /app
COPY training/ training/
COPY ci/ ci/
COPY serving/tts/guards.py serving/tts/guards.py

# entry: training/burst/run_train.sh <config.yaml> <artifact-name>
# (downloads the private corpus tarball, trains, patches _name_or_path, pushes)
ENV PYTHONPATH=/app
ENTRYPOINT ["bash"]
