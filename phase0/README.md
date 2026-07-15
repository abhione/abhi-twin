# phase0 — Spark bring-up

Run these **on the Spark** (over Tailscale SSH) once it's imaged with the NVIDIA
Ubuntu 24.04 aarch64 image (CUDA 13.0, PyTorch NGC nightly cu130).

```bash
make phase0          # setup_dirs.sh + symlink_nvrtc.sh
make verify-phase0   # gate: nvidia-smi, capability (12,1), tailscale, dirs, docker
```

Checklist (manual steps the scripts can't do for you):

- [ ] Flash / boot the NVIDIA-shipped Ubuntu 24.04 aarch64 image
- [ ] `curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up`
      — note the hostname, set `TWIN_SPARK_HOST` in `.env` on the Mac
- [ ] Install Docker + NVIDIA Container Toolkit (`nvidia-ctk runtime configure`)
- [ ] `huggingface-cli login` with a token that can read/write your private repos
- [ ] Create Brev account + API key; Vast.ai as fallback (spec §5)
- [ ] `git clone` this repo onto the Spark, `cp .env.example .env`, fill in keys
- [ ] `make phase0 && make verify-phase0`

Gotcha reminders (all enforced by `ci/preflight.py`, run via the verify gate):

- **Never `pip install torch`** — NGC nightly cu130 aarch64 only. A pip torch
  silently falls back to CPU and fails the capability check.
- CUDA-12.8-pinned packages need the `libnvrtc.so.13 → libnvrtc.so.12.8` symlink.
