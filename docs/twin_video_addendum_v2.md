Addendum to twin_build_spec_v2.md. Phase v1.5, promoted to a full deliverable — and the single-
Spark version is actually the more valuable publication artifact.
The community consensus was "no one has done this on Spark." Fresh research changes the
shape of the problem:
MuseTalk V1.5 is confirmed working on sm_120 (RTX 5060 Ti, 5070 Ti, 5090) with
torch==2.10.0+cu128 — see PyTorch forums confirmation.
LivePortrait is confirmed working on RTX 5090 with pre-compiled
MultiScaleDeformableAttention for CUDA 12.8/Torch 2.7 — see KlingTeam HF discussion.
DGX Spark is sm_121 (GB10) — a Grace-Blackwell variant of sm_120. Kernel-level
differences are minor; the real work is aarch64 wheels + CUDA 13 vs 12.8.
So the porting problem isn't "make talking-head models work on Blackwell" (already done on
desktop 50-series). It's "take the working sm_120 x86_64 recipe and forward-port to sm_121
aarch64 on CUDA 13" — one architecture bump and one ABI change. That's a weekend of work
per model.
Why single-Spark is a stronger publication target than two-Spark: more Spark owners have
one than two. A recipe that lands on one Spark reproduces for the whole community. A two-
Spark recipe only reproduces for buyers who committed $8k+. The most-cited Spark
community posts (Milo, Martinb, logos_flux) are all single-node builds. Follow the pattern.
Ranking the four candidates from the research report for a Spark first-port:
Model
Approach
sm_120
status
aarch64 lift
Quality
ceiling
Recommendation
MuseTalk
V1.5
Latent-space
lip-sync
inpainting,
~30 FPS at
256×256
Confirmed
working
sm_120
Small
Good;
identity
strongly
preserved
Start here
LivePortrait
Implicit
keypoint
warping
from driving
video/audio
Confirmed
working
sm_120
Small–medium (needs
MultiScaleDeformableAttention
rebuild)
Good;
better full-
face
motion
Second port after
MuseTalk
Photoreal Video Twin on One DGX Spark — First-in-
Community Recipe
Why one Spark is the right target for the publication
Model choice — MuseTalk as the beachhead


Model
Approach
sm_120
status
aarch64 lift
Quality
ceiling
Recommendation
Hallo2
Diffusion-
based;
higher
fidelity,
slower
A100-only in
official docs
Large — full aarch64/sm_121
rebuild
Best of
the four
Reach goal; likely
infeasible
alongside 72B LLM
on one Spark
SadTalker
Older 3DMM-
based
Not verified
on Blackwell
Medium; less maintained
Weakest
of the four
Skip
Verdict for one-Spark v1.5: MuseTalk-only. LivePortrait composite becomes practical only if
you drop the LLM to Qwen2.5-32B or add the second Spark. Hallo2 requires more memory
headroom than one Spark can afford alongside a persona LLM.
From the main spec's serving table, all-loaded:
Component
Memory
Persona LLM (Qwen2.5-72B NVFP4)
~50 GB
MuseTalk V1.5 (streaming inference)
~10 GB
Milo voice TTS
~5 GB
Whisper STT
~4 GB
Qdrant + BGE-M3
~4 GB
Orchestrator + Ray + FastAPI + LiveKit
~6 GB
OS + drivers + headroom
~10 GB
Total
~89 GB with ~39 GB free
Comfortable. MuseTalk lazy-loads only when a WebRTC video session opens, so text/voice-only
interactions don't pay for it — that gives even more headroom for LLM KV cache when video
isn't active.
Identity reference: 30–60 s frontal video, 1080p 30fps, natural lighting, mouth closed at
start, full expression range (neutral, smile, surprise, speaking). This is what MuseTalk
latches onto.
Fine-tune corpus (optional, for identity strengthening): 3–5 min varied speech in the same
recording setup. WhisperX-force-aligned. Only train an identity LoRA if the base MuseTalk
output doesn't lock onto your specific features.
Eval clips: 10–20 held-out audio clips of you speaking, for A/B: real Abhi vs. MuseTalk-Abhi,
blind judged by 3–5 people who know your face.
Memory budget with MuseTalk live
Video corpus (already in the main spec — repeating here for reference)


Goal: working real-time lip-sync of your face from any audio, running on your one Spark
alongside the rest of the twin stack. Publish the recipe.
nvcr.io/nvidia/pytorch:25.11-py3-aarch64 (or latest NGC aarch64 build). Verify:
python -c "import torch; assert torch.cuda.get_device_capability() == (12, 1); print
Expected: NVIDIA GB10.
Clone official repo. Expect friction on:
mmcv and mmengine — no aarch64 wheels; build from source with:
TORCH_CUDA_ARCH_LIST="12.1" MMCV_WITH_OPS=1 pip install mmcv --no-binary mmcv
openmim model download step — works fine, just slow (models are ~4 GB total).
onnxruntime-gpu — stock PyPI has no sm_121 kernels (Blackwell wheel repo confirms this for
sm_120; sm_121 is same fix). Options:
1. Build ORT from source targeting sm_121 (best, ~30 min build).
2. Fall back to onnxruntime (CPU) for face detection — usable but slower.
3. Replace insightface's ONNX face detector with a CUDA-native PyTorch alternative (e.g.
facexlib).
insightface — CPU-fallback silent regression; verify providers=
['CUDAExecutionProvider'] actually resolves.
Run the stock MuseTalk demo (Obama sample) end-to-end. Expected on GB10:
~30 FPS at 256×256
~15 FPS at 512×512
~8–10 GB memory
If FPS drops below 10, you have an ORT-fallback or PTX-JIT issue — check nvidia-smi dmon
while it runs; a GPU that's not being hit indicates a CPU fallback somewhere in the pipeline.
The port — MuseTalk on Spark, single-node
Step 1 — Base container
Step 2 — Install MuseTalk V1.5
Step 3 — Sanity inference


Point MuseTalk at your identity reference video + one of your Milo TTS outputs. Judge quality:
Lip sync locks but identity drifts → phase Step 5 (identity fine-tune).
Lip sync itself is bad → check audio preprocessing (SR must be 16 kHz; Whisper features
expected). Usually a resampling bug.
Face detection fails → ORT fallback bit you; go back to Step 2.
MuseTalk uses a VAE + UNet inpainting head; a small LoRA on the UNet trained against your 3–5
min corpus locks identity harder. Reference: Milo's iterative approach is a good template — small
dataset, low LR, early-stop by epoch 5, avoid full SFT unless a LoRA plateaus.
Cloud burst: package video corpus → HF → Brev H100 → train ~6–10 h → push adapter → local
Spark hf download → merge or hot-load. ~$12–20 per iteration.
Wrap MuseTalk in a FastAPI service that accepts audio chunks over WebSocket and streams
JPEG frames back. Follow the sentence-boundary buffering pattern from logos_flux's 766ms
pipeline. Target: < 500 ms end-to-end audio→first-frame latency.
Lazy-load pattern: MuseTalk model stays unloaded until first WebRTC video session opens.
Frees ~10 GB for LLM KV cache during text/voice-only interactions.
MuseTalk output frames → LiveKit ingress → WebRTC out. Client is browser or FaceTime-style
app on vader/anakin. Same LangGraph node structure from the main spec, just adds a
video_out sink alongside the existing audio_out.
To make this land as a real first-in-community recipe:
1. Reproducibility gate. Any Spark owner must be able to git clone && docker compose up
and get a working demo in under 30 minutes. This is the bar for r/LocalLLaMA and NVIDIA
Developer Forums.
2. Benchmark table. FPS at 256×256 and 512×512, memory usage, latency (audio-in → first-
frame-out), power draw, concurrent-with-LLM performance. Match the Milo / Martinb /
logos_flux precedent of leading with concrete numbers.
3. Gotchas as first-class content. The Spark community's most-cited posts document what
broke and how they fixed it. Publish the wheel-hunt, the libnvrtc.so.13 symlink, the ORT
sm_121 build, any _name_or_path recurrences — that's what other builders search for.
4. Cross-post targets, in priority order:
Step 4 — Feed your face
Step 5 (optional) — Identity fine-tune on cloud H100
Step 6 — Streaming server
Step 7 — Wire into orchestrator
Publication artifact


NVIDIA Developer Forums — DGX Spark section — Martinb, tsuru_mitsu, provos are active
there
r/LocalLLaMA — biggest audience
Hacker News — Jeff Geerling / Simon Willison audience
NVIDIA/dgx-spark-playbooks PR
Hugging Face model card + Space demo
PR to the MuseTalk upstream repo adding a deploy/dgx-spark/ section
5. Optionally submit to GTC 2027. Talk title: "A Photoreal Personal AI Twin on a Single DGX
Spark." (The single-Spark framing is more compelling than dual — it's the box everyone
has.)
The identity fine-tune undertrains or overfits — same failure modes Milo hit with voice (v5
overfit at 1e-4, v6 underfit at 1e-5). Mitigation: adopt his exact eval loop — synthesize a fixed
set of prompts after each epoch, human-judge, early-stop by drift.
Uncanny valley. MuseTalk's default output is competent but not photoreal — mouth region
can look slightly waxen. Fix in postprocess: light Gaussian blur + film grain on the mouth
region, or upscale via Real-ESRGAN (also needs sm_121 build).
Memory pressure with video + 72B LLM. If you feel it, drop LLM to Qwen2.5-32B-NVFP4
first — biggest single lever. LivePortrait composite is not viable on one Spark alongside a
72B LLM.
You'd be the QA. No one else has done this, so no one else can tell you if your output looks
right. Recruit 3–5 people who know your face for blind A/B.
Consent scope. A photoreal video twin is exactly the artifact regulators, employers, and
platforms worry about. Watermark outputs (imperceptible via Google SynthID or similar),
keep a signed usage log, don't publish clips that could be mistaken for real-you-on-camera
without a visible label.
v1.5 ships when:
Your twin, running on the single Spark, can hold a live video conversation with you (voice-
in → STT → persona LLM → TTS → MuseTalk → WebRTC out) at < 3 s round-trip.
The port recipe is public, reproducible, and cited by at least one other Spark builder.
You have a benchmark table someone at NVIDIA can point to.
You've submitted a PR (or issue with reproducer) to at least one of the upstream repos.
At that point you are, definitionally, first-in-community — and the recipe reproduces on the box
everyone bought.
Risks specific to this phase
What "done" looks like


MuseTalk sm_120 confirmation: PyTorch forums post 85
LivePortrait RTX 5090 install: KlingTeam/LivePortrait discussion #40
Blackwell ONNX Runtime pre-built wheel: Natfii/onnxruntime-gpu-blackwell
Style-Bert-VITS2 Blackwell containerization (Dockerfile template): Zenn article
Building CUDA extensions targeting sm_120 in NGC: Medium walkthrough
Adjacent Spark video pipeline (fictional subject, but WAN2.2/MultiTalk on Spark works):
Amegilla — rap animation on Spark
Single-Spark full digital-human precedent (stylized VRM): tsuru_mitsu VITA Reception
Master reference for Spark training gotchas: natolambert/dgx-spark-setup
Voice-fine-tune iterative template (same pattern for MuseTalk identity LoRA): Milo Voice
Cloner
Streaming buffering pattern for < 500 ms latency: logos_flux 766ms pipeline
Appendix — direct references
