Personal digital twin on one GB10 DGX Spark. Text persona + voice clone + orchestrator in v1; photoreal
video twin in v1.5 as a first-in-community publication artifact.
One Spark, always serving. The Spark is the twin's permanent home — LLM, TTS, STT,
video, orchestrator all live here. Never taken offline for training.
Cloud-burst for training. Every fine-tune (persona LoRA, voice SFT, MuseTalk identity
LoRA) runs on a rented H100 or Spark-in-the-cloud via Brev or Vast.ai. Adapters/weights sync
back to the local Spark. Typical burst cost: $40–80 per fine-tune, hours not days.
Local-first, cloud-fallback for serving too. Sensitive text never leaves the Spark. Complex
reasoning can optionally route to a frontier cloud model via an intent router, following the
NVIDIA/HF Reachy Mini pattern.
Everything is an OpenAI-compatible endpoint. LLM, TTS, STT, embeddings — all served on
/v1/* so the orchestrator sees one uniform interface. Follows Martinb's Qwen3-TTS multi-port
stack precedent.
Skip the community's known gotchas by design. SDPA not flash-attn on Blackwell (Lambert
). Full SFT not LoRA for TTS (Milo v8). Absolute local paths in from_pretrained (Milo
_name_or_path bug). Epsilon-clamp the HiFi-GAN input (Martinb white-noise bug). Symlink
libnvrtc.so.13 for Chatterbox if used (martimramos).
Total: 128 GB unified. Everything that runs concurrently at serving time:
Component
Memory
Notes
Persona LLM (Qwen2.5-72B, NVFP4)
~50 GB
Serving footprint including KV cache headroom
Milo voice TTS (Qwen3-TTS-1.7B, fine-tuned)
~5 GB
Whisper large-v3-turbo STT
~4 GB
CTranslate2 built from source for sm_121
MuseTalk V1.5 (video twin)
~10 GB
Real-time lip-sync
Qdrant + BGE-M3 embeddings
~4 GB
RAG over Brain + corpus
Orchestrator + Ray + FastAPI + LiveKit
~6 GB
OS + drivers + headroom
~10 GB
Total
~89 GB
~39 GB free
Two escape valves if it ever gets tight:
Abhi Twin — Build Spec v2 (Single DGX Spark + Cloud-
Burst Training)
1. Design principles
2. Memory budget (this is the whole game on one Spark)


Drop LLM to Qwen2.5-32B-NVFP4 (~22 GB) — frees 28 GB. Persona quality delta is smaller
than intuition suggests for a personal writing style.
Lazy-load the video twin. MuseTalk only enters memory when a WebRTC video session
opens; text/voice conversations don't pay for it.
┌─────────────────────────── DGX SPARK ─────────────────────────────┐
│                                                                    │
│   ┌────────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│   │ Persona LLM    │───▶│ Orchestrator │───▶│ TTS (Milo)       │──┼─┐
│   │ (72B NVFP4)    │    │  + Router    │    │ STT (Whisper)    │  │ │
│   └────────────────┘    │  + Memory    │    │ MuseTalk (video) │  │ │
│           ▲             │  + RAG       │    └──────────────────┘  │ │
│           │             └──────────────┘             │             │ │
│           └─────────── Qdrant (Brain + corpus) ──────┘             │ │
│                                                                    │ │
└────────────────────────────────────────────────────────────────────┘ │
                                                                       │
                              WebRTC / OpenAI API                      │
                                        ▼                              │
                       Mac (vader/anakin) — client + Comet bridge      │
                                                                       │
                              Cloud burst on demand                    │
                                        ▼                              │
                         ┌────────────────────────┐                    │
                         │ Brev H100 / Vast.ai    │  ◄─────────────────┘
                         │  training only,        │       weight sync
                         │  spun down when done   │       (rsync/HF)
                         └────────────────────────┘
Ubuntu 24.04 aarch64  (NVIDIA-shipped image)
CUDA 13.0             (sm_121 Blackwell)
PyTorch 2.10 NGC nightly (cu130-aarch64)
Python 3.12
Docker + NVIDIA Container Toolkit
Tailscale:  private mesh — Spark, Macs, cloud training nodes
Local storage: /twin/{corpus,checkpoints,adapters,models,logs}
Verified-safe package pins (from the research):
torch from NGC nightly only, never pip install torch (silently CPU-falls-back per
logos_flux and Simon Willison's report)
No flash-attn — SDPA is faster on Blackwell (Lambert)
vLLM: nightly cu130 aarch64 wheels only; if serving fails, build from source
CTranslate2: build v4.7.2+ from source to target sm_121 (per tsuru_mitsu)
3. Hardware topology
4. Base system


onnxruntime-gpu: no stock sm_121 kernels — build from source or use the Blackwell wheel
pattern
Symlink /usr/local/cuda/lib64/libnvrtc.so.13 → libnvrtc.so.12.8 if any package pins
CUDA 12.8
The whole reason cloud-burst works for this workload: the twin retrains rarely. Once every few
weeks after you've collected new corpus data, not continuously. Cloud burst turns a $4k Spark
purchase decision into "do I want another $4k box or five years of on-demand training on tap
for the same money?"
Preferred provider order:
1. Brev — NVIDIA's own service, launches DGX-Spark-equivalent (single GB10) and H100
instances. First choice because environments are pre-built for NVIDIA workflows and
match the local Spark's software stack exactly. Recent Reachy Mini blog uses Brev as the
reference deployment.
2. Vast.ai — spot-market H100/H200. Cheapest path (~$1.50–2.50/hr for H100). Community-run,
so environment quality varies; verify NCCL and driver version on every rental.
3. Lambda Cloud — reliable H100 on-demand at ~$2.50–3/hr. Enterprise-adjacent UX.
4. RunPod — H100 on-demand at ~$2/hr; good runpodctl CLI.
Workflow per fine-tune:
1. Package corpus → tar → upload to HF private repo or S3 (~1–5 min)
2. Launch instance, mount corpus, pull training image
3. Run fine-tune (see per-phase configs below)
4. Push checkpoints/adapters back to HF private repo
5. Terminate instance
6. On local Spark: `hf download` the adapter, hot-load into vLLM
Rough burst budgets (H100, ~$2/hr):
Persona LoRA on 10k pairs: 8–12 h → $16–24
Milo voice full SFT: 6–10 h → $12–20
MuseTalk identity LoRA: 6–10 h → $12–20
Total end-to-end retrain: ~$40–65. Cheaper than a nice dinner.
This is the actual hard part. Budget ~2 weeks of elapsed time for collection + cleaning. Runs
locally on the Spark since it's just data prep.
5. Cloud-burst training strategy
6. Corpus extraction plan


Source
How to pull
Est. yield
Notes
Gmail Sent
Google Takeout → mbox → mail-
parser
3–5k
threads
Filter length ≥ 40 words; strip
signatures/quoted text
iMessage
chat.db on vader Mac → imessage-
exporter
5–10k
msgs
Coalesce your side of conversations
into turns
Slack DMs
Slack export API
1–3k msgs
Only DMs and threads you started
Apple Notes
apple-notes-liberator (vader)
200–500
notes
Long-form voice
GitHub commits/PRs
gh connector
500–2000
PR descriptions and commit bodies
only, not code
Perplexity Brain
pplx digest +
memory/knowledge/*
~50 notes
Your durable opinions
Blog/tweets/long-
form
manual
varies
Highest signal per token
Cleaning pipeline (/twin/corpus/build.py, runs on the Spark):
1. Dedup (MinHash, Jaccard ≥ 0.85 → drop)
2. PII scrub (presidio for names/emails/phone/SSN — replace, don't delete)
3. Length filter (≥ 30 tokens, ≤ 4096)
4. Quality filter (perplexity of a base Qwen2.5-7B; drop top 5% weirdest)
5. Format as {"messages": [...]} — user side is either the actual counterpart's message (for
chat data) or a synthetic prompt reconstructed via Qwen2.5-72B ("given this reply, what was
likely asked?")
6. Hold out 5% as eval set, stratified by source
7. tar czf corpus-v1.tar.gz and upload to your private HF repo for burst training
Deliberate scripts (highest quality): Record 15 min of you reading varied text — Harvard
sentences, technical excerpts from your own writing, emotional/conversational passages.
USB condenser (RØDE NT-USB / Shure MV7). 48 kHz mono WAV.
Natural speech: 15–30 min of a mock podcast or a real Zoom recording (get consent) — strip
the other speaker.
Denoise: demucs for separation, then Adobe Enhance or resemble-enhance for final polish.
Segment: pyannote VAD → 5–15 s clips → force-align transcripts with whisperx → produce
(audio_path, transcript) pairs.
6.1 Text corpus (target: 8–15k high-quality turn pairs)
6.2 Audio corpus (target: 30–60 min clean speech)


Identity reference: 30–60 s frontal video, 1080p 30fps, natural lighting, mouth closed at
start, full expression range (neutral, smile, surprise, speaking). This is what MuseTalk
latches onto.
Identity fine-tune corpus (optional): 3–5 min varied speech in the same setup, WhisperX-
aligned to transcripts. Used only if the stock MuseTalk output doesn't lock hard enough
onto your specific face.
Eval clips: 10–20 held-out audio clips of you speaking (real recordings) — for blind A/B: real
Abhi vs. MuseTalk-Abhi.
Base model: Qwen2.5-72B-Instruct-AWQ (proven fit on Spark; NVFP4 quant variant for local
serving). Training location: cloud H100.
Framework: LLaMA-Factory, matching the little-pink-lora recipe exactly, adjusted for our larger
corpus.
configs/persona-lora.yaml:
model_name_or_path: Qwen/Qwen2.5-72B-Instruct-AWQ
finetuning_type: lora
lora_target: all
lora_rank: 32
# bumped from lmxxf's 16 — we have more data
lora_alpha: 64
learning_rate: 8.0e-5
# slightly lower than 1e-4 baseline
num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
# Blackwell / Hopper both prefer bf16 over fp16
attn_implementation: sdpa
# NOT flash_attn
gradient_checkpointing: true
Expected wall time on H100: ~8–12 h for ~10k pairs at 3 epochs. Push adapter to
hf.co/abhi/persona-v1 (private). Local Spark pulls with hf download abhi/persona-v1 --
local-dir /twin/adapters/persona-v1.
Eval harness (eval/persona.py, runs on Spark):
Held-out perplexity on the 5% eval split.
Style match: cosine similarity of sentence-transformers/all-mpnet-base-v2 embeddings,
twin output vs. real replies to the same prompt.
Blind A/B: 30 prompts, twin reply vs. your reply, judged by Qwen2.5-72B (bias-controlled by
swapping order). Target ≥ 40% "indistinguishable" rate before shipping.
6.3 Video corpus (for MuseTalk in v1.5)
7. Persona LLM — following little-pink-lora


Fact recall: 20 questions about your projects/people (ProSource, Sunny, VSP from your
Brain) — measures RAG effectiveness, not fine-tune memorization.
Serving on Spark: vLLM with the LoRA adapter merged and quantized to NVFP4 (following the
Hermes Agent playbook using nvidia/Qwen3.6-35B-A3B-NVFP4 as the pattern). Endpoint:
http://localhost:8000/v1/chat/completions.
RAG layer: Qdrant on the Spark, indexed corpus of your Brain wiki + notes + docs. BGE-M3
embeddings. Orchestrator injects top-8 retrieved chunks into system prompt.
Base model: Qwen3-TTS-12Hz-1.7B (Milo's proven pick). Full SFT, not LoRA — this is Milo's most
important lesson. Training location: cloud H100.
configs/voice-sft.yaml:
model_name_or_path: Qwen/Qwen3-TTS-12Hz-1.7B
finetuning_type: full
# not lora
learning_rate: 2.0e-5
# Milo v8
num_train_epochs: 25
# early-stop expected by epoch 5–8
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
bf16: true
save_strategy: epoch
early_stopping_patience: 3
early_stopping_threshold: 0.01
Expected wall time on H100: ~4–6 h.
Milo gotcha fix — before pushing checkpoint to HF, patch config.json to strip _name_or_path or
set it to a placeholder like local://voice-v1. On the Spark, load with
AutoModel.from_pretrained("/twin/checkpoints/voice-v1/", local_files_only=True).
Expected metrics: Milo hit summed loss ~10 (avg ~0.65/codec layer). Target the same. RTF 1.15×
eager attention — acceptable; SDPA should push below 1.0×.
Fallback if Milo recipe underperforms your voice: Chatterbox Multilingual zero-shot from a 15-
second reference clip, per martimramos guide. Lower quality ceiling but zero training time —
good stopgap.
Serving: OpenAI-compatible /v1/audio/speech on Spark port 8001. Streaming variant on 8002
(sentence-boundary buffering per logos_flux — target < 800 ms TTFA).
faster-whisper large-v3-turbo, CTranslate2 built from source for sm_121 (per tsuru_mitsu).
Endpoint: /v1/audio/transcriptions, Spark port 8003. Not a twin component but required for
voice-in.
8. Voice clone — following Milo v8
9. STT (input side)


Full detail in the companion doc twin_video_addendum_v2.md. Summary here:
MuseTalk V1.5 on Spark — confirmed viable via sm_120 forward-port to sm_121 aarch64
(PyTorch forums)
Runs concurrently with the LLM at ~10 GB
Fine-tune your identity on cloud H100, serve on Spark
Publication artifact: first working MuseTalk-on-DGX-Spark recipe (single-Spark makes it
more reproducible for others)
Fork NVIDIA's Hermes Agent playbook. It already gives you:
vLLM-served persona LLM with persistent memory
Skill-writing loop (agent modifies its own tool set)
Telegram/Discord/Slack ingress
Scheduled tasks
Additions specific to the twin:
┌──────────────────── AbhiTwin Orchestrator (LangGraph on Spark) ─────────┐
│                                                                          │
│   INTAKE ──► ROUTER ──► RETRIEVER ──► PERSONA LLM ──► TOOLS ──► VOICE   │
│                │           │              │             │        │       │
│                │           │              │             │        └─► TTS │
│                │           │              │             │         + video│
│                │           │              │             └─► email/cal/gh │
│                │           │              │                 (connectors) │
│                │           │              └─► optional cloud fallback    │
│                │           │                  (frontier model via API)   │
│                │           └─► Qdrant (Brain + corpus)                   │
│                │           └─► memory service (per-user, persistent)     │
│                └─► classifies: casual chat / task / sensitive / research│
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
Router policy (mirrors Reachy Mini pattern):
Sensitive (contains email/iMessage/finance) → local persona LLM only
Casual/persona → local persona LLM
Deep research → escalate to your existing Perplexity/Claude/frontier stack via API
Tool-heavy → local LLM with connector calls
Interfaces:
WebRTC ingress via LiveKit on Spark (talk.twin.local:443) — usable from any device
10. Video twin — MuseTalk (v1.5 phase)
11. Orchestrator — Hermes Agent spine


iMessage bridge on vader (already has imessage feature) → tool call → orchestrator
CLI: twin chat, twin voice, twin ask <q> (Python click app)
Phase
Deliverable
Duration
Success criteria
0
Spark imaged, Tailscale,
base stack verified, HF/Brev
accounts ready
1–2 days
nvidia-smi clean,
torch.cuda.get_device_capability() ==
(12,1)
1
Corpus v1: text extraction +
cleaning pipeline
5–7 days
≥ 8k pairs, PII-scrubbed, held-out set frozen, uploaded to HF
2
Persona LoRA v1 trained on
Brev H100, served on Spark
2 days
(mostly
waiting)
Blind A/B ≥ 30% indistinguishable, PPL within 15% of baseline
3
Voice corpus recorded + Milo
SFT v1 on H100
3–4 days
RTF < 1.5×, subjective MOS ≥ 3.8 in self-eval
4
STT + orchestrator + RAG
wired together, WebRTC talk
endpoint
4 days
End-to-end voice conversation < 3 s round trip
5
Persona v2 (with feedback
data) + eval harness
automated
ongoing
Blind A/B ≥ 40%
v1.5
MuseTalk video twin —
port to Spark, publish recipe
1–2 weeks
Live video conversation < 3 s round-trip; recipe reproduces on
another Spark
v2
(opt)
Second Spark added if a
specific bottleneck justifies
it
—
Concrete workload that one Spark can't handle
Total time to a working v1 twin (phases 0–4): ~2.5 weeks of focused work. v1.5 video twin
publication adds another 1–2 weeks. Faster than the two-Spark version (3 weeks) because fewer
moving parts.
Unified-memory freeze (Lambert). On this single-Spark spec you're closer to the memory
ceiling than the two-Spark version, so this matters more. Monitor with nvidia-smi dmon -s
um. If you feel memory pressure, drop LLM to Qwen2.5-32B-NVFP4 first — biggest single
lever.
_name_or_path HF lookup on offline load (Milo) — sanity-check with
local_files_only=True on every load in CI.
Blackwell white-noise TTS (Martinb) — add an epsilon clamp in the HiFi-GAN wrapper and
a unit test that synthesizes a 10 s sample and asserts no NaN in the mel spectrogram.
torch CPU-fallback silent regression — CI check that asserts torch.cuda.is_available()
and torch.cuda.get_device_capability() == (12, 1) at container start.
12. Rollout phases
13. Failure modes to watch


Overfitting the persona LoRA to boilerplate (email signatures, "Best, Abhi"). Aggressive
signature stripping in cleaning + a "boilerplate rate" eval metric.
Voice model drifting to reference speaker's cadence if audio corpus is too small — Milo saw
this at v5. Stop early, evaluate often.
Cloud burst environment drift. The Brev/Vast image is not byte-identical to your Spark.
Solve by shipping a Dockerfile that pins CUDA/Torch/CUDNN versions and building the
same image locally and remotely. Sanity-check with nvidia-smi + torch.version.cuda on
both.
1. Do you want the twin agentic (can send email/schedule for you) or advisory only (drafts,
never sends)? Different guardrail budgets.
2. Do you want iMessage participation — the twin responds as you to specific contacts when
you're offline?
3. Voice-clone consent scope — only for personal/agent use, or would you ever want to
publish twin-narrated content (has legal implications, matches totosse17's pattern)?
4. Cloud-burst provider choice — Brev (NVIDIA-native, matches Spark stack), Vast (cheapest),
Lambda (most reliable), or RunPod (best CLI)? Recommend starting with Brev and
dropping to Vast for repeat runs once the recipe is stable.
Buy the second Spark only when you can name a specific pain that would go away. The likely
triggers, in order:
1. You're retraining more than once a week and cloud-burst is annoying. Adapter-hot-loading
and re-uploading corpus gets old fast. Second Spark as a dedicated trainer removes the
friction.
2. You want the LivePortrait + MuseTalk composite (better full-face motion) and one Spark is
memory-tight when both are loaded alongside the LLM.
3. You want to publish the 405B linked-mode serving demo — separate flex from the twin, but
a real "wow" artifact for the community.
4. You have a customer/consulting engagement where the second Spark pays for itself in one
project.
Until then, one Spark plus H100-on-demand is the right shape. The two-Spark spec
(twin_build_spec.md) is your v2 growth path, not a v1 prerequisite.
Persona LoRA recipe: lmxxf/little-pink-lora + little-pink-ai
Training infra & gotchas: natolambert/dgx-spark-setup, qwen3-dgx-spark-sft
Voice SFT recipe: Milo Voice Cloner
14. Open questions
15. When to actually add a second Spark
Appendix A — Reference builds this spec inherits from


Voice serving stack: Martinb Qwen3-TTS multi-port, xTTS Docker
Chatterbox fallback: martimramos/dgx-spark-ml-guide
Low-latency voice loop: logos_flux 766ms pipeline
Orchestrator spine: Hermes Agent playbook
Router pattern: NVIDIA/HF Reachy Mini assistant
Video twin viability: MuseTalk on sm_120, LivePortrait on RTX 5090
Single-GB10 full digital-human precedent: tsuru_mitsu VITA Reception
