# Recording the voice + video corpus (human checklist)

Spec §6.2/§6.3. Total sitting time: ~90 minutes across two sessions. Everything
records on the Mac; raw media stays local (gitignored, never uploaded except as
the private HF tarball for the burst trainer).

## Session 1 — voice (target: 30–60 min clean speech)

### Setup
- [ ] USB condenser mic (RØDE NT-USB or Shure MV7), 10–15 cm from mouth, slight off-axis
- [ ] Quiet room, soft furnishings, HVAC off; phone on do-not-disturb
- [ ] Record **48 kHz mono WAV** (QuickTime → File → New Audio Recording →
      Maximum quality, or `sox -d -r 48000 -c 1 take1.wav`)
- [ ] 10 s of room tone first (the denoiser wants it)

### Deliberate scripts (~15 min — highest quality per minute)
- [ ] Harvard sentences, lists 1–10 (72 sentences): https://www.cs.columbia.edu/~hgs/audio/harvard.html
      Read naturally, not announcer-voice. Re-take a sentence if you stumble.
- [ ] Technical excerpts **from your own writing** (~5 min): pull 3–4 long
      paragraphs from your emails/notes — reading your own words captures your cadence
- [ ] Emotional/conversational passages (~5 min): tell two stories you actually
      tell people — laugh where you'd laugh

### Natural speech (15–30 min)
- [ ] Mock podcast: have someone interview you over Zoom **with consent to
      record**, or record your side of a real call
- [ ] Strip the other speaker later — don't worry about it while recording

### Post-processing (scripted, run per file)
```bash
demucs --two-stems=vocals take1.wav          # separate your voice
# then Adobe Podcast Enhance or resemble-enhance on the vocals stem
# segment + align (RUN ON SPARK or any GPU box):
whisperx vocals_enhanced.wav --align --output_format json
# pyannote VAD -> 5–15 s clips -> (audio_path, transcript) pairs into
# /twin/corpus/audio/  (this is voice-sft.yaml's dataset_dir)
```

## Session 2 — video (v1.5 MuseTalk)

Same room, camera at eye level, 1080p 30fps, natural window light on your face
(not behind you), plain background.

- [ ] **Identity reference (required): 30–60 s frontal.** Start with mouth
      CLOSED for 2 s. Then run the expression range: neutral → smile →
      surprise → normal speaking. Save as `/twin/corpus/video/identity.mp4`
- [ ] **Identity fine-tune corpus (optional): 3–5 min varied speech**, same
      setup — only needed if stock MuseTalk doesn't lock onto your face
      (WhisperX-align it like the audio corpus)
- [ ] **Eval clips: 10–20 held-out clips of real you speaking** (reuse podcast
      video if you have it) — these are the blind A/B set; recruit 3–5 people
      who know your face as judges

## Consent + provenance (addendum risk list — do these, they're not optional)
- [ ] Written consent from anyone else appearing in recordings
- [ ] Keep a signed usage log for the voice/video models
- [ ] Watermark published twin output (SynthID or similar) and label it as synthetic
