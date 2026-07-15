"""Streaming wrapper around MuseTalk V1.5 realtime inference. RUN ON SPARK —
installed into the container as musetalk/pipeline.py by docker/musetalk.Dockerfile.

Upstream ships batch scripts (scripts/realtime_inference.py); this adapts that
loop into MuseTalkStreamPipeline.stream_frames(audio_chunk) -> iterator of JPEG
bytes for serving/video/server.py. Aligned with the V1.5 repo layout — if
upstream refactors module paths, fix the imports here (this file is the whole
integration surface).
"""

from __future__ import annotations

import numpy as np
import torch

# upstream MuseTalk modules (repo cloned to /opt/musetalk, on PYTHONPATH)
from musetalk.utils.blending import get_image_blending
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.utils import datagen, load_all_model
from musetalk.whisper.audio2feature import Audio2Feature


class MuseTalkStreamPipeline:
    """Precomputes identity latents once, then lip-syncs incoming 16 kHz PCM
    chunks frame-by-frame (sentence-boundary buffering happens upstream in the
    orchestrator; here we render as soon as whisper features exist)."""

    def __init__(self, model_dir: str, identity_video: str, resolution: int = 256):
        self.resolution = resolution
        self.audio_processor: Audio2Feature
        self.audio_processor, self.vae, self.unet, self.pe = load_all_model(model_dir)
        self.timesteps = torch.tensor([0], device="cuda")

        # identity prep (addendum "Feed your face"): frames, bboxes, VAE latents
        frames = read_imgs(identity_video)
        self.coord_list, self.frame_list = get_landmark_and_bbox(frames)
        self.input_latent_list = [
            self.vae.get_latents_for_unet(frame) for frame in self.frame_list
        ]
        self._cycle = 0

    @torch.no_grad()
    def stream_frames(self, pcm16_chunk: bytes, jpeg_quality: int = 85):
        import cv2

        audio = np.frombuffer(pcm16_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        whisper_chunks = self.audio_processor.audio2feat_chunks(audio, fps=25)

        for whisper_batch, latent_batch in datagen(
            whisper_chunks, self.input_latent_list, batch_size=1, start=self._cycle
        ):
            audio_feature = self.pe(whisper_batch.to("cuda"))
            pred_latents = self.unet.model(
                latent_batch.to("cuda"), self.timesteps, encoder_hidden_states=audio_feature
            ).sample
            recon = self.vae.decode_latents(pred_latents)
            idx = self._cycle % len(self.frame_list)
            frame = get_image_blending(
                self.frame_list[idx], recon[0], self.coord_list[idx]
            )
            if self.resolution != frame.shape[0]:
                frame = cv2.resize(frame, (self.resolution, self.resolution))
            ok, jpeg = cv2.imencode(".jpg", frame,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if ok:
                yield jpeg.tobytes()
            self._cycle += 1
