"""Production-path stubs: voice and face verification.

These define the interface a production deployment would plug real
models into. They emit no evidence in the demo (no media streams in the
simulator), but the engine already fuses them — dropping in a real
implementation requires zero changes elsewhere.

Production plan (see README):
- VoiceVerifier: ECAPA-TDNN embeddings (SpeechBrain) per participant
  audio stream, cosine similarity vs an enrollment sample from the
  phone screen. Also detects mid-interview speaker swap via embedding
  drift.
- FaceVerifier: InsightFace/ArcFace embedding of the webcam face vs a
  reference photo (ATS profile / ID document), plus TalkNet active
  speaker detection to bind the speaking voice to the on-camera face.
"""

from __future__ import annotations

from ..models import Evidence, MeetingEvent, Participant
from .base import Extractor


class VoiceVerifier(Extractor):
    name = "voice_match"

    def on_event(
        self, event: MeetingEvent, participants: dict[str, Participant]
    ) -> list[Evidence]:
        # Real impl: consume event.data["audio_chunk"], maintain rolling
        # embedding per participant, emit Evidence("voice_match", pid,
        # log_lr_from_cosine_sim, ...). Simulator carries no audio.
        return []


class FaceVerifier(Extractor):
    name = "face_match"

    def on_event(
        self, event: MeetingEvent, participants: dict[str, Participant]
    ) -> list[Evidence]:
        # Real impl: consume event.data["video_frame"] on a sampled cadence,
        # ArcFace cosine vs reference photo when available.
        return []
