"""Diarization-derived evidence (simulated payloads in the demo).

In production this extractor sits behind a real diarization pipeline
(pyannote segmentation + ECAPA-TDNN embeddings) running on the
per-participant audio streams. The pipeline's per-utterance outputs are
attached to UTTERANCE events as `data["diarization"]`:

    {
      "num_speakers": 1,           # distinct voice clusters in THIS
                                   # participant's stream so far
      "attribution_match": true,   # independent diarization agrees with
                                   # the platform's speaker attribution
      "enrollment_similarity": 0.86  # cosine sim of live voice embedding
                                     # vs candidate's enrollment sample
                                     # (phone screen); absent if no
                                     # enrollment audio exists
    }

The simulator carries these payloads verbatim, so the fusion behaviour
is demonstrated end-to-end without shipping torch/pyannote. Swapping in
the real pipeline changes only who produces the payload.

Three signals:
- enrollment voice match  -> identification evidence (voice_match)
- attribution mismatch    -> negative evidence + trust degradation
- >1 speaker in one participant's stream -> zero-weight FRAUD FLAG
  (proxy answering off-camera); identification is unaffected but the
  flag is surfaced to downstream fraud detectors, like rename_flag.
"""

from __future__ import annotations

from ..models import Evidence, EventType, MeetingEvent, Participant
from .base import Extractor


class DiarizationAnalyzer(Extractor):
    name = "diarization"

    def on_event(
        self, event: MeetingEvent, participants: dict[str, Participant]
    ) -> list[Evidence]:
        if event.type != EventType.UTTERANCE:
            return []
        d = event.data.get("diarization")
        if not d:
            return []

        pid = event.participant_id
        out: list[Evidence] = []

        sim = d.get("enrollment_similarity")
        if sim is not None:
            if sim >= 0.75:
                out.append(Evidence(
                    "voice_match", pid, 2.0,
                    f"voice matches candidate enrollment sample (cos={sim:.2f})",
                    event.ts,
                ))
            elif sim >= 0.55:
                out.append(Evidence(
                    "voice_match", pid, 0.5,
                    f"voice weakly matches enrollment sample (cos={sim:.2f})",
                    event.ts,
                ))
            else:
                out.append(Evidence(
                    "voice_match", pid, -1.0,
                    f"voice diverged from enrollment sample (cos={sim:.2f}) "
                    "— possible speaker swap",
                    event.ts,
                ))

        if d.get("attribution_match") is False:
            out.append(Evidence(
                "diarization_consistency", pid, -1.5,
                "acoustic diarization disagrees with platform speaker "
                "attribution — transcript for this participant is untrusted",
                event.ts,
            ))

        if int(d.get("num_speakers", 1)) > 1:
            out.append(Evidence(
                "multi_speaker_flag", pid, 0.0,
                f"{d['num_speakers']} distinct voices detected in this "
                "participant's audio stream — possible proxy/whisperer, "
                "flagged for fraud review",
                event.ts,
            ))
        return out
