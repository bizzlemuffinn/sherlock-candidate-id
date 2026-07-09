"""Behavioral evidence from meeting mechanics.

Signals (each deliberately weak — fusion does the heavy lifting):
- talk share: candidates typically speak 50-75% of an interview
- camera: candidate keeps camera on; silent camera-off participants
  are usually observers
- screen share: the participant sharing during a technical round is
  overwhelmingly the candidate
- 1:N speaking pattern: in a panel, several interviewers speak a little,
  one candidate speaks a lot

Continuous signals are re-emitted on every utterance and REPLACE their
previous value in the fusion store (keyed by signal+participant), so
they converge instead of compounding.
"""

from __future__ import annotations

from ..models import Evidence, EventType, MeetingEvent, Participant
from .base import Extractor

# Don't score talk share until the meeting has some substance.
MIN_TOTAL_TALK_SECONDS = 60.0


class BehaviorTracker(Extractor):
    name = "behavior"

    def on_event(
        self, event: MeetingEvent, participants: dict[str, Participant]
    ) -> list[Evidence]:
        out: list[Evidence] = []
        present = [p for p in participants.values() if p.present]

        if event.type == EventType.SCREEN_SHARE_START:
            out.append(Evidence(
                "screen_share", event.participant_id, 1.5,
                "sharing screen (candidate usually shares during coding/system-design)",
                event.ts,
            ))

        if event.type in (EventType.CAMERA_ON, EventType.CAMERA_OFF):
            p = participants[event.participant_id]
            out.append(Evidence(
                "camera", p.id,
                0.4 if p.camera_on else -0.8,
                "camera on" if p.camera_on
                else "camera off (observers usually stay camera-off)",
                event.ts,
            ))

        if event.type == EventType.UTTERANCE:
            total = sum(p.talk_seconds for p in present)
            if total >= MIN_TOTAL_TALK_SECONDS:
                for p in present:
                    share = p.talk_seconds / total
                    # piecewise mapping share -> log-LR
                    if share >= 0.45:
                        lr, why = 1.2, f"dominant talk share {share:.0%}"
                    elif share >= 0.30:
                        lr, why = 0.5, f"high talk share {share:.0%}"
                    elif share >= 0.10:
                        lr, why = -0.4, f"moderate talk share {share:.0%}"
                    elif p.utterance_count > 0:
                        lr, why = -0.8, f"low talk share {share:.0%}"
                    else:
                        lr, why = -1.2, "completely silent (observer pattern)"
                    out.append(Evidence("talk_share", p.id, lr, why, event.ts))
        return out
