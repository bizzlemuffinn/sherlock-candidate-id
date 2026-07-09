"""Orchestrator: event stream -> participant state -> extractors -> fusion.

Identity is keyed on the platform participant ID, never the display
name, so a mid-meeting rename cannot transfer accumulated evidence to a
different person.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from .extractors import (
    BehaviorTracker,
    DiarizationAnalyzer,
    Extractor,
    FaceVerifier,
    LLMReasoner,
    NameMatcher,
    TranscriptAnalyzer,
    VoiceVerifier,
)
from .fusion import FusionEngine
from .models import (
    Evidence,
    EventType,
    InterviewContext,
    MeetingEvent,
    Participant,
    Verdict,
)


class CandidateIdentifier:
    def __init__(
        self,
        context: InterviewContext,
        confidence_threshold: float = 0.60,
        use_llm: bool = True,
        extractors: Optional[list[Extractor]] = None,
    ):
        self.context = context
        self.participants: dict[str, Participant] = {}
        self.fusion = FusionEngine(confidence_threshold)
        self.timeline: list[Verdict] = []
        self.evidence_log: list[Evidence] = []

        if extractors is not None:
            self.extractors = extractors
        else:
            self.extractors = [
                NameMatcher(context),
                BehaviorTracker(context),
                TranscriptAnalyzer(context),
                DiarizationAnalyzer(context),
                VoiceVerifier(context),
                FaceVerifier(context),
            ]
            if use_llm:
                llm = LLMReasoner(context)
                if llm.enabled:
                    self.extractors.append(llm)

    @property
    def llm_enabled(self) -> bool:
        return any(isinstance(e, LLMReasoner) for e in self.extractors)

    # ------------------------------------------------------------------

    def _apply_state(self, event: MeetingEvent) -> None:
        pid = event.participant_id
        if event.type == EventType.JOIN:
            name = event.data.get("display_name", pid)
            self.participants[pid] = Participant(
                id=pid, display_name=name, joined_at=event.ts,
                name_history=[name],
            )
            return

        p = self.participants.get(pid)
        if p is None:
            return
        if event.type == EventType.LEAVE:
            p.left_at = event.ts
        elif event.type == EventType.RENAME:
            p.display_name = event.data.get("new_name", p.display_name)
            p.rename_count += 1
            p.name_history.append(p.display_name)
        elif event.type == EventType.CAMERA_ON:
            p.camera_on = True
        elif event.type == EventType.CAMERA_OFF:
            p.camera_on = False
        elif event.type == EventType.SCREEN_SHARE_START:
            p.is_sharing = True
        elif event.type == EventType.SCREEN_SHARE_STOP:
            p.is_sharing = False
        elif event.type == EventType.UTTERANCE:
            p.talk_seconds += float(event.data.get("duration", 0.0))
            p.utterance_count += 1

    # ------------------------------------------------------------------

    def process(self, event: MeetingEvent) -> Verdict:
        self._apply_state(event)

        for extractor in self.extractors:
            for ev in extractor.on_event(event, self.participants):
                self.fusion.add(ev)
                self.evidence_log.append(ev)

        present = [p.id for p in self.participants.values() if p.present]
        verdict = self.fusion.verdict(present, event.ts)
        self.timeline.append(verdict)
        return verdict

    def run(self, events: list[MeetingEvent]) -> Verdict:
        verdict = None
        for event in sorted(events, key=lambda e: e.ts):
            verdict = self.process(event)
        return verdict

    # ------------------------------------------------------------------

    def explain(self) -> dict:
        """Full explainability dump: verdict + per-participant evidence."""
        present = [p.id for p in self.participants.values() if p.present]
        verdict = self.fusion.verdict(
            present, self.timeline[-1].ts if self.timeline else 0.0
        )
        return {
            "verdict": asdict(verdict),
            # zero-weight *_flag evidence: no identification influence,
            # surfaced for downstream fraud detectors
            "fraud_flags": [
                {"ts": e.ts, "participant_id": e.participant_id,
                 "signal": e.signal, "reason": e.reason}
                for e in self.evidence_log if e.signal.endswith("_flag")
            ],
            "participants": {
                pid: {
                    "display_name": self.participants[pid].display_name,
                    "posterior": verdict.posteriors.get(pid, 0.0),
                    "evidence": [
                        {"signal": e.signal, "log_lr": round(e.log_lr, 2),
                         "reason": e.reason}
                        for e in self.fusion.evidence_for(pid)
                    ],
                }
                for pid in present
            },
        }
