"""Meeting event stream abstraction.

The engine consumes `MeetingEvent`s and does not care where they come
from. `ScenarioStream` replays a JSON scenario file (used for the demo
and tests). A production `RecallStream` / `VexaStream` would subscribe
to a meeting-bot webhook/websocket and translate platform payloads into
the same `MeetingEvent` shape — the only integration point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .models import EventType, InterviewContext, MeetingEvent


def load_scenario(path: str | Path) -> tuple[InterviewContext, list[MeetingEvent], dict]:
    raw = json.loads(Path(path).read_text())
    ctx = InterviewContext(
        candidate_name=raw["context"]["candidate_name"],
        candidate_email=raw["context"].get("candidate_email", ""),
        interviewer_names=raw["context"].get("interviewer_names", []),
        meeting_title=raw["context"].get("meeting_title", ""),
    )
    events = [
        MeetingEvent(
            ts=float(e["ts"]),
            type=EventType(e["type"]),
            participant_id=e["participant_id"],
            data=e.get("data", {}),
        )
        for e in raw["events"]
    ]
    meta = {
        "name": raw.get("name", Path(path).stem),
        "description": raw.get("description", ""),
        "expected_candidate": raw.get("expected_candidate"),
        "expect_uncertain": raw.get("expect_uncertain", False),
        "expect_fraud_flags": raw.get("expect_fraud_flags", []),
    }
    return ctx, sorted(events, key=lambda e: e.ts), meta


class ScenarioStream:
    """Iterates a scenario's events in timestamp order."""

    def __init__(self, path: str | Path):
        self.context, self.events, self.meta = load_scenario(path)

    def __iter__(self) -> Iterator[MeetingEvent]:
        return iter(self.events)
