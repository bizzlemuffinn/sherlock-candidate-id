"""Core data models for the candidate identification engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EventType(str, Enum):
    JOIN = "join"
    LEAVE = "leave"
    RENAME = "rename"
    CAMERA_ON = "camera_on"
    CAMERA_OFF = "camera_off"
    SCREEN_SHARE_START = "screen_share_start"
    SCREEN_SHARE_STOP = "screen_share_stop"
    UTTERANCE = "utterance"


@dataclass
class MeetingEvent:
    """A single event emitted by the meeting platform (or simulator)."""

    ts: float  # seconds from meeting start
    type: EventType
    participant_id: str
    data: dict[str, Any] = field(default_factory=dict)
    # UTTERANCE: {"text": str, "duration": float}
    # RENAME:    {"new_name": str}
    # JOIN:      {"display_name": str}


@dataclass
class Participant:
    """Live state for one meeting participant."""

    id: str
    display_name: str
    joined_at: float = 0.0
    left_at: Optional[float] = None
    camera_on: bool = False
    is_sharing: bool = False
    talk_seconds: float = 0.0
    utterance_count: int = 0
    questions_asked: int = 0
    answers_given: int = 0
    rename_count: int = 0
    name_history: list[str] = field(default_factory=list)

    @property
    def present(self) -> bool:
        return self.left_at is None


@dataclass
class InterviewContext:
    """External metadata known before the meeting starts."""

    candidate_name: str
    candidate_email: str = ""
    interviewer_names: list[str] = field(default_factory=list)
    scheduled_start: float = 0.0  # seconds offset; 0 == meeting start
    meeting_title: str = ""


@dataclass
class Evidence:
    """One piece of evidence about one participant.

    log_lr is the log-likelihood ratio:
        log( P(signal | participant IS candidate) / P(signal | is NOT) )
    Positive supports candidacy, negative opposes it.

    Evidence is keyed by (signal, participant_id): re-emitting the same
    signal REPLACES the previous value instead of double counting, so
    continuous signals (talk share, camera uptime) can be refreshed on
    every event without inflating the posterior.
    """

    signal: str
    participant_id: str
    log_lr: float
    reason: str
    ts: float = 0.0


@dataclass
class Verdict:
    """Engine output after any event."""

    ts: float
    candidate_id: Optional[str]  # None => uncertain
    confidence: float  # posterior of top participant
    posteriors: dict[str, float]
    status: str  # "confident" | "uncertain" | "no_participants"
    explanation: list[str]
