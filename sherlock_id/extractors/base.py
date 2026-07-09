"""Extractor interface.

An extractor observes meeting events and emits Evidence. Extractors are
independent and stateless with respect to each other; the fusion engine
is the only place scores combine. Adding a new signal (voice embedding,
face verification, ...) means adding one extractor — nothing else changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Evidence, InterviewContext, MeetingEvent, Participant


class Extractor(ABC):
    name: str = "base"

    def __init__(self, context: InterviewContext):
        self.context = context

    @abstractmethod
    def on_event(
        self,
        event: MeetingEvent,
        participants: dict[str, Participant],
    ) -> list[Evidence]:
        """Return zero or more Evidence items in response to an event."""
        raise NotImplementedError
