"""Conversational-role evidence from the speaker-attributed transcript.

Deterministic heuristics (no API needed):
- question/answer structure: interviewers ask, candidates answer at length
- self-introduction: "I'm <name>" / "my name is <name>" checked against
  the candidate name — the single strongest transcript signal, because a
  display name can be wrong but people introduce themselves truthfully
- interviewer phrasing: "tell me about", "walk me through", "next question"
- candidate phrasing: first-person experience ("I worked at", "my role was")

The LLM reasoner (llm_reasoner.py) covers the long tail these regexes
miss; both feed the same fusion engine under different signal names.
"""

from __future__ import annotations

import re

from ..models import Evidence, EventType, MeetingEvent, Participant
from .base import Extractor
from .name_matcher import name_similarity

INTERVIEWER_PATTERNS = re.compile(
    r"(tell (me|us) about|walk (me|us) through|why did you|"
    r"what would you|how would you|can you (explain|describe|share)|"
    r"next question|let'?s move on|do you have any questions for|"
    r"thanks for joining|introduce yourself|your resume|your cv|"
    r"we are looking for|the role involves|any questions about the (role|team|company))",
    re.I,
)

CANDIDATE_PATTERNS = re.compile(
    r"(i worked (at|on|with)|my (role|responsibility|project|team) (was|is)|"
    r"in my (previous|current|last) (role|job|company)|i built|i designed|"
    r"i implemented|my experience (with|in)|i led|i was responsible)",
    re.I,
)

SELF_INTRO = re.compile(
    r"\b(?:i'?m|i am|my name is|myself|this is)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
    re.I,
)

MIN_ANSWER_SECONDS = 25.0


class TranscriptAnalyzer(Extractor):
    name = "transcript"

    def __init__(self, context):
        super().__init__(context)
        self._last_question_by: str | None = None

    def on_event(
        self, event: MeetingEvent, participants: dict[str, Participant]
    ) -> list[Evidence]:
        if event.type != EventType.UTTERANCE:
            return []

        text = event.data.get("text", "")
        duration = float(event.data.get("duration", 0.0))
        pid = event.participant_id
        p = participants[pid]
        out: list[Evidence] = []

        is_question = bool(INTERVIEWER_PATTERNS.search(text)) or (
            text.rstrip().endswith("?") and duration < 20
        )

        # self-introduction against the *real* candidate name
        m = SELF_INTRO.search(text)
        if m:
            claimed = m.group(1)
            sim = name_similarity(claimed, self.context.candidate_name)
            if sim > 0.75:
                out.append(Evidence(
                    "self_intro", pid, 2.5,
                    f'self-introduced as "{claimed}" — matches candidate '
                    f'"{self.context.candidate_name}" (sim={sim:.2f})',
                    event.ts,
                ))
            elif sim > 0.0 and any(
                name_similarity(claimed, n) > 0.75
                for n in self.context.interviewer_names
            ):
                out.append(Evidence(
                    "self_intro", pid, -2.0,
                    f'self-introduced as "{claimed}" — matches an interviewer',
                    event.ts,
                ))

        if is_question:
            p.questions_asked += 1
            self._last_question_by = pid
            ratio = p.questions_asked / max(1, p.utterance_count)
            if p.questions_asked >= 2 and ratio > 0.4:
                out.append(Evidence(
                    "qa_role", pid, -1.2,
                    f"asks questions ({p.questions_asked} so far) — interviewer pattern",
                    event.ts,
                ))
        else:
            answered_after_q = (
                self._last_question_by is not None
                and self._last_question_by != pid
            )
            long_answer = duration >= MIN_ANSWER_SECONDS
            first_person = bool(CANDIDATE_PATTERNS.search(text))
            if answered_after_q and (long_answer or first_person):
                p.answers_given += 1
                self._last_question_by = None
                strength = min(1.5, 0.5 + 0.25 * p.answers_given)
                why = []
                if long_answer:
                    why.append(f"{duration:.0f}s answer following a question")
                if first_person:
                    why.append("first-person experience language")
                out.append(Evidence(
                    "qa_role", pid, strength,
                    f"answers questions ({p.answers_given} so far): "
                    + ", ".join(why),
                    event.ts,
                ))
        return out
