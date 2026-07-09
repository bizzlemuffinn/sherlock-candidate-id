"""LLM conversational-role reasoner (optional, needs ANTHROPIC_API_KEY).

Every WINDOW utterances, sends the recent speaker-attributed transcript
plus participant metadata to Claude and asks it to assign roles with a
per-participant candidate probability and stated reasoning. The output
is converted to bounded log-likelihood ratios and fused like any other
signal — the LLM never gets to decide alone.

Why an LLM here at all: regexes catch the common interview phrasing;
the LLM handles the long tail — cross-cultural nicknames and
transliterations, indirect self-references, code-switching, sarcasm,
multi-interviewer dynamics — with a natural-language explanation for
free.
"""

from __future__ import annotations

import json
import os

from ..models import Evidence, EventType, InterviewContext, MeetingEvent, Participant
from .base import Extractor

WINDOW = 8          # utterances per LLM call
MAX_LINES = 30      # transcript lines sent per call
MODEL = "claude-opus-4-8"

SCHEMA = {
    "type": "object",
    "properties": {
        "assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "participant_id": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": ["candidate", "interviewer", "observer", "unknown"],
                    },
                    "candidate_probability": {"type": "number"},
                    "reasoning": {"type": "string"},
                },
                "required": [
                    "participant_id", "role",
                    "candidate_probability", "reasoning",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["assessments"],
    "additionalProperties": False,
}

SYSTEM = """You identify which participant in a live job-interview meeting \
is the CANDIDATE (the person being interviewed). Display names are \
unreliable: candidates join as "MacBook Pro", use nicknames, or get \
mislabeled. Rely on conversational roles: interviewers ask questions and \
steer; the candidate answers at length, describes their own experience, \
and introduces themself by their real name. Observers stay silent. \
Assess every participant, give a calibrated candidate_probability in \
[0,1], and one concise sentence of reasoning each."""


class LLMReasoner(Extractor):
    name = "llm_role"

    def __init__(self, context: InterviewContext):
        super().__init__(context)
        self._transcript: list[str] = []
        self._since_last_call = 0
        self._client = None
        self.enabled = bool(os.environ.get("ANTHROPIC_API_KEY"))
        if self.enabled:
            try:
                import anthropic

                self._client = anthropic.Anthropic()
            except ImportError:
                self.enabled = False

    def on_event(
        self, event: MeetingEvent, participants: dict[str, Participant]
    ) -> list[Evidence]:
        if event.type != EventType.UTTERANCE or not self.enabled:
            return []

        p = participants[event.participant_id]
        self._transcript.append(
            f'[{p.id} | "{p.display_name}"]: {event.data.get("text", "")}'
        )
        self._since_last_call += 1
        if self._since_last_call < WINDOW:
            return []
        self._since_last_call = 0
        return self._call_llm(event.ts, participants)

    def _call_llm(
        self, ts: float, participants: dict[str, Participant]
    ) -> list[Evidence]:
        roster = "\n".join(
            f'- {p.id}: display_name="{p.display_name}", '
            f"camera={'on' if p.camera_on else 'off'}, "
            f"talk_seconds={p.talk_seconds:.0f}, "
            f"screen_sharing={p.is_sharing}, renames={p.rename_count}"
            for p in participants.values()
            if p.present
        )
        prompt = f"""Interview metadata (from ATS/calendar — may be wrong):
- expected candidate name: {self.context.candidate_name}
- candidate email: {self.context.candidate_email or "unknown"}
- interviewers on the invite: {", ".join(self.context.interviewer_names) or "unknown"}

Participants currently in the meeting:
{roster}

Recent transcript:
{chr(10).join(self._transcript[-MAX_LINES:])}

Assess each present participant."""

        try:
            response = self._client.messages.create(
                model=MODEL,
                max_tokens=2000,
                thinking={"type": "adaptive"},
                system=SYSTEM,
                output_config={
                    "format": {"type": "json_schema", "schema": SCHEMA}
                },
                messages=[{"role": "user", "content": prompt}],
            )
            if response.stop_reason == "refusal":
                return []
            text = next(
                b.text for b in response.content if b.type == "text"
            )
            data = json.loads(text)
        except Exception:
            # LLM is a bonus signal — degrade gracefully, never crash the engine
            return []

        out: list[Evidence] = []
        for a in data.get("assessments", []):
            pid = a["participant_id"]
            if pid not in participants:
                continue
            prob = min(0.99, max(0.01, float(a["candidate_probability"])))
            # probability -> bounded log-odds contribution, scaled to keep
            # the LLM a strong-but-not-dominant voice in the fusion
            import math

            log_lr = 0.6 * math.log(prob / (1 - prob))
            out.append(Evidence(
                self.name, pid, log_lr,
                f"LLM: {a['role']} (p={prob:.2f}) — {a['reasoning']}",
                ts,
            ))
        return out
