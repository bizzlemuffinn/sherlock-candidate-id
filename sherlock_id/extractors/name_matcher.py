"""Name evidence: display name vs candidate name / email / interviewer names.

Runs on JOIN and RENAME. Key properties:
- A non-name display ("MacBook Pro", "iPhone", "User") contributes ~zero
  evidence, NOT negative evidence — the candidate often joins that way.
- Matching an interviewer name from the calendar is strong negative
  evidence (role elimination is often how the candidate gets found).
- A mid-meeting rename is itself a weak fraud-relevant signal.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from ..models import Evidence, MeetingEvent, EventType, Participant
from .base import Extractor

NICKNAMES = {
    "bill": "william", "will": "william", "bob": "robert", "rob": "robert",
    "mike": "michael", "dave": "david", "jim": "james", "jimmy": "james",
    "liz": "elizabeth", "beth": "elizabeth", "kate": "katherine",
    "alex": "alexander", "sam": "samuel", "tom": "thomas", "tony": "anthony",
    "chris": "christopher", "nick": "nicholas", "sags": "sagar",
    "raj": "rajesh", "monty": "montgomery", "andy": "andrew",
}

DEVICE_NAMES = re.compile(
    r"(macbook|iphone|ipad|galaxy|pixel|user|guest|admin|desktop|laptop|"
    r"phone|room|meeting|unknown|anonymous)\b",
    re.I,
)


def _canon(token: str) -> str:
    t = re.sub(r"[^a-z]", "", token.lower())
    return NICKNAMES.get(t, t)


def _tokens(name: str) -> list[str]:
    return [_canon(t) for t in re.split(r"[\s._\-@]+", name) if _canon(t)]


def name_similarity(display: str, reference: str) -> float:
    """0..1 similarity robust to token order, nicknames, partial names."""
    a, b = _tokens(display), _tokens(reference)
    if not a or not b:
        return 0.0
    # token-level best matching
    scores = []
    for ta in a:
        best = max(
            (SequenceMatcher(None, ta, tb).ratio() for tb in b), default=0.0
        )
        scores.append(best)
    token_score = sum(scores) / len(scores)
    whole = SequenceMatcher(None, " ".join(a), " ".join(b)).ratio()
    return max(token_score, whole)


class NameMatcher(Extractor):
    name = "name_match"

    def on_event(
        self, event: MeetingEvent, participants: dict[str, Participant]
    ) -> list[Evidence]:
        if event.type not in (EventType.JOIN, EventType.RENAME):
            return []

        p = participants[event.participant_id]
        display = p.display_name
        out: list[Evidence] = []

        # candidate name / email local-part
        refs = [self.context.candidate_name]
        if self.context.candidate_email:
            refs.append(self.context.candidate_email.split("@")[0])
        cand_sim = max(name_similarity(display, r) for r in refs)

        # interviewer names from calendar
        ivw_sim = max(
            (name_similarity(display, n) for n in self.context.interviewer_names),
            default=0.0,
        )

        if DEVICE_NAMES.search(display) and cand_sim < 0.5:
            out.append(Evidence(
                self.name, p.id, 0.0,
                f'display name "{display}" is a device/generic name — no name signal',
                event.ts,
            ))
        elif ivw_sim > 0.75 and ivw_sim > cand_sim:
            out.append(Evidence(
                self.name, p.id, -2.5,
                f'"{display}" matches interviewer from calendar invite '
                f"(sim={ivw_sim:.2f})",
                event.ts,
            ))
        elif cand_sim > 0.85:
            out.append(Evidence(
                self.name, p.id, 2.2,
                f'"{display}" strongly matches candidate '
                f'"{self.context.candidate_name}" (sim={cand_sim:.2f})',
                event.ts,
            ))
        elif cand_sim > 0.55:
            out.append(Evidence(
                self.name, p.id, 1.0,
                f'"{display}" partially matches candidate '
                f'"{self.context.candidate_name}" (sim={cand_sim:.2f})',
                event.ts,
            ))
        else:
            out.append(Evidence(
                self.name, p.id, -0.3,
                f'"{display}" matches neither candidate nor interviewers',
                event.ts,
            ))

        if event.type == EventType.RENAME:
            out.append(Evidence(
                "rename_flag", p.id, 0.0,
                f"renamed mid-meeting ({p.rename_count}x): "
                f"{' -> '.join(p.name_history[-2:])} — flagged for fraud review",
                event.ts,
            ))
        return out
