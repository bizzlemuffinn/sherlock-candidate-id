"""Bayesian evidence fusion.

Model: exactly one present participant is the candidate (categorical
hypothesis space, uniform prior over present participants). Each
extractor emits Evidence with a log-likelihood ratio for a participant.

    posterior(p)  ∝  prior(p) * exp( Σ log_lr(signal, p) )

Evidence is stored per (signal, participant) key, so refreshed signals
replace their previous value instead of compounding. This keeps
continuous signals (talk share, camera uptime) honest.

If the top posterior falls below `confidence_threshold`, the engine
reports "uncertain" rather than guessing — a wrong candidate ID would
poison every downstream fraud detector, so abstaining is the safer
failure mode.
"""

from __future__ import annotations

import math
from typing import Optional

from .models import Evidence, Verdict

# Cap per-signal contribution so no single extractor can saturate the
# posterior on its own (bonus: "multiple weak signals" by construction).
MAX_ABS_LOG_LR = 3.0


class FusionEngine:
    def __init__(self, confidence_threshold: float = 0.60):
        self.confidence_threshold = confidence_threshold
        # (signal, participant_id) -> Evidence
        self._evidence: dict[tuple[str, str], Evidence] = {}

    def add(self, ev: Evidence) -> None:
        ev.log_lr = max(-MAX_ABS_LOG_LR, min(MAX_ABS_LOG_LR, ev.log_lr))
        self._evidence[(ev.signal, ev.participant_id)] = ev

    def remove_participant(self, participant_id: str) -> None:
        self._evidence = {
            k: v for k, v in self._evidence.items() if k[1] != participant_id
        }

    def evidence_for(self, participant_id: str) -> list[Evidence]:
        return sorted(
            (v for k, v in self._evidence.items() if k[1] == participant_id),
            key=lambda e: abs(e.log_lr),
            reverse=True,
        )

    def posteriors(self, present_ids: list[str]) -> dict[str, float]:
        if not present_ids:
            return {}
        scores = {}
        for pid in present_ids:
            total = sum(
                v.log_lr for k, v in self._evidence.items() if k[1] == pid
            )
            scores[pid] = total
        # softmax over log scores == normalized posterior with uniform prior
        m = max(scores.values())
        exp = {pid: math.exp(s - m) for pid, s in scores.items()}
        z = sum(exp.values())
        return {pid: e / z for pid, e in exp.items()}

    def verdict(self, present_ids: list[str], ts: float) -> Verdict:
        post = self.posteriors(present_ids)
        if not post:
            return Verdict(ts, None, 0.0, {}, "no_participants", [])

        top_id, top_p = max(post.items(), key=lambda kv: kv[1])
        # A one-person hypothesis space is trivially 100% — that is not
        # evidence, so never report confident until >=2 participants exist.
        confident = top_p >= self.confidence_threshold and len(post) >= 2

        explanation = [
            f"[{ev.signal}] {ev.reason} ({'+' if ev.log_lr >= 0 else ''}{ev.log_lr:.2f})"
            for ev in self.evidence_for(top_id)[:6]
        ]
        return Verdict(
            ts=ts,
            candidate_id=top_id if confident else None,
            confidence=top_p,
            posteriors=post,
            status="confident" if confident else "uncertain",
            explanation=explanation,
        )
