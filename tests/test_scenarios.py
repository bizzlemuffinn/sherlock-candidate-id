"""Scenario-driven tests. Each scenario JSON encodes its expected outcome;
the engine must identify the right participant (or correctly abstain)
WITHOUT the LLM reasoner — deterministic signals alone must pass.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sherlock_id import CandidateIdentifier, ScenarioStream  # noqa: E402
from sherlock_id.extractors.name_matcher import name_similarity  # noqa: E402

SCENARIOS = sorted((Path(__file__).resolve().parents[1] / "scenarios").glob("*.json"))


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_scenario(path):
    stream = ScenarioStream(path)
    engine = CandidateIdentifier(stream.context, use_llm=False)
    verdict = engine.run(stream.events)

    if stream.meta["expect_uncertain"]:
        assert verdict.status == "uncertain", (
            f"should abstain, got {verdict.candidate_id} "
            f"@ {verdict.confidence:.2f}"
        )
    else:
        assert verdict.status == "confident", (
            f"should be confident, top posterior {verdict.confidence:.2f}"
        )
        assert verdict.candidate_id == stream.meta["expected_candidate"]
        assert verdict.explanation, "verdict must carry an explanation"

    flagged = {e.signal for e in engine.evidence_log if e.signal.endswith("_flag")}
    for expected_flag in stream.meta["expect_fraud_flags"]:
        assert expected_flag in flagged, f"missing fraud flag {expected_flag}"


def test_confidence_monotonic_in_happy_path():
    """Confidence in the true candidate should broadly rise as evidence
    accumulates (allowing small local dips from replaced signals)."""
    stream = ScenarioStream(SCENARIOS[0])  # 01_happy_path
    engine = CandidateIdentifier(stream.context, use_llm=False)
    engine.run(stream.events)
    target = stream.meta["expected_candidate"]
    series = [
        v.posteriors.get(target, 0.0)
        for v in engine.timeline
        if target in v.posteriors
    ]
    assert series[-1] > 0.8
    assert series[-1] >= series[0]


def test_rename_does_not_reset_identity():
    """Evidence accrued before a rename must survive it (identity is the
    participant ID, not the display name)."""
    stream = ScenarioStream(
        Path(__file__).resolve().parents[1]
        / "scenarios" / "03_rename_and_observers.json"
    )
    engine = CandidateIdentifier(stream.context, use_llm=False)
    verdict = engine.run(stream.events)
    assert verdict.candidate_id == "p_sam"
    sam = engine.participants["p_sam"]
    assert sam.rename_count == 1
    assert sam.name_history == ["SP", "Sam"]


def test_name_similarity():
    assert name_similarity("Bill Gates", "William Gates") > 0.85
    assert name_similarity("sagar.singh", "Sagar Singh") > 0.85
    assert name_similarity("MacBook Pro", "Rahul Verma") < 0.4
    assert name_similarity("Sam", "Samuel Peterson") > 0.55


def test_proxy_speaker_flags_do_not_break_identification():
    """Scenario 06: a second voice appears in the candidate's stream and
    the enrollment voice match collapses. Identification must survive
    (other signals still dominate) while fraud flags fire with zero
    weight on the posterior."""
    stream = ScenarioStream(
        Path(__file__).resolve().parents[1]
        / "scenarios" / "06_proxy_speaker.json"
    )
    engine = CandidateIdentifier(stream.context, use_llm=False)
    verdict = engine.run(stream.events)

    assert verdict.candidate_id == "p_karan"
    flags = [e for e in engine.evidence_log if e.signal == "multi_speaker_flag"]
    assert flags, "multi-speaker fraud flag must fire"
    assert all(f.log_lr == 0.0 for f in flags), "flags carry no posterior weight"
    # voice_match evidence must have flipped negative after the swap
    voice = engine.fusion.evidence_for("p_karan")
    vm = next(e for e in voice if e.signal == "voice_match")
    assert vm.log_lr < 0, "latest enrollment similarity should be negative"


def test_leaving_participant_removed_from_hypothesis_space():
    from sherlock_id.models import EventType, InterviewContext, MeetingEvent

    ctx = InterviewContext(candidate_name="A B", interviewer_names=["C D"])
    engine = CandidateIdentifier(ctx, use_llm=False)
    engine.process(MeetingEvent(0, EventType.JOIN, "p1", {"display_name": "A B"}))
    engine.process(MeetingEvent(1, EventType.JOIN, "p2", {"display_name": "C D"}))
    verdict = engine.process(MeetingEvent(2, EventType.LEAVE, "p1", {}))
    assert "p1" not in verdict.posteriors
