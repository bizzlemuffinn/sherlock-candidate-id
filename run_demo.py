#!/usr/bin/env python3
"""CLI demo: replay a scenario through the engine and print the
confidence timeline, final verdict, and full explanation.

Usage:
    python run_demo.py scenarios/02_macbook_pro.json
    python run_demo.py --all
    python run_demo.py --all --no-llm     # skip LLM even if key present
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sherlock_id import CandidateIdentifier, ScenarioStream

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[1m", "\033[0m",
)


def bar(p: float, width: int = 24) -> str:
    filled = int(round(p * width))
    return "█" * filled + "░" * (width - filled)


def run(path: Path, use_llm: bool) -> bool:
    stream = ScenarioStream(path)
    ctx, meta = stream.context, stream.meta
    engine = CandidateIdentifier(ctx, use_llm=use_llm)

    print(f"\n{BOLD}=== {meta['name']} ==={RESET}")
    print(f"{DIM}{meta['description']}{RESET}")
    print(f"expected candidate (ATS): {ctx.candidate_name} "
          f"| interviewers: {', '.join(ctx.interviewer_names)}"
          f" | LLM reasoner: {'on' if engine.llm_enabled else 'off'}\n")

    for event in stream:
        verdict = engine.process(event)
        p = engine.participants[event.participant_id]
        label = f"{event.type.value:<18} {p.display_name:<16}"
        top = (
            f"{engine.participants[max(verdict.posteriors, key=verdict.posteriors.get)].display_name} "
            f"{max(verdict.posteriors.values()):.2f}"
            if verdict.posteriors else "-"
        )
        status_color = GREEN if verdict.status == "confident" else YELLOW
        print(f"  t={event.ts:>5.0f}s  {label} -> "
              f"{status_color}{verdict.status:<9}{RESET} top: {top}")

    result = engine.explain()
    verdict = result["verdict"]

    print(f"\n{BOLD}Final posteriors:{RESET}")
    for pid, info in sorted(
        result["participants"].items(),
        key=lambda kv: kv[1]["posterior"], reverse=True,
    ):
        print(f"  {info['display_name']:<18} {bar(info['posterior'])} "
              f"{info['posterior']:.2f}")

    if verdict["status"] == "confident":
        name = engine.participants[verdict["candidate_id"]].display_name
        print(f"\n{BOLD}Verdict:{RESET} {GREEN}CANDIDATE = {name} "
              f"({verdict['candidate_id']}) @ {verdict['confidence']:.0%}{RESET}")
    else:
        print(f"\n{BOLD}Verdict:{RESET} {YELLOW}UNCERTAIN "
              f"(top {verdict['confidence']:.0%} < threshold){RESET}")

    print(f"\n{BOLD}Why:{RESET}")
    top_pid = max(verdict["posteriors"], key=verdict["posteriors"].get)
    for e in result["participants"][top_pid]["evidence"][:8]:
        sign = "+" if e["log_lr"] >= 0 else ""
        print(f"  [{e['signal']:<12}] {sign}{e['log_lr']:<6} {e['reason']}")

    if result["fraud_flags"]:
        print(f"\n{BOLD}{RED}Fraud flags (for downstream detectors):{RESET}")
        for f in result["fraud_flags"]:
            who = engine.participants[f["participant_id"]].display_name
            print(f"  {RED}⚑{RESET} t={f['ts']:.0f}s {who} [{f['signal']}] {f['reason']}")

    # pass/fail against scenario expectation
    if meta["expect_uncertain"]:
        ok = verdict["status"] == "uncertain"
    else:
        ok = verdict["candidate_id"] == meta["expected_candidate"]
    flagged = {f["signal"] for f in result["fraud_flags"]}
    ok = ok and all(s in flagged for s in meta["expect_fraud_flags"])
    print(f"\n  scenario check: "
          f"{GREEN + 'PASS' if ok else RED + 'FAIL'}{RESET}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", help="path to scenario JSON")
    ap.add_argument("--all", action="store_true", help="run every scenario")
    ap.add_argument("--no-llm", action="store_true",
                    help="disable the LLM reasoner even if a key is set")
    args = ap.parse_args()

    use_llm = not args.no_llm
    if args.all:
        paths = sorted(Path("scenarios").glob("*.json"))
    elif args.scenario:
        paths = [Path(args.scenario)]
    else:
        ap.print_help()
        return 2

    results = [run(p, use_llm) for p in paths]
    print(f"\n{BOLD}{sum(results)}/{len(results)} scenarios passed{RESET}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
