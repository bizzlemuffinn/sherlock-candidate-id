#!/usr/bin/env python3
"""Live dashboard: replays a scenario through the engine over a
WebSocket so you can watch confidence evolve event-by-event.

    pip install fastapi uvicorn
    python server.py            # http://localhost:8000
    python server.py --speed 5  # 5x faster replay
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from sherlock_id import CandidateIdentifier, ScenarioStream

app = FastAPI(title="Sherlock Candidate Identifier")
SPEED = 10.0  # replay acceleration factor
SCENARIO_DIR = Path(__file__).parent / "scenarios"


@app.get("/")
async def index() -> HTMLResponse:
    html = (Path(__file__).parent / "dashboard" / "index.html").read_text()
    return HTMLResponse(html)


@app.get("/scenarios")
async def scenarios() -> list[dict]:
    out = []
    for p in sorted(SCENARIO_DIR.glob("*.json")):
        raw = json.loads(p.read_text())
        out.append({"file": p.name, "name": raw.get("name", p.stem),
                    "description": raw.get("description", "")})
    return out


@app.websocket("/ws/{scenario_file}")
async def replay(ws: WebSocket, scenario_file: str) -> None:
    await ws.accept()
    path = SCENARIO_DIR / scenario_file
    if not path.exists():
        await ws.close(code=4004)
        return

    stream = ScenarioStream(path)
    engine = CandidateIdentifier(stream.context)

    await ws.send_json({
        "type": "init",
        "context": asdict(stream.context),
        "meta": stream.meta,
        "llm_enabled": engine.llm_enabled,
    })

    try:
        prev_ts = 0.0
        for event in stream:
            await asyncio.sleep(max(0.0, (event.ts - prev_ts) / SPEED))
            prev_ts = event.ts

            n_before = len(engine.evidence_log)
            verdict = engine.process(event)
            new_evidence = engine.evidence_log[n_before:]

            await ws.send_json({
                "type": "update",
                "event": {
                    "ts": event.ts,
                    "kind": event.type.value,
                    "participant_id": event.participant_id,
                    "text": event.data.get("text", ""),
                },
                "participants": {
                    pid: {
                        "display_name": p.display_name,
                        "present": p.present,
                        "camera_on": p.camera_on,
                        "is_sharing": p.is_sharing,
                        "talk_seconds": round(p.talk_seconds, 1),
                    }
                    for pid, p in engine.participants.items()
                },
                "verdict": asdict(verdict),
                "evidence": [
                    {"signal": e.signal, "participant_id": e.participant_id,
                     "log_lr": round(e.log_lr, 2), "reason": e.reason,
                     "flag": e.signal.endswith("_flag")}
                    for e in new_evidence
                    if abs(e.log_lr) > 0.01 or e.signal.endswith("_flag")
                ],
            })

        await ws.send_json({"type": "done", "explain": engine.explain()})
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--speed", type=float, default=10.0,
                    help="replay acceleration (10 = 10x real time)")
    args = ap.parse_args()
    SPEED = args.speed
    uvicorn.run(app, host="127.0.0.1", port=args.port)
