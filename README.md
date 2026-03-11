# rppg_live_backend_service

FastAPI backend for the real-time rPPG live-proof-of-person track.

This service is the backend side of the canonical modular architecture used in the paper:

- thin mobile demo app on top of a reusable Mobile rPPG Acquisition SDK
- compact patch-summary transport over REST + WebSocket
- centralized Python inference using shared `rppg_core`
- replayable session artifacts for live-study analysis

## Responsibilities

- session lifecycle management
- packet validation, sequence tracking, and timestamp-aware ingestion
- shared preprocessing and centralized classical rPPG execution from compact RGB summaries
- provisional/stable result logic and final live decision logic
- replayable session artifacts:
  - `packet_trace.jsonl`
  - `quality_timeline.json`
  - `bpm_timeline.json`
  - `diagnostics.json`
  - `result.json`
- aggregate operational experiment outputs for the paper

Active-session state is buffered in memory. Replay artifacts are persisted on session finalization.

## API Surface

- `POST /v1/liveness/sessions`
- `POST /v1/liveness/sessions/{session_id}/stop`
- `WS /v1/liveness/sessions/{session_id}/stream?token=...`
- `GET /v1/liveness/sessions/{session_id}/result`
- `GET /v1/liveness/sessions/{session_id}/diagnostics`

## Setup

Create and populate the local virtual environment:

```bash
cd /Users/marting/Documents/Papers/rppg_live_backend_service
make install
```

Install replay/video extras into the same `.venv` if you need the replay pipeline:

```bash
cd /Users/marting/Documents/Papers/rppg_live_backend_service
make install-replay
```

## Run

Start the API from the project virtual environment:

```bash
cd /Users/marting/Documents/Papers/rppg_live_backend_service
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

Health check:

```bash
curl http://127.0.0.1:8001/health
```

## Experiments

- replay evaluation: `scripts/live_replay_evaluate.py`
- aggregate metrics: `scripts/aggregate_live_liveness.py`

The replay-driven paper batch is written under:

- `outputs/data/live_sessions/`
- `outputs/data/aggregate/`

## Validation Status

Current verification in this workspace includes:

- backend unit tests
- successful local API startup from the project `.venv`
- replay-driven session generation and aggregation for the paper

What is not claimed here:

- continuous durable storage for active sessions
- production-grade deployment hardening
- physical-device mobile validation inside this backend repo alone

## Shared Core Dependency

This service imports shared rPPG logic from `../rppg_core` so offline and live tracks reuse the same method family.
