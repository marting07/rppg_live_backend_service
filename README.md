# rppg_live_backend_service

FastAPI backend for real-time Track B evaluation.

## Backend Responsibilities

- session lifecycle management
- packet validation, sequence tracking, and timestamp-aware ingestion
- shared preprocessing and centralized classical rPPG execution from compact RGB summaries
- quality and estimation logic with provisional/stable live events
- replayable session artifacts (`packet_trace`, quality timeline, BPM timeline, final summary)
- aggregate operational experiment outputs for the paper

## API

- `POST /v1/liveness/sessions`
- `POST /v1/liveness/sessions/{session_id}/stop`
- `WS /v1/liveness/sessions/{session_id}/stream?token=...`
- `GET /v1/liveness/sessions/{session_id}/result`
- `GET /v1/liveness/sessions/{session_id}/diagnostics`

## Experiments

- replay evaluation: `scripts/live_replay_evaluate.py`
- aggregate metrics: `scripts/aggregate_live_liveness.py`

## Environment Setup

- `make install`: create `.venv` and install the API/runtime dependencies
- `make install-replay`: extend the same `.venv` with replay/video dependencies

## Shared Core Dependency

This service imports shared rPPG logic from `../rppg_core`.
