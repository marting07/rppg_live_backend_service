# rppg_live_backend_service

FastAPI backend for real-time Track B evaluation.

## Backend Responsibilities

- session lifecycle management
- packet validation and timestamp-aware buffering
- shared preprocessing and classical rPPG execution
- quality and estimation logic (provisional/stable behavior)
- replayable session artifacts and aggregate experiment outputs

## API

- `POST /v1/liveness/sessions`
- `WS /v1/liveness/sessions/{session_id}/stream?token=...`
- `GET /v1/liveness/sessions/{session_id}/result`

## Experiments

- replay evaluation: `scripts/live_replay_evaluate.py`
- aggregate metrics: `scripts/aggregate_live_liveness.py`

## Shared Core Dependency

This service imports shared rPPG logic from `../rppg_core`.
