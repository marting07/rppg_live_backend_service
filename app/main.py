from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from app.schemas import CaptureConfig, SessionCreateRequest, SessionCreateResponse, SessionResultResponse
from app.session_store import SessionStore

app = FastAPI(title="rPPG Liveness API", version="0.1.0")
store = SessionStore(output_root=Path("outputs/data/live_sessions"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/liveness/sessions", response_model=SessionCreateResponse)
def create_session(request: SessionCreateRequest) -> SessionCreateResponse:
    session = store.create_session(
        device_id=request.device_id,
        app_version=request.app_version,
        preferred_method=request.preferred_method,
    )
    cfg = CaptureConfig()
    stream_url = f"/v1/liveness/sessions/{session.session_id}/stream?token={session.token}"
    return SessionCreateResponse(
        session_id=session.session_id,
        stream_url=stream_url,
        access_token=session.token,
        expires_at_unix=session.expires_at_unix,
        capture_config=cfg,
    )


@app.get("/v1/liveness/sessions/{session_id}/result", response_model=SessionResultResponse)
def get_result(session_id: str) -> SessionResultResponse:
    state = store.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session_not_found")
    if state.result is None:
        return SessionResultResponse(session_id=session_id, status=state.status)
    return SessionResultResponse(**state.result)


@app.post("/v1/liveness/sessions/{session_id}/stop", response_model=SessionResultResponse)
def stop_session(session_id: str) -> SessionResultResponse:
    result = store.complete_session(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="session_not_found")
    return SessionResultResponse(**result)


@app.get("/v1/liveness/sessions/{session_id}/diagnostics")
def get_diagnostics(session_id: str) -> dict[str, object]:
    diagnostics = store.diagnostics(session_id)
    if diagnostics is None:
        raise HTTPException(status_code=404, detail="session_not_found")
    return diagnostics


@app.websocket("/v1/liveness/sessions/{session_id}/stream")
async def stream(session_id: str, websocket: WebSocket) -> None:
    await websocket.accept()
    state = store.get(session_id)
    if state is None:
        await websocket.send_json({"type": "error", "code": "session_not_found"})
        await websocket.close(code=1008)
        return

    token = websocket.query_params.get("token", "")
    if token != state.token:
        await websocket.send_json({"type": "error", "code": "invalid_token"})
        await websocket.close(code=1008)
        return

    state.status = "running"
    last_seq = -1
    try:
        while True:
            payload: dict[str, Any] = await websocket.receive_json()
            msg_type = str(payload.get("type", ""))

            if msg_type == "end_stream":
                result = store.complete_session(session_id)
                await websocket.send_json({"type": "complete", "result": result})
                await websocket.close(code=1000)
                return

            if msg_type != "sample_summary_chunk":
                await websocket.send_json({"type": "error", "code": "unsupported_type"})
                continue

            seq = int(payload.get("seq", -1))
            if seq <= last_seq:
                await websocket.send_json({"type": "error", "code": "out_of_order_seq"})
                continue
            if last_seq >= 0 and seq > last_seq + 1:
                state.evaluator.record_packet_gap(seq - last_seq - 1)
            last_seq = seq

            timestamp_ms = int(payload.get("timestamp_ms", 0))
            patches = payload.get("patches")
            if not isinstance(patches, list) or not patches:
                await websocket.send_json({"type": "error", "code": "missing_patches"})
                continue

            ingest = state.evaluator.ingest_summary_packet(
                seq=seq,
                timestamp_ms=timestamp_ms,
                patches=patches,
                local_quality=payload.get("local_quality"),
                payload_size_bytes=len(json.dumps(payload)),
            )
            await websocket.send_json(
                {
                    "type": "ack",
                    "seq": seq,
                    "accepted": ingest.get("accepted", 0),
                    "timestamp_ms": timestamp_ms,
                }
            )

            await websocket.send_json(
                {
                    "type": "quality_feedback",
                    "seq": seq,
                    "brightness": ingest.get("brightness", 0.0),
                    "motion_score": ingest.get("motion_score", 0.0),
                    "roi_coverage": ingest.get("roi_coverage", 0.0),
                    "message": "hold_steady" if ingest.get("accepted", 0) else "improve_quality",
                }
            )

            result_event = ingest.get("result_event")
            if isinstance(result_event, dict):
                await websocket.send_json(result_event)
    except WebSocketDisconnect:
        if state.status != "complete":
            state.status = "disconnected"
    except Exception as exc:
        await websocket.send_text(json.dumps({"type": "error", "code": "internal_error", "detail": str(exc)}))
        await websocket.close(code=1011)
