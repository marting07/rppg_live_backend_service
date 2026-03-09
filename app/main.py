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
def create_session(_: SessionCreateRequest) -> SessionCreateResponse:
    session = store.create_session()
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
    frame_count = 0
    try:
        while True:
            payload: dict[str, Any] = await websocket.receive_json()
            msg_type = str(payload.get("type", ""))

            if msg_type == "end_stream":
                result = store.complete_session(session_id)
                await websocket.send_json({"type": "complete", "result": result})
                await websocket.close(code=1000)
                return

            if msg_type != "roi_frame_chunk":
                await websocket.send_json({"type": "error", "code": "unsupported_type"})
                continue

            seq = int(payload.get("seq", -1))
            if seq <= last_seq:
                await websocket.send_json({"type": "error", "code": "out_of_order_seq"})
                continue
            last_seq = seq

            image_b64 = payload.get("image_bytes_b64")
            if not isinstance(image_b64, str) or not image_b64:
                await websocket.send_json({"type": "error", "code": "missing_image_bytes_b64"})
                continue

            ingest = state.evaluator.ingest_roi_payload(image_b64)
            frame_count += 1
            await websocket.send_json({"type": "ack", "seq": seq, "accepted": ingest.get("accepted", 0)})

            if frame_count % 12 == 0:
                await websocket.send_json(
                    {
                        "type": "quality_feedback",
                        "frame_count": frame_count,
                        "brightness": ingest.get("brightness", 0.0),
                        "motion": ingest.get("motion", 0.0),
                        "message": "hold_steady" if ingest.get("accepted", 0) else "improve_quality",
                    }
                )
    except WebSocketDisconnect:
        if state.status != "complete":
            state.status = "disconnected"
    except Exception as exc:
        await websocket.send_text(json.dumps({"type": "error", "code": "internal_error", "detail": str(exc)}))
        await websocket.close(code=1011)
