from __future__ import annotations

import secrets
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.evaluator import StreamEvaluator


@dataclass
class SessionState:
    session_id: str
    token: str
    created_at_unix: int
    expires_at_unix: int
    device_id: str | None = None
    app_version: str | None = None
    preferred_method: str | None = None
    status: str = "pending"
    evaluator: StreamEvaluator = field(default_factory=StreamEvaluator)
    result: dict[str, object] | None = None


class SessionStore:
    def __init__(self, output_root: Path) -> None:
        self._sessions: dict[str, SessionState] = {}
        self.output_root = output_root

    def create_session(
        self,
        *,
        ttl_seconds: int = 900,
        device_id: str | None = None,
        app_version: str | None = None,
        preferred_method: str | None = None,
    ) -> SessionState:
        now = int(time.time())
        sid = str(uuid.uuid4())
        state = SessionState(
            session_id=sid,
            token=secrets.token_urlsafe(24),
            created_at_unix=now,
            expires_at_unix=now + ttl_seconds,
            device_id=device_id,
            app_version=app_version,
            preferred_method=preferred_method,
            evaluator=StreamEvaluator(preferred_method=preferred_method),
        )
        self._sessions[sid] = state
        return state

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def diagnostics(self, session_id: str) -> dict[str, object] | None:
        state = self._sessions.get(session_id)
        if state is None:
            return None
        latest_quality = state.evaluator.quality_timeline[-1] if state.evaluator.quality_timeline else {}
        latest_result = state.evaluator.bpm_timeline[-1] if state.evaluator.bpm_timeline else {}
        latest_coherence = state.evaluator.coherence_timeline[-1] if state.evaluator.coherence_timeline else {}
        latest_patch_group_bpm = state.evaluator.patch_group_bpm_timeline[-1] if state.evaluator.patch_group_bpm_timeline else {}
        latest_patch_group_quality = (
            state.evaluator.patch_group_quality_timeline[-1] if state.evaluator.patch_group_quality_timeline else {}
        )
        return {
            "session_id": session_id,
            "status": state.status,
            "device_id": state.device_id,
            "app_version": state.app_version,
            "preferred_method": state.preferred_method,
            "packets_received": state.evaluator.quality.total_packets,
            "packets_accepted": state.evaluator.quality.accepted_packets,
            "latest_quality": latest_quality,
            "latest_coherence": latest_coherence,
            "latest_patch_group_bpm": latest_patch_group_bpm,
            "latest_patch_group_quality": latest_patch_group_quality,
            "latest_result_event": latest_result,
        }

    def complete_session(self, session_id: str) -> dict[str, object] | None:
        state = self._sessions.get(session_id)
        if state is None:
            return None
        state.result = state.evaluator.finalize(session_id=session_id, output_root=self.output_root)
        state.status = "complete"
        return state.result
