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
    status: str = "pending"
    evaluator: StreamEvaluator = field(default_factory=StreamEvaluator)
    result: dict[str, object] | None = None


class SessionStore:
    def __init__(self, output_root: Path) -> None:
        self._sessions: dict[str, SessionState] = {}
        self.output_root = output_root

    def create_session(self, ttl_seconds: int = 900) -> SessionState:
        now = int(time.time())
        sid = str(uuid.uuid4())
        state = SessionState(
            session_id=sid,
            token=secrets.token_urlsafe(24),
            created_at_unix=now,
            expires_at_unix=now + ttl_seconds,
        )
        self._sessions[sid] = state
        return state

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def complete_session(self, session_id: str) -> dict[str, object] | None:
        state = self._sessions.get(session_id)
        if state is None:
            return None
        state.result = state.evaluator.finalize(session_id=session_id, output_root=self.output_root)
        state.status = "complete"
        return state.result
