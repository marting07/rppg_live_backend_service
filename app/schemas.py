from __future__ import annotations

from pydantic import BaseModel, Field


class SessionCreateRequest(BaseModel):
    device_id: str | None = None
    app_version: str | None = None
    preferred_method: str | None = None
    expected_duration_seconds: float = Field(default=10.0, ge=4.0, le=20.0)


class CaptureConfig(BaseModel):
    fps: int = 12
    patch_rows: int = 2
    patch_cols: int = 3
    min_duration_seconds: int = 8
    max_duration_seconds: int = 12
    transport_format: str = "patch_rgb_v1"


class SessionCreateResponse(BaseModel):
    session_id: str
    stream_url: str
    access_token: str
    expires_at_unix: int
    capture_config: CaptureConfig


class SessionResultResponse(BaseModel):
    session_id: str
    status: str
    run_id: str | None = None
    decision: str | None = None
    liveness_score: float | None = None
    confidence: float | None = None
    selected_method: str | None = None
    method_scores: dict[str, float] = Field(default_factory=dict)
    quality_summary: dict[str, float] = Field(default_factory=dict)
    operational_metrics: dict[str, float | None] = Field(default_factory=dict)
    method_summary: dict[str, dict[str, float]] = Field(default_factory=dict)
    failure_reasons: list[str] = Field(default_factory=list)
