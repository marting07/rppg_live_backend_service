from __future__ import annotations

import base64
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import cv2  # type: ignore
import numpy as np

PAPERS_ROOT = Path(__file__).resolve().parents[2]
if str(PAPERS_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPERS_ROOT))

from rppg_core import ChromMethod, GreenMethod, POSMethod, SSRMethod


@dataclass
class QualityStats:
    accepted_frames: int = 0
    total_frames: int = 0
    mean_brightness: float = 0.0
    mean_motion_score: float = 0.0


@dataclass
class StreamEvaluator:
    fs: float = 12.0
    buffer_seconds: int = 10
    methods: dict[str, object] = field(default_factory=dict)
    quality: QualityStats = field(default_factory=QualityStats)
    _last_gray: np.ndarray | None = None
    _hr_history: dict[str, list[float]] = field(default_factory=dict)
    _conf_history: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        buf = int(self.fs * self.buffer_seconds)
        self.methods = {
            "green": GreenMethod(fs=self.fs, buffer_size=buf),
            "chrom": ChromMethod(fs=self.fs, buffer_size=buf),
            "pos": POSMethod(fs=self.fs, buffer_size=buf),
            "ssr": SSRMethod(fs=self.fs, buffer_size=buf),
        }
        self._hr_history = {k: [] for k in self.methods}
        self._conf_history = {k: [] for k in self.methods}

    def _decode_b64_image(self, image_bytes_b64: str) -> np.ndarray | None:
        try:
            raw = base64.b64decode(image_bytes_b64, validate=True)
        except Exception:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame

    def _quality_gate(self, roi: np.ndarray) -> tuple[bool, dict[str, float]]:
        if roi is None or roi.size == 0:
            return False, {"brightness": 0.0, "motion": 1.0, "size_ok": 0.0}

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray) / 255.0)

        motion = 0.0
        if self._last_gray is not None and self._last_gray.shape == gray.shape:
            motion = float(np.mean(np.abs(gray.astype(np.float64) - self._last_gray.astype(np.float64))) / 255.0)
        self._last_gray = gray

        size_ok = 1.0 if (roi.shape[0] * roi.shape[1]) >= (64 * 64) else 0.0
        ok = (brightness >= 0.12) and (brightness <= 0.95) and (motion <= 0.20) and (size_ok > 0.0)
        return ok, {"brightness": brightness, "motion": motion, "size_ok": size_ok}

    def ingest_roi_payload(self, image_bytes_b64: str) -> dict[str, float | int | str]:
        roi = self._decode_b64_image(image_bytes_b64)
        self.quality.total_frames += 1
        if roi is None:
            return {"accepted": 0, "error": "invalid_image"}

        ok, q = self._quality_gate(roi)
        self.quality.mean_brightness += q["brightness"]
        self.quality.mean_motion_score += q["motion"]
        if not ok:
            return {"accepted": 0, **q}

        self.quality.accepted_frames += 1
        for name, method in self.methods.items():
            method.update(roi)
            hr = method.get_hr()
            conf = method.get_confidence() if hasattr(method, "get_confidence") else None
            if hr is not None and np.isfinite(hr):
                self._hr_history[name].append(float(hr))
            if conf is not None and np.isfinite(conf):
                self._conf_history[name].append(float(conf))
        return {"accepted": 1, **q}

    def finalize(self, session_id: str, output_root: Path) -> dict[str, object]:
        output_root.mkdir(parents=True, exist_ok=True)
        run_id = hashlib.sha1(f"{session_id}|{self.quality.total_frames}".encode("utf-8")).hexdigest()[:12]

        method_scores: dict[str, float] = {}
        valid_methods = 0
        for name, values in self._hr_history.items():
            if len(values) >= 5:
                med = float(np.median(np.array(values, dtype=np.float64)))
                plausible = 45.0 <= med <= 180.0
                conf_mean = float(np.mean(np.array(self._conf_history[name], dtype=np.float64))) if self._conf_history[name] else 0.0
                score = 1.0 if plausible else 0.0
                score = min(1.0, score * 0.7 + min(conf_mean / 2.0, 1.0) * 0.3)
                method_scores[name] = score
                if score >= 0.5:
                    valid_methods += 1
            else:
                method_scores[name] = 0.0

        quality_ratio = (
            float(self.quality.accepted_frames) / float(self.quality.total_frames)
            if self.quality.total_frames > 0
            else 0.0
        )
        method_ratio = float(valid_methods) / float(len(self.methods))
        liveness_score = float(np.clip(0.6 * method_ratio + 0.4 * quality_ratio, 0.0, 1.0))
        confidence = float(np.clip((method_ratio + quality_ratio) / 2.0, 0.0, 1.0))

        failure_reasons: list[str] = []
        decision = "live"
        if self.quality.total_frames < int(self.fs * 4):
            decision = "inconclusive"
            failure_reasons.append("too_few_frames")
        elif quality_ratio < 0.40:
            decision = "inconclusive"
            failure_reasons.append("low_quality_ratio")
        elif liveness_score < 0.65:
            decision = "not_live"
            failure_reasons.append("insufficient_method_agreement")

        quality_summary = {
            "accepted_ratio": quality_ratio,
            "mean_brightness": (self.quality.mean_brightness / max(self.quality.total_frames, 1)),
            "mean_motion_score": (self.quality.mean_motion_score / max(self.quality.total_frames, 1)),
            "total_frames": float(self.quality.total_frames),
        }

        result = {
            "session_id": session_id,
            "status": "complete",
            "run_id": run_id,
            "decision": decision,
            "liveness_score": liveness_score,
            "confidence": confidence,
            "method_scores": method_scores,
            "quality_summary": quality_summary,
            "failure_reasons": failure_reasons,
            "computed_at": datetime.now(UTC).isoformat(),
        }

        run_dir = output_root / session_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
