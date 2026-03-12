from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PAPERS_ROOT = Path(__file__).resolve().parents[2]
if str(PAPERS_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPERS_ROOT))

from rppg_core import ChromMethod, GreenMethod, POSMethod


@dataclass
class QualityStats:
    accepted_packets: int = 0
    total_packets: int = 0
    mean_brightness: float = 0.0
    mean_motion_score: float = 0.0
    mean_face_coverage: float = 0.0
    bytes_received: int = 0
    dropped_packets: int = 0


@dataclass
class StreamEvaluator:
    fs: float = 12.0
    buffer_seconds: int = 10
    preferred_method: str | None = None
    methods: dict[str, object] = field(default_factory=dict)
    quality: QualityStats = field(default_factory=QualityStats)
    packet_trace: list[dict[str, Any]] = field(default_factory=list)
    quality_timeline: list[dict[str, Any]] = field(default_factory=list)
    bpm_timeline: list[dict[str, Any]] = field(default_factory=list)
    _hr_history: dict[str, list[float]] = field(default_factory=dict)
    _conf_history: dict[str, list[float]] = field(default_factory=dict)
    _latest_result_event: dict[str, Any] | None = None
    _latest_result_kind: str | None = None
    _session_start_ms: int | None = None
    _first_estimate_ms: int | None = None
    _first_stable_ms: int | None = None
    _last_timestamp_ms: int | None = None

    def __post_init__(self) -> None:
        buf = int(self.fs * self.buffer_seconds)
        available_methods = {
            "green": GreenMethod(fs=self.fs, buffer_size=buf),
            "chrom": ChromMethod(fs=self.fs, buffer_size=buf),
            "pos": POSMethod(fs=self.fs, buffer_size=buf),
        }
        preferred = (self.preferred_method or "").lower().strip()
        if preferred in available_methods:
            self.methods = {preferred: available_methods[preferred]}
        else:
            # Selected live methods use compact RGB summaries without backend-side ROI images.
            self.methods = available_methods
        self._hr_history = {k: [] for k in self.methods}
        self._conf_history = {k: [] for k in self.methods}

    def ingest_summary_packet(
        self,
        *,
        seq: int,
        timestamp_ms: int,
        patches: list[dict[str, Any]],
        local_quality: dict[str, Any] | None,
        payload_size_bytes: int,
    ) -> dict[str, Any]:
        self.quality.total_packets += 1
        self.quality.bytes_received += max(payload_size_bytes, 0)

        if self._session_start_ms is None:
            self._session_start_ms = timestamp_ms

        if not patches:
            return {"accepted": 0, "error": "missing_patches"}

        bgr = self._weighted_mean_bgr(patches)
        brightness = self._resolve_brightness(bgr, local_quality)
        motion = self._resolve_float(local_quality, "motion_score", default=0.0)
        coverage = self._resolve_float(local_quality, "roi_coverage", default=1.0)
        face_present = self._resolve_bool(local_quality, "face_present", default=True)

        expected_dt_ms = 1000.0 / max(self.fs, 1e-6)
        inter_packet_ms = None
        jitter_ms = 0.0
        if self._last_timestamp_ms is not None:
            inter_packet_ms = float(max(0, timestamp_ms - self._last_timestamp_ms))
            jitter_ms = abs(inter_packet_ms - expected_dt_ms)
        self._last_timestamp_ms = timestamp_ms

        accepted = bool(
            face_present
            and 0.12 <= brightness <= 0.95
            and motion <= 0.20
            and coverage >= 0.60
        )

        self.quality.mean_brightness += brightness
        self.quality.mean_motion_score += motion
        self.quality.mean_face_coverage += coverage

        packet_info = {
            "seq": seq,
            "timestamp_ms": timestamp_ms,
            "accepted": int(accepted),
            "brightness": brightness,
            "motion_score": motion,
            "roi_coverage": coverage,
            "face_present": int(face_present),
            "inter_packet_ms": inter_packet_ms,
            "jitter_ms": jitter_ms,
            "payload_size_bytes": payload_size_bytes,
        }
        self.packet_trace.append(packet_info)

        if not accepted:
            self.quality_timeline.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "accepted": 0,
                    "brightness": brightness,
                    "motion_score": motion,
                    "roi_coverage": coverage,
                }
            )
            return {"accepted": 0, **packet_info}

        self.quality.accepted_packets += 1
        roi = self._synthetic_roi_from_bgr(bgr)
        method_state: dict[str, dict[str, float]] = {}
        for name, method in self.methods.items():
            method.update(roi)
            hr = method.get_hr()
            conf = method.get_confidence() if hasattr(method, "get_confidence") else None
            if hr is not None and np.isfinite(hr):
                self._hr_history[name].append(float(hr))
            if conf is not None and np.isfinite(conf):
                self._conf_history[name].append(float(conf))
            if hr is not None and np.isfinite(hr):
                method_state[name] = {
                    "bpm": float(hr),
                    "confidence": float(conf) if conf is not None and np.isfinite(conf) else 0.0,
                }

        self.quality_timeline.append(
            {
                "timestamp_ms": timestamp_ms,
                "accepted": 1,
                "brightness": brightness,
                "motion_score": motion,
                "roi_coverage": coverage,
            }
        )

        result_event = self._build_result_event(timestamp_ms=timestamp_ms, method_state=method_state)
        response = {"accepted": 1, **packet_info}
        if result_event is not None:
            response["result_event"] = result_event
        return response

    def record_packet_gap(self, missing_packets: int) -> None:
        self.quality.dropped_packets += max(0, missing_packets)

    def finalize(self, session_id: str, output_root: Path) -> dict[str, object]:
        output_root.mkdir(parents=True, exist_ok=True)
        run_id = hashlib.sha1(f"{session_id}|{self.quality.total_packets}".encode("utf-8")).hexdigest()[:12]

        quality_ratio = (
            float(self.quality.accepted_packets) / float(self.quality.total_packets)
            if self.quality.total_packets > 0
            else 0.0
        )
        recoverable_methods = sum(1 for values in self._hr_history.values() if len(values) >= 5)
        recoverability_rate = float(recoverable_methods) / float(len(self.methods)) if self.methods else 0.0

        per_method_summary: dict[str, dict[str, float]] = {}
        stable_method_votes = 0
        for name, values in self._hr_history.items():
            if not values:
                per_method_summary[name] = {"samples": 0.0, "median_bpm": 0.0, "mean_confidence": 0.0}
                continue
            arr = np.array(values, dtype=np.float64)
            conf_arr = np.array(self._conf_history[name], dtype=np.float64) if self._conf_history[name] else np.array([], dtype=np.float64)
            median_bpm = float(np.median(arr))
            bpm_std = float(np.std(arr))
            mean_confidence = float(np.mean(conf_arr)) if conf_arr.size else 0.0
            per_method_summary[name] = {
                "samples": float(arr.size),
                "median_bpm": median_bpm,
                "bpm_std": bpm_std,
                "mean_confidence": mean_confidence,
            }
            if arr.size >= 5 and bpm_std <= 5.0 and mean_confidence >= 1.05:
                stable_method_votes += 1

        method_ratio = float(stable_method_votes) / float(len(self.methods)) if self.methods else 0.0
        liveness_score = float(np.clip(0.55 * method_ratio + 0.25 * quality_ratio + 0.20 * recoverability_rate, 0.0, 1.0))
        confidence = float(np.clip((method_ratio + quality_ratio + recoverability_rate) / 3.0, 0.0, 1.0))

        decision = "live"
        failure_reasons: list[str] = []
        if self.quality.total_packets < int(self.fs * 4):
            decision = "inconclusive"
            failure_reasons.append("too_few_packets")
        elif quality_ratio < 0.40:
            decision = "inconclusive"
            failure_reasons.append("low_quality_ratio")
        elif recoverability_rate < 0.34:
            decision = "inconclusive"
            failure_reasons.append("low_recoverability")
        elif liveness_score < 0.65:
            decision = "not_live"
            failure_reasons.append("insufficient_method_stability")

        mean_payload_bytes = float(self.quality.bytes_received) / float(self.quality.total_packets) if self.quality.total_packets else 0.0
        mean_bandwidth_bps = mean_payload_bytes * self.fs
        inter_packet_values = [
            float(item["inter_packet_ms"])
            for item in self.packet_trace
            if item.get("inter_packet_ms") is not None
        ]
        jitter_values = [float(item.get("jitter_ms", 0.0)) for item in self.packet_trace]

        result = {
            "session_id": session_id,
            "status": "complete",
            "run_id": run_id,
            "decision": decision,
            "liveness_score": liveness_score,
            "confidence": confidence,
            "selected_method": self._select_best_method(),
            "method_scores": {name: float(summary.get("mean_confidence", 0.0)) for name, summary in per_method_summary.items()},
            "quality_summary": {
                "accepted_ratio": quality_ratio,
                "mean_brightness": self.quality.mean_brightness / max(self.quality.total_packets, 1),
                "mean_motion_score": self.quality.mean_motion_score / max(self.quality.total_packets, 1),
                "mean_face_coverage": self.quality.mean_face_coverage / max(self.quality.total_packets, 1),
                "total_packets": float(self.quality.total_packets),
                "accepted_packets": float(self.quality.accepted_packets),
            },
            "operational_metrics": {
                "time_to_first_estimate_ms": self._elapsed_or_none(self._first_estimate_ms),
                "time_to_stable_estimate_ms": self._elapsed_or_none(self._first_stable_ms),
                "valid_window_rate": quality_ratio,
                "recoverability_rate": recoverability_rate,
                "stable_estimate_rate": method_ratio,
                "mean_inter_packet_ms": float(np.mean(inter_packet_values)) if inter_packet_values else 0.0,
                "mean_jitter_ms": float(np.mean(jitter_values)) if jitter_values else 0.0,
                "mean_payload_bytes": mean_payload_bytes,
                "mean_bandwidth_bps": mean_bandwidth_bps,
                "packet_loss_rate": self._packet_loss_rate(),
            },
            "method_summary": per_method_summary,
            "failure_reasons": failure_reasons,
            "computed_at": datetime.now(UTC).isoformat(),
        }

        run_dir = output_root / session_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        self._write_jsonl(run_dir / "packet_trace.jsonl", self.packet_trace)
        (run_dir / "quality_timeline.json").write_text(json.dumps(self.quality_timeline, indent=2), encoding="utf-8")
        (run_dir / "bpm_timeline.json").write_text(json.dumps(self.bpm_timeline, indent=2), encoding="utf-8")
        diagnostics = {
            "latest_result_kind": self._latest_result_kind,
            "packet_trace_path": str(run_dir / "packet_trace.jsonl"),
            "quality_timeline_path": str(run_dir / "quality_timeline.json"),
            "bpm_timeline_path": str(run_dir / "bpm_timeline.json"),
        }
        (run_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        return result

    def _build_result_event(self, *, timestamp_ms: int, method_state: dict[str, dict[str, float]]) -> dict[str, Any] | None:
        if not method_state:
            return None
        selected_method = self._choose_method(method_state)
        selected = method_state[selected_method]
        confidences = np.array([entry["confidence"] for entry in method_state.values()], dtype=np.float64)
        bpms = np.array([entry["bpm"] for entry in method_state.values()], dtype=np.float64)
        stable = bool(
            bpms.size >= 2
            and np.std(bpms) <= 5.0
            and float(np.mean(confidences)) >= 1.05
        )
        kind = "stable_result" if stable else "provisional_result"
        if self._first_estimate_ms is None:
            self._first_estimate_ms = timestamp_ms
        if stable and self._first_stable_ms is None:
            self._first_stable_ms = timestamp_ms

        result_event = {
            "type": kind,
            "selected_method": selected_method,
            "bpm": float(selected["bpm"]),
            "confidence": float(np.mean(confidences)),
            "method_state": method_state,
            "timestamp_ms": timestamp_ms,
        }
        self._latest_result_event = result_event
        self._latest_result_kind = kind
        self.bpm_timeline.append(result_event)
        return result_event

    def _select_best_method(self) -> str | None:
        ranked = self._rank_methods()
        if not ranked:
            return None
        return ranked[0][2]

    def _choose_method(self, method_state: dict[str, dict[str, float]]) -> str:
        preferred = self.preferred_method
        if preferred and preferred in method_state:
            return preferred

        ranked = [entry for entry in self._rank_methods(method_state=method_state) if entry[2] in method_state]
        if ranked:
            return ranked[0][2]
        return max(method_state.items(), key=lambda item: (item[1]["confidence"], item[1]["bpm"]))[0]

    def _rank_methods(self, *, method_state: dict[str, dict[str, float]] | None = None) -> list[tuple[float, int, str]]:
        ranked = []
        for name, values in self._hr_history.items():
            if not values:
                continue
            mean_conf = float(np.mean(np.array(self._conf_history[name], dtype=np.float64))) if self._conf_history[name] else 0.0
            current_conf = 0.0
            if method_state is not None and name in method_state:
                current_conf = float(method_state[name].get("confidence", 0.0))
            ranked.append((mean_conf + 0.05 * current_conf, len(values), name))
        ranked.sort(reverse=True)
        return ranked

    def _packet_loss_rate(self) -> float:
        delivered = self.quality.total_packets + self.quality.dropped_packets
        if delivered <= 0:
            return 0.0
        return float(self.quality.dropped_packets) / float(delivered)

    def _elapsed_or_none(self, timestamp_ms: int | None) -> float | None:
        if timestamp_ms is None or self._session_start_ms is None:
            return None
        return float(max(0, timestamp_ms - self._session_start_ms))

    @staticmethod
    def _resolve_float(local_quality: dict[str, Any] | None, key: str, *, default: float) -> float:
        if local_quality is None:
            return default
        value = local_quality.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _resolve_bool(local_quality: dict[str, Any] | None, key: str, *, default: bool) -> bool:
        if local_quality is None:
            return default
        return bool(local_quality.get(key, default))

    @staticmethod
    def _weighted_mean_bgr(patches: list[dict[str, Any]]) -> tuple[float, float, float]:
        weights = []
        b_vals = []
        g_vals = []
        r_vals = []
        for patch in patches:
            mean_rgb = patch.get("mean_rgb", [])
            if not isinstance(mean_rgb, list) or len(mean_rgb) != 3:
                continue
            weight = float(patch.get("weight", 1.0))
            r_vals.append(float(mean_rgb[0]))
            g_vals.append(float(mean_rgb[1]))
            b_vals.append(float(mean_rgb[2]))
            weights.append(max(weight, 1e-6))
        if not weights:
            return 0.0, 0.0, 0.0
        w = np.array(weights, dtype=np.float64)
        b = float(np.average(np.array(b_vals, dtype=np.float64), weights=w))
        g = float(np.average(np.array(g_vals, dtype=np.float64), weights=w))
        r = float(np.average(np.array(r_vals, dtype=np.float64), weights=w))
        return b, g, r

    @staticmethod
    def _resolve_brightness(bgr: tuple[float, float, float], local_quality: dict[str, Any] | None) -> float:
        if local_quality is not None and local_quality.get("brightness") is not None:
            try:
                return float(local_quality["brightness"])
            except (TypeError, ValueError):
                pass
        b, g, r = bgr
        return float(np.mean([b, g, r]) / 255.0)

    @staticmethod
    def _synthetic_roi_from_bgr(bgr: tuple[float, float, float]) -> np.ndarray:
        roi = np.zeros((8, 8, 3), dtype=np.uint8)
        b, g, r = bgr
        roi[:, :, 0] = np.clip(int(round(b)), 0, 255)
        roi[:, :, 1] = np.clip(int(round(g)), 0, 255)
        roi[:, :, 2] = np.clip(int(round(r)), 0, 255)
        return roi

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
