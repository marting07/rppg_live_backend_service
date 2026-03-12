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

from rppg_core import ChromMethod, ICAMethod, POSMethod


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
    _accepted_bgr_history: list[tuple[float, float, float]] = field(default_factory=list)
    _session_start_seq: int | None = None
    _first_estimate_seq: int | None = None
    _first_stable_seq: int | None = None

    def __post_init__(self) -> None:
        buf = int(self.fs * self.buffer_seconds)
        # Backend remains the source of truth and evaluates multiple methods for decision robustness.
        self.methods = {
            "chrom": ChromMethod(fs=self.fs, buffer_size=buf),
            "pos": POSMethod(fs=self.fs, buffer_size=buf),
            "ica": ICAMethod(fs=self.fs, buffer_size=buf),
        }
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
        if self._session_start_seq is None:
            self._session_start_seq = seq

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
        self._accepted_bgr_history.append(bgr)
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

        result_event = self._build_result_event(seq=seq, timestamp_ms=timestamp_ms, method_state=method_state)
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
        method_stability_scores: list[float] = []
        for name, values in self._hr_history.items():
            if not values:
                per_method_summary[name] = {"samples": 0.0, "median_bpm": 0.0, "mean_confidence": 0.0, "stability_score": 0.0}
                continue
            arr = np.array(values, dtype=np.float64)
            conf_arr = np.array(self._conf_history[name], dtype=np.float64) if self._conf_history[name] else np.array([], dtype=np.float64)
            median_bpm = float(np.median(arr))
            bpm_std = float(np.std(arr))
            mean_confidence = float(np.mean(conf_arr)) if conf_arr.size else 0.0
            stability_score = self._method_stability_score(
                samples=int(arr.size),
                bpm_std=bpm_std,
                mean_confidence=mean_confidence,
            )
            per_method_summary[name] = {
                "samples": float(arr.size),
                "median_bpm": median_bpm,
                "bpm_std": bpm_std,
                "mean_confidence": mean_confidence,
                "stability_score": stability_score,
            }
            method_stability_scores.append(stability_score)

        method_ratio = float(np.mean(np.array(method_stability_scores, dtype=np.float64))) if method_stability_scores else 0.0
        plausibility = self._signal_plausibility(per_method_summary)
        agreement_support = self._agreement_support(per_method_summary)
        liveness_score = float(
            np.clip(0.42 * method_ratio + 0.23 * quality_ratio + 0.15 * recoverability_rate + 0.20 * plausibility["score"], 0.0, 1.0)
        )
        confidence = float(np.clip((method_ratio + quality_ratio + recoverability_rate + plausibility["score"]) / 4.0, 0.0, 1.0))

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
        elif agreement_support["recoverable_methods"] < 2:
            decision = "inconclusive"
            failure_reasons.append("single_method_evidence")
        elif agreement_support["agreeing_methods"] < 2:
            decision = "inconclusive"
            failure_reasons.append("insufficient_method_agreement")
        elif not agreement_support["has_chromatic_support"]:
            decision = "inconclusive"
            failure_reasons.append("missing_chromatic_support")
        elif plausibility["score"] < 0.72:
            decision = "inconclusive"
            failure_reasons.append("weak_physiological_plausibility")
        elif plausibility["suspicious"]:
            decision = "inconclusive"
            failure_reasons.extend(plausibility["reasons"])
        elif liveness_score < 0.55:
            decision = "inconclusive"
            failure_reasons.append("insufficient_method_stability")
        elif method_ratio < 0.25 and quality_ratio >= 0.70 and recoverability_rate >= 0.70:
            decision = "not_live"
            failure_reasons.append("persistent_instability")

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
                "time_to_first_estimate_ms": self._elapsed_from_seq(self._first_estimate_seq),
                "time_to_stable_estimate_ms": self._elapsed_from_seq(self._first_stable_seq),
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
            "plausibility_summary": {
                "score": plausibility["score"],
                "channel_pulsatility": plausibility["channel_pulsatility"],
                "channel_divergence": plausibility["channel_divergence"],
                "brightness_variation": plausibility["brightness_variation"],
                "method_agreement_score": plausibility["method_agreement_score"],
                "recoverable_methods": float(agreement_support["recoverable_methods"]),
                "agreeing_methods": float(agreement_support["agreeing_methods"]),
                "has_chromatic_support": float(1 if agreement_support["has_chromatic_support"] else 0),
            },
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

    def _build_result_event(self, *, seq: int, timestamp_ms: int, method_state: dict[str, dict[str, float]]) -> dict[str, Any] | None:
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
        if self._first_estimate_seq is None:
            self._first_estimate_seq = seq
        if stable and self._first_stable_ms is None:
            self._first_stable_ms = timestamp_ms
        if stable and self._first_stable_seq is None:
            self._first_stable_seq = seq

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

    def _elapsed_from_seq(self, seq: int | None) -> float | None:
        if seq is None or self._session_start_seq is None:
            return None
        return float(max(0.0, ((seq - self._session_start_seq) / max(self.fs, 1e-6)) * 1000.0))

    def _signal_plausibility(self, per_method_summary: dict[str, dict[str, float]]) -> dict[str, Any]:
        if not self._accepted_bgr_history:
            return {
                "score": 0.0,
                "suspicious": True,
                "reasons": ["no_accepted_signal"],
                "channel_pulsatility": 0.0,
                "channel_divergence": 0.0,
                "brightness_variation": 0.0,
                "method_agreement_score": 0.0,
            }

        arr = np.array(self._accepted_bgr_history, dtype=np.float64)
        # History is stored as BGR.
        b = arr[:, 0]
        g = arr[:, 1]
        r = arr[:, 2]
        eps = 1e-6
        g_norm = (g / max(float(np.mean(g)), eps)) - 1.0
        r_norm = (r / max(float(np.mean(r)), eps)) - 1.0
        b_norm = (b / max(float(np.mean(b)), eps)) - 1.0
        brightness = np.mean(arr, axis=1) / 255.0

        channel_pulsatility = float(np.std(g_norm))
        channel_divergence = float(max(np.std(g_norm - r_norm), np.std(g_norm - b_norm)))
        brightness_variation = float(np.std(brightness))
        method_agreement_score = self._method_agreement_score(per_method_summary)

        pulsatility_score = float(np.clip(channel_pulsatility / 0.006, 0.0, 1.0))
        divergence_score = float(np.clip(channel_divergence / 0.004, 0.0, 1.0))
        brightness_score = float(np.clip(brightness_variation / 0.01, 0.0, 1.0))
        plausibility_score = float(
            np.clip(
                0.40 * pulsatility_score
                + 0.25 * divergence_score
                + 0.15 * brightness_score
                + 0.20 * method_agreement_score,
                0.0,
                1.0,
            )
        )

        suspicious_reasons: list[str] = []
        if channel_pulsatility < 0.0025:
            suspicious_reasons.append("low_pulsatility")
        if channel_divergence < 0.0015:
            suspicious_reasons.append("low_channel_divergence")
        if len(per_method_summary) > 1 and method_agreement_score < 0.50:
            suspicious_reasons.append("poor_method_agreement")
        if brightness_variation < 0.0015 and channel_pulsatility < 0.003:
            suspicious_reasons.append("low_information_signal")
        if channel_pulsatility < 0.004 and brightness_variation < 0.003:
            suspicious_reasons.append("brightness_dominant_signal")

        return {
            "score": plausibility_score,
            "suspicious": bool(suspicious_reasons),
            "reasons": suspicious_reasons,
            "channel_pulsatility": channel_pulsatility,
            "channel_divergence": channel_divergence,
            "brightness_variation": brightness_variation,
            "method_agreement_score": method_agreement_score,
        }

    @staticmethod
    def _method_stability_score(*, samples: int, bpm_std: float, mean_confidence: float) -> float:
        sample_score = float(np.clip(samples / 8.0, 0.0, 1.0))
        bpm_score = float(np.clip((10.0 - bpm_std) / 10.0, 0.0, 1.0))
        confidence_score = float(np.clip(mean_confidence / 0.75, 0.0, 1.0))
        return float(np.clip(0.35 * sample_score + 0.40 * bpm_score + 0.25 * confidence_score, 0.0, 1.0))

    @staticmethod
    def _method_agreement_score(per_method_summary: dict[str, dict[str, float]]) -> float:
        bpms = [
            float(summary["median_bpm"])
            for summary in per_method_summary.values()
            if float(summary.get("samples", 0.0)) >= 5.0 and float(summary.get("median_bpm", 0.0)) > 0.0
        ]
        if len(bpms) <= 1:
            return 1.0
        spread = float(np.std(np.array(bpms, dtype=np.float64)))
        return float(np.clip((15.0 - spread) / 15.0, 0.0, 1.0))

    @staticmethod
    def _agreement_support(per_method_summary: dict[str, dict[str, float]]) -> dict[str, Any]:
        recoverable = {
            name: float(summary["median_bpm"])
            for name, summary in per_method_summary.items()
            if float(summary.get("samples", 0.0)) >= 5.0 and float(summary.get("median_bpm", 0.0)) > 0.0
        }
        if not recoverable:
            return {"recoverable_methods": 0, "agreeing_methods": 0, "has_chromatic_support": False}
        anchor = float(np.median(np.array(list(recoverable.values()), dtype=np.float64)))
        agreeing_names = [name for name, bpm in recoverable.items() if abs(bpm - anchor) <= 8.0]
        has_chromatic_support = any(name in {"chrom", "pos"} for name in agreeing_names)
        return {
            "recoverable_methods": len(recoverable),
            "agreeing_methods": len(agreeing_names),
            "has_chromatic_support": has_chromatic_support,
        }

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
