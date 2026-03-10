#!/usr/bin/env python3
"""Replay local videos through the live summary-stream evaluator for Track B experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Any

import cv2  # type: ignore
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluator import StreamEvaluator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out", default=Path("outputs/data/aggregate/live_liveness_summary.csv"), type=Path)
    p.add_argument("--fps", type=float, default=12.0)
    p.add_argument("--max-seconds", type=float, default=12.0)
    return p.parse_args()


def stable_session_id(video_path: str, expected_label: str) -> str:
    digest = hashlib.sha1(f"{video_path}|{expected_label}".encode("utf-8")).hexdigest()[:12]
    return f"live_{digest}"


def center_roi(frame: np.ndarray, roi_w: int = 96, roi_h: int = 96) -> np.ndarray:
    h, w = frame.shape[:2]
    cx = w // 2
    cy = h // 3
    x0 = max(0, cx - roi_w // 2)
    y0 = max(0, cy - roi_h // 2)
    x1 = min(w, x0 + roi_w)
    y1 = min(h, y0 + roi_h)
    return frame[y0:y1, x0:x1]


def build_patch_payload(roi: np.ndarray, seq: int, timestamp_ms: int) -> dict[str, Any]:
    rows = 2
    cols = 3
    patches: list[dict[str, Any]] = []
    patch_h = max(1, roi.shape[0] // rows)
    patch_w = max(1, roi.shape[1] // cols)
    prev_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(prev_gray) / 255.0)
    for row in range(rows):
        for col in range(cols):
            y0 = row * patch_h
            y1 = roi.shape[0] if row == rows - 1 else min(roi.shape[0], (row + 1) * patch_h)
            x0 = col * patch_w
            x1 = roi.shape[1] if col == cols - 1 else min(roi.shape[1], (col + 1) * patch_w)
            patch = roi[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            mean_bgr = np.mean(patch.reshape(-1, 3), axis=0)
            patches.append(
                {
                    "patch_id": f"r{row}c{col}",
                    "mean_rgb": [float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0])],
                    "weight": float(patch.shape[0] * patch.shape[1]),
                }
            )

    return {
        "type": "sample_summary_chunk",
        "seq": seq,
        "timestamp_ms": timestamp_ms,
        "patches": patches,
        "local_quality": {
            "face_present": True,
            "brightness": brightness,
            "motion_score": 0.0,
            "roi_coverage": 1.0,
        },
    }


def run_one(video_path: Path, expected_label: str, fps: float, max_seconds: float, output_root: Path, session_id: str) -> dict[str, object]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "session_id": session_id,
            "status": "failed",
            "decision": "",
            "liveness_score": "",
            "confidence": "",
            "expected_label": expected_label,
            "selected_method": "",
            "time_to_first_estimate_ms": "",
            "time_to_stable_estimate_ms": "",
            "valid_window_rate": "",
            "recoverability_rate": "",
            "mean_bandwidth_bps": "",
            "packet_loss_rate": "",
            "error": f"cannot_open_video:{video_path}",
        }

    evaluator = StreamEvaluator(fs=fps)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = src_fps if src_fps and src_fps > 0 else 30.0
    step = max(1, int(round(src_fps / fps)))
    max_frames = int(max_seconds * fps)

    frame_idx = 0
    sent = 0
    while sent < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            roi = center_roi(frame)
            packet = build_patch_payload(roi, seq=sent, timestamp_ms=int(round((sent / fps) * 1000.0)))
            evaluator.ingest_summary_packet(
                seq=packet["seq"],
                timestamp_ms=packet["timestamp_ms"],
                patches=packet["patches"],
                local_quality=packet["local_quality"],
                payload_size_bytes=len(str(packet)),
            )
            sent += 1
        frame_idx += 1
    cap.release()

    result = evaluator.finalize(session_id=session_id, output_root=output_root)
    operational = result.get("operational_metrics", {})
    return {
        "session_id": session_id,
        "status": str(result.get("status", "")),
        "decision": str(result.get("decision", "")),
        "liveness_score": f"{float(result.get('liveness_score', 0.0)):.6f}",
        "confidence": f"{float(result.get('confidence', 0.0)):.6f}",
        "expected_label": expected_label,
        "selected_method": str(result.get("selected_method", "") or ""),
        "time_to_first_estimate_ms": _fmt_metric(operational.get("time_to_first_estimate_ms")),
        "time_to_stable_estimate_ms": _fmt_metric(operational.get("time_to_stable_estimate_ms")),
        "valid_window_rate": _fmt_metric(operational.get("valid_window_rate")),
        "recoverability_rate": _fmt_metric(operational.get("recoverability_rate")),
        "mean_bandwidth_bps": _fmt_metric(operational.get("mean_bandwidth_bps")),
        "packet_loss_rate": _fmt_metric(operational.get("packet_loss_rate")),
        "error": "",
    }


def _fmt_metric(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6f}"


def main() -> int:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)

    with args.manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows: list[dict[str, object]] = []
    output_root = Path("outputs/data/live_sessions")

    for row in rows:
        video = Path(str(row.get("video_path", "")).strip())
        expected = str(row.get("expected_label", "live")).strip() or "live"
        sid = str(row.get("session_id", "")).strip() or stable_session_id(str(video), expected)
        out_rows.append(run_one(video, expected, args.fps, args.max_seconds, output_root, sid))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "session_id",
                "status",
                "decision",
                "liveness_score",
                "confidence",
                "expected_label",
                "selected_method",
                "time_to_first_estimate_ms",
                "time_to_stable_estimate_ms",
                "valid_window_rate",
                "recoverability_rate",
                "mean_bandwidth_bps",
                "packet_loss_rate",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote replay summary: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
