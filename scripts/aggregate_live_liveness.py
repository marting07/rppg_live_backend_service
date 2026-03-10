#!/usr/bin/env python3
"""Aggregate Track B live-study summary into operational metrics for the paper."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--out", default=Path("outputs/data/aggregate/live_liveness_metrics.csv"), type=Path)
    return p.parse_args()


def safe_mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def parse_float(row: dict[str, str], key: str) -> float | None:
    raw = str(row.get(key, "")).strip()
    if not raw:
        return None
    return float(raw)


def main() -> int:
    args = parse_args()
    with args.input.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    ok_rows = [r for r in rows if (r.get("status", "") == "complete")]
    total = len(ok_rows)
    live_rows = [r for r in ok_rows if r.get("decision", "") == "live"]

    first_estimates = [value for r in ok_rows if (value := parse_float(r, "time_to_first_estimate_ms")) is not None]
    stable_estimates = [value for r in ok_rows if (value := parse_float(r, "time_to_stable_estimate_ms")) is not None]
    valid_window_rates = [value for r in ok_rows if (value := parse_float(r, "valid_window_rate")) is not None]
    recoverability_rates = [value for r in ok_rows if (value := parse_float(r, "recoverability_rate")) is not None]
    bandwidth_values = [value for r in ok_rows if (value := parse_float(r, "mean_bandwidth_bps")) is not None]
    packet_loss_values = [value for r in ok_rows if (value := parse_float(r, "packet_loss_rate")) is not None]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "samples",
                "live_decision_rate",
                "mean_time_to_first_estimate_ms",
                "mean_time_to_stable_estimate_ms",
                "mean_valid_window_rate",
                "mean_recoverability_rate",
                "mean_bandwidth_bps",
                "mean_packet_loss_rate",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "samples": str(total),
                "live_decision_rate": f"{(len(live_rows) / float(total) if total else 0.0):.6f}",
                "mean_time_to_first_estimate_ms": f"{safe_mean(first_estimates):.6f}",
                "mean_time_to_stable_estimate_ms": f"{safe_mean(stable_estimates):.6f}",
                "mean_valid_window_rate": f"{safe_mean(valid_window_rates):.6f}",
                "mean_recoverability_rate": f"{safe_mean(recoverability_rates):.6f}",
                "mean_bandwidth_bps": f"{safe_mean(bandwidth_values):.6f}",
                "mean_packet_loss_rate": f"{safe_mean(packet_loss_values):.6f}",
            }
        )

    print(f"Wrote liveness metrics: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
