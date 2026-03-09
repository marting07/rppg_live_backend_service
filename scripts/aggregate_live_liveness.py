#!/usr/bin/env python3
"""Aggregate Track B liveness summary into publication-friendly metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--out", default=Path("outputs/data/aggregate/live_liveness_metrics.csv"), type=Path)
    return p.parse_args()


def safe_rate(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def main() -> int:
    args = parse_args()
    with args.input.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    ok_rows = [r for r in rows if (r.get("status", "") == "complete")]
    total = len(ok_rows)
    correct = sum(1 for r in ok_rows if r.get("decision", "") == r.get("expected_label", ""))

    tp = sum(1 for r in ok_rows if r.get("decision", "") == "live" and r.get("expected_label", "") == "live")
    fp = sum(1 for r in ok_rows if r.get("decision", "") == "live" and r.get("expected_label", "") != "live")
    fn = sum(1 for r in ok_rows if r.get("decision", "") != "live" and r.get("expected_label", "") == "live")

    precision = safe_rate(tp, tp + fp)
    recall = safe_rate(tp, tp + fn)
    accuracy = safe_rate(correct, total)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["samples", "accuracy", "precision_live", "recall_live"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "samples": str(total),
                "accuracy": f"{accuracy:.6f}",
                "precision_live": f"{precision:.6f}",
                "recall_live": f"{recall:.6f}",
            }
        )

    print(f"Wrote liveness metrics: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
