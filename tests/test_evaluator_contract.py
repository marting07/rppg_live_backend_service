from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.evaluator import StreamEvaluator


def make_packet(seq: int, timestamp_ms: int) -> dict[str, object]:
    pulse = 6.0 if seq % 2 == 0 else -6.0
    return {
        "seq": seq,
        "timestamp_ms": timestamp_ms,
        "patches": [
            {"patch_id": "p0", "mean_rgb": [132.0 + pulse, 118.0 + pulse, 95.0], "weight": 1.0},
            {"patch_id": "p1", "mean_rgb": [128.0 - pulse, 116.0 + pulse, 94.0], "weight": 1.0},
            {"patch_id": "p2", "mean_rgb": [130.0 + pulse, 119.0 - pulse, 96.0], "weight": 1.0},
        ],
        "local_quality": {
            "face_present": True,
            "brightness": 0.50,
            "motion_score": 0.03,
            "roi_coverage": 0.95,
        },
    }


class EvaluatorContractTests(unittest.TestCase):
    def test_finalize_writes_artifacts(self) -> None:
        evaluator = StreamEvaluator(fs=12.0)

        for idx in range(80):
            packet = make_packet(seq=idx, timestamp_ms=idx * 83)
            ingest = evaluator.ingest_summary_packet(
                seq=packet["seq"],
                timestamp_ms=packet["timestamp_ms"],
                patches=packet["patches"],
                local_quality=packet["local_quality"],
                payload_size_bytes=len(json.dumps(packet)),
            )
            self.assertEqual(ingest["accepted"], 1)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            result = evaluator.finalize("test_session", out)
            self.assertEqual(result["status"], "complete")
            self.assertTrue((out / "test_session" / "result.json").exists())
            self.assertTrue((out / "test_session" / "packet_trace.jsonl").exists())
            self.assertTrue((out / "test_session" / "quality_timeline.json").exists())
            self.assertTrue((out / "test_session" / "bpm_timeline.json").exists())
            self.assertTrue((out / "test_session" / "coherence_timeline.json").exists())
            self.assertTrue((out / "test_session" / "patch_group_bpm_timeline.json").exists())
            self.assertTrue((out / "test_session" / "patch_group_quality_timeline.json").exists())
            self.assertIn("operational_metrics", result)
            self.assertIn("selected_method", result)
            self.assertIn("corroboration_method", result)
            self.assertIn("coherence_summary", result)
            self.assertEqual(result["coherence_summary"]["decision_groups"], ["forehead", "left_cheek", "right_cheek"])
            self.assertEqual(result["coherence_summary"]["auxiliary_groups"], [])
            self.assertIn("group_summary", result["coherence_summary"])

    def test_finalize_reports_conservative_inconclusive_with_sane_timing(self) -> None:
        evaluator = StreamEvaluator(fs=12.0)

        for idx in range(80):
            packet = make_packet(seq=idx, timestamp_ms=idx * 83)
            evaluator.ingest_summary_packet(
                seq=packet["seq"],
                timestamp_ms=packet["timestamp_ms"],
                patches=packet["patches"],
                local_quality=packet["local_quality"],
                payload_size_bytes=len(json.dumps(packet)),
            )

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            result = evaluator.finalize("test_session_guardrail", out)
            self.assertEqual(result["decision"], "inconclusive")
            self.assertIn(
                result["failure_reasons"][0],
                {"low_recoverability", "missing_chromatic_support", "no_corroboration_method", "insufficient_method_agreement"},
            )
            coherence = result["coherence_summary"]
            self.assertIn("recoverable_groups", coherence)
            self.assertIn("agreeing_groups", coherence)
            self.assertIn("sample_balance", coherence)
            self.assertIn("confidence_balance", coherence)
            self.assertIn("signal_balance", coherence)
            metrics = result["operational_metrics"]
            if metrics["time_to_first_estimate_ms"] is not None:
                self.assertGreaterEqual(metrics["time_to_first_estimate_ms"], 0.0)
                self.assertLess(metrics["time_to_first_estimate_ms"], 20000.0)
            if metrics["time_to_stable_estimate_ms"] is not None:
                self.assertLess(metrics["time_to_stable_estimate_ms"], 20000.0)


if __name__ == "__main__":
    unittest.main()
