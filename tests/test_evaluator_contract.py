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
        saw_result_event = False

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
            if "result_event" in ingest:
                saw_result_event = True
                self.assertIn(ingest["result_event"]["type"], ["provisional_result", "stable_result"])

        self.assertTrue(saw_result_event)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            result = evaluator.finalize("test_session", out)
            self.assertEqual(result["status"], "complete")
            self.assertTrue((out / "test_session" / "result.json").exists())
            self.assertTrue((out / "test_session" / "packet_trace.jsonl").exists())
            self.assertTrue((out / "test_session" / "quality_timeline.json").exists())
            self.assertTrue((out / "test_session" / "bpm_timeline.json").exists())
            self.assertIn("operational_metrics", result)


if __name__ == "__main__":
    unittest.main()
