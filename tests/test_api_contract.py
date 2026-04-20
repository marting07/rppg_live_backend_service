from __future__ import annotations

import math
import unittest

from fastapi.testclient import TestClient

from app.main import app


def make_summary_payload(seq: int, timestamp_ms: int) -> dict[str, object]:
    pulse = math.sin(2.0 * math.pi * 1.2 * (timestamp_ms / 1000.0))
    return {
        "type": "sample_summary_chunk",
        "seq": seq,
        "timestamp_ms": timestamp_ms,
        "patches": [
            {"patch_id": "p0", "mean_rgb": [130.0 + pulse * 6.0, 118.0 + pulse * 4.0, 95.0], "weight": 1.0},
            {"patch_id": "p1", "mean_rgb": [132.0 + pulse * 5.0, 120.0 + pulse * 3.0, 96.0], "weight": 1.0},
            {"patch_id": "p2", "mean_rgb": [128.0 + pulse * 4.0, 117.0 + pulse * 2.0, 94.0], "weight": 1.0},
        ],
        "local_quality": {
            "face_present": True,
            "brightness": 0.48,
            "motion_score": 0.04,
            "roi_coverage": 0.92,
        },
        "passive_artifacts": {
            "moire_score": 0.12,
            "brightness_banding_score": 0.18,
            "reflectance_variation": 0.05,
            "flat_contrast_score": 0.22,
            "global_brightness_drift": 0.01,
        },
    }


class ApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_session_lifecycle(self) -> None:
        create = self.client.post("/v1/liveness/sessions", json={"device_id": "test", "preferred_method": "pos"})
        self.assertEqual(create.status_code, 200)
        body = create.json()
        session_id = body["session_id"]
        token = body["access_token"]
        self.assertEqual(body["capture_config"]["transport_format"], "patch_rgb_v1")

        seen_quality = False
        with self.client.websocket_connect(f"/v1/liveness/sessions/{session_id}/stream?token={token}") as ws:
            for i in range(24):
                ws.send_json(make_summary_payload(seq=i, timestamp_ms=i * 83))
                ack = ws.receive_json()
                quality = ws.receive_json()
                self.assertEqual(ack["type"], "ack")
                self.assertEqual(quality["type"], "quality_feedback")
                seen_quality = True
            ws.send_json({"type": "end_stream"})
            complete = ws.receive_json()
            self.assertEqual(complete.get("type"), "complete")

        self.assertTrue(seen_quality)

        result = self.client.get(f"/v1/liveness/sessions/{session_id}/result")
        self.assertEqual(result.status_code, 200)
        rj = result.json()
        self.assertEqual(rj["status"], "complete")
        self.assertIn(rj["decision"], ["live", "not_live", "inconclusive"])
        self.assertIn("operational_metrics", rj)
        self.assertIn("replay_summary", rj)
        self.assertIn("score", rj["replay_summary"])

        diagnostics = self.client.get(f"/v1/liveness/sessions/{session_id}/diagnostics")
        self.assertEqual(diagnostics.status_code, 200)
        dj = diagnostics.json()
        self.assertEqual(dj["session_id"], session_id)
        self.assertGreaterEqual(dj["packets_received"], 24)
        self.assertIn("latest_coherence", dj)
        self.assertIn("latest_replay", dj)
        self.assertIn("latest_patch_group_bpm", dj)
        self.assertIn("latest_patch_group_quality", dj)

    def test_missing_token_rejected(self) -> None:
        create = self.client.post("/v1/liveness/sessions", json={})
        session_id = create.json()["session_id"]
        with self.client.websocket_connect(f"/v1/liveness/sessions/{session_id}/stream") as ws:
            error = ws.receive_json()
            self.assertEqual(error["type"], "error")
            self.assertEqual(error["code"], "invalid_token")


if __name__ == "__main__":
    unittest.main()
