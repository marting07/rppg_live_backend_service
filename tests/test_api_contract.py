from __future__ import annotations

import base64
import unittest

import cv2  # type: ignore
import numpy as np
from fastapi.testclient import TestClient

from app.main import app


def make_roi_b64() -> str:
    roi = np.full((96, 96, 3), 120, dtype=np.uint8)
    ok, enc = cv2.imencode(".jpg", roi)
    assert ok
    return base64.b64encode(enc.tobytes()).decode("ascii")


class ApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_session_lifecycle(self) -> None:
        create = self.client.post("/v1/liveness/sessions", json={"device_id": "test"})
        self.assertEqual(create.status_code, 200)
        body = create.json()
        session_id = body["session_id"]
        token = body["access_token"]

        with self.client.websocket_connect(f"/v1/liveness/sessions/{session_id}/stream?token={token}") as ws:
            for i in range(30):
                ws.send_json(
                    {
                        "type": "roi_frame_chunk",
                        "session_id": session_id,
                        "seq": i,
                        "timestamp_ms": i * 83,
                        "image_format": "jpeg",
                        "image_bytes_b64": make_roi_b64(),
                    }
                )
                _ = ws.receive_json()
            ws.send_json({"type": "end_stream"})
            complete = ws.receive_json()
            self.assertEqual(complete.get("type"), "complete")

        result = self.client.get(f"/v1/liveness/sessions/{session_id}/result")
        self.assertEqual(result.status_code, 200)
        rj = result.json()
        self.assertEqual(rj["status"], "complete")
        self.assertIn(rj["decision"], ["live", "not_live", "inconclusive"])

    def test_missing_token_rejected(self) -> None:
        create = self.client.post("/v1/liveness/sessions", json={})
        session_id = create.json()["session_id"]
        with self.assertRaises(Exception):
            with self.client.websocket_connect(f"/v1/liveness/sessions/{session_id}/stream"):
                pass


if __name__ == "__main__":
    unittest.main()
