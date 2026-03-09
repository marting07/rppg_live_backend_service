from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

import cv2  # type: ignore
import numpy as np

from app.evaluator import StreamEvaluator


class EvaluatorContractTests(unittest.TestCase):
    def test_finalize_writes_artifact(self) -> None:
        evaluator = StreamEvaluator(fs=12.0)
        roi = np.full((96, 96, 3), 128, dtype=np.uint8)
        ok, enc = cv2.imencode(".jpg", roi)
        assert ok
        b64 = base64.b64encode(enc.tobytes()).decode("ascii")

        for _ in range(40):
            evaluator.ingest_roi_payload(b64)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            result = evaluator.finalize("test_session", out)
            self.assertEqual(result["status"], "complete")
            self.assertTrue((out / "test_session" / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
