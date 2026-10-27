import ast
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from road_hazard.api import Application
from road_hazard.backends.tensorrt import letterbox
from road_hazard.config import load_config, probability
from road_hazard.jobs import JobManager
from road_hazard.media import process_image, process_video, write_json
from road_hazard.postprocess import decode_yolov8, nms


class FakeDetector:
    def __init__(self, config=None):
        self.calls = 0

    def predict(self, image, confidence):
        self.calls += 1
        return [{"class_id": 0, "label": "pothole", "detection_type": "pothole",
                 "confidence": 0.9, "bbox_xyxy": [10.0, 10.0, 40.0, 40.0]}]

    def close(self):
        pass


def make_video(path, frames=7):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("Test environment needs MJPEG encoding")
    for index in range(frames):
        writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
    writer.release()


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_python36_syntax(self):
        for source in Path("road_hazard").rglob("*.py"):
            ast.parse(source.read_text(), feature_version=(3, 6))

    def test_invalid_confidence(self):
        for value in (float("nan"), float("inf"), 0, -0.1, 1.1):
            with self.assertRaises(ValueError):
                probability(value)

    def test_config_paths_selection_and_missing_model(self):
        (self.root / "weights.pt").touch()
        config = {"backend": "pytorch", "models": [
            {"name": "pothole", "path": "weights.pt", "labels": ["pothole"]},
            {"name": "obstacle", "path": "missing.pt", "labels": ["cone"], "enabled": False}]}
        write_json(self.root / "config.json", config)
        loaded = load_config(self.root / "config.json")
        self.assertEqual(loaded["models"][0]["path"], str((self.root / "weights.pt").resolve()))
        with self.assertRaisesRegex(ValueError, "Missing model"):
            load_config(self.root / "config.json", "obstacle")
        with self.assertRaisesRegex(ValueError, "Unknown models"):
            load_config(self.root / "config.json", "unknown")

    def test_letterbox_decode_roundtrip(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[:, :, 2] = 255
        tensor, scale, padding = letterbox(frame, 320)
        self.assertEqual(tensor.shape, (1, 3, 320, 320))
        self.assertEqual((scale, padding), (1.6, (0, 80)))
        self.assertEqual(tensor[0, 0, 100, 100], 1)
        raw = np.array([[[160], [160], [160], [80], [0.9]]], dtype=np.float32)
        decoded = decode_yolov8(raw, ["pothole"], "pothole", 0.5, scale, padding, frame.shape)
        np.testing.assert_allclose(decoded[0]["bbox_xyxy"], [50, 25, 150, 75])
        self.assertEqual(decode_yolov8(raw, ["pothole"], "pothole", 0.95, scale, padding, frame.shape), [])
        with self.assertRaises(ValueError):
            decode_yolov8(raw, ["cone", "barrel"], "obstacle", 0.5, scale, padding, frame.shape)

    def test_nms_preserves_distinct_hazards_and_classes(self):
        original = FakeDetector().predict(None, 0.25)[0]
        detections = [original, dict(original, confidence=0.7),
                      dict(original, detection_type="waterlogging", label="water_logging"),
                      dict(original, detection_type="obstacle", label="cone"),
                      dict(original, detection_type="obstacle", label="barrier")]
        self.assertEqual(len(nms(detections)), 4)
        self.assertEqual(len(nms(detections, limit=2)), 2)

    def test_image_writes_real_artifacts_and_rejects_invalid(self):
        source = self.root / "image.png"
        cv2.imwrite(str(source), np.zeros((48, 64, 3), dtype=np.uint8))
        result = process_image(source, self.root / "out", FakeDetector(), 0.25)
        self.assertEqual(result["counts"], {"pothole": 1})
        self.assertIsNotNone(cv2.imread(str(self.root / "out/annotated.jpg")))
        with self.assertRaisesRegex(ValueError, "max_pixels"):
            process_image(source, self.root / "out", FakeDetector(), 0.25, max_pixels=10)
        source.write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "decodable"):
            process_image(source, self.root / "out", FakeDetector(), 0.25)

    def test_video_stride_timing_and_frame_count(self):
        source = self.root / "input.avi"
        make_video(source)
        detector = FakeDetector()
        result = process_video(source, self.root / "out", detector, 0.25, stride=3)
        self.assertEqual((result["frames"], result["inferred_frames"], detector.calls), (7, 3, 3))
        records = [json.loads(line) for line in (self.root / "out/frames.jsonl").read_text().splitlines()]
        self.assertEqual([record["frame_index"] for record in records], [0, 3, 6])
        self.assertEqual([record["timestamp_seconds"] for record in records], [0, 0.3, 0.6])
        cap = cv2.VideoCapture(str(self.root / "out/annotated.avi"))
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 7)
        self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 10)
        cap.read()
        ok, skipped = cap.read()
        self.assertTrue(ok)
        self.assertLess(float(skipped.std()), 1)  # No stale boxes on skipped uniform frame.
        cap.release()
        with self.assertRaises(ValueError):
            process_video(source, self.root / "bad", detector, 0.25, stride=0)


class APITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.config = {"backend": "pytorch", "confidence": 0.25, "max_pixels": 10000,
                       "models": [{"name": "pothole"}]}
        self.manager = JobManager(self.config, self.temporary.name, capacity=1, detector_factory=FakeDetector)
        self.app = Application(self.manager, ["http://frontend:5173"], max_upload_mb=1, min_free_mb=0)
        _, encoded = cv2.imencode(".png", np.zeros((48, 64, 3), dtype=np.uint8))
        self.image = encoded.tobytes()

    def tearDown(self):
        self.manager.close()
        self.temporary.cleanup()

    def call(self, method, path, body=b"", query="", **extra):
        environ = {"REQUEST_METHOD": method, "PATH_INFO": path, "QUERY_STRING": query,
                   "CONTENT_LENGTH": str(len(body)), "wsgi.input": io.BytesIO(body)}
        environ.update(extra)
        captured = {}
        def start_response(status, headers):
            captured.update(status=int(status[:3]), headers=dict(headers))
        response = self.app(environ, start_response)
        try:
            raw = b"".join(response)
        finally:
            if hasattr(response, "close"):
                response.close()
        value = json.loads(raw) if raw and captured["headers"]["Content-Type"] == "application/json" else raw
        return captured, value

    def wait_job(self, job_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = self.manager.get(job_id)
            if job["status"] not in ("queued", "running"):
                return job
            time.sleep(0.01)
        self.fail("Job did not finish")

    def test_image_job_download_delete(self):
        response, created = self.call("POST", "/v1/detect/image", self.image)
        self.assertEqual(response["status"], 202)
        job = self.wait_job(created["id"])
        self.assertEqual(job["status"], "completed")
        response, data = self.call("GET", "/v1/jobs/{}/files/annotated.jpg".format(job["id"]))
        self.assertEqual(response["status"], 200)
        self.assertTrue(data.startswith(b"\xff\xd8"))
        self.assertFalse((self.manager.output / job["id"] / "input.image").exists())
        response, _ = self.call("GET", "/v1/jobs/{}/files/job.json".format(job["id"]))
        self.assertEqual(response["status"], 404)
        response, _ = self.call("DELETE", "/v1/jobs/" + job["id"])
        self.assertEqual(response["status"], 204)
        response, _ = self.call("GET", "/v1/jobs/" + job["id"])
        self.assertEqual(response["status"], 404)

    def test_video_job(self):
        video = self.manager.output / "test.avi"
        make_video(video)
        response, created = self.call("POST", "/v1/detect/video", video.read_bytes(), "stride=3")
        self.assertEqual(response["status"], 202)
        job = self.wait_job(created["id"])
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["result"]["inferred_frames"], 3)

    def test_bad_media_fails_and_releases_slot(self):
        with self.assertLogs("road_hazard.jobs", level="ERROR"):
            _, created = self.call("POST", "/v1/detect/image", b"invalid")
            self.assertEqual(self.wait_job(created["id"])["status"], "failed")
        self.assertTrue(self.manager.reserve())
        self.manager.slots.release()

    def test_validation_and_limits(self):
        for query in ("confidence=nan", "confidence=0", "confidence=1.1", "stride=0", "stride=cat"):
            response, _ = self.call("POST", "/v1/detect/image", self.image, query)
            self.assertEqual(response["status"], 400)
        response, _ = self.call("POST", "/v1/detect/image", self.image, CONTENT_LENGTH="2000000")
        self.assertEqual(response["status"], 413)
        response, _ = self.call("POST", "/v1/detect/image", self.image, CONTENT_LENGTH="")
        self.assertEqual(response["status"], 411)
        response, _ = self.call("POST", "/v1/detect/image", b"x", CONTENT_LENGTH="2")
        self.assertEqual(response["status"], 400)
        self.assertTrue(self.manager.reserve())
        response, _ = self.call("POST", "/v1/detect/image", self.image)
        self.assertEqual(response["status"], 429)
        self.manager.slots.release()

    def test_cors_and_traversal(self):
        response, _ = self.call("OPTIONS", "/v1/detect/image", HTTP_ORIGIN="http://frontend:5173")
        self.assertEqual(response["headers"]["Access-Control-Allow-Origin"], "http://frontend:5173")
        response, _ = self.call("POST", "/v1/detect/image", self.image, HTTP_ORIGIN="http://untrusted")
        self.assertEqual(response["status"], 403)
        response, _ = self.call("GET", "/v1/jobs/../files/job.json")
        self.assertEqual(response["status"], 404)

    def test_restart_marks_interrupted_job_failed(self):
        job_id = "a" * 32
        directory = self.manager.output / job_id
        directory.mkdir()
        write_json(directory / "job.json", {"id": job_id, "status": "running"})
        (directory / "input.video").write_bytes(b"partial")
        self.manager.close()
        self.manager = JobManager(self.config, self.temporary.name, detector_factory=FakeDetector)
        self.assertEqual(self.manager.get(job_id)["status"], "failed")
        self.assertFalse((directory / "input.video").exists())

    def test_models_stay_on_one_thread(self):
        threads = []
        class ThreadDetector(FakeDetector):
            def __init__(self, config):
                super().__init__(config)
                threads.append(threading.get_ident())
            def predict(self, image, confidence):
                threads.append(threading.get_ident())
                return super().predict(image, confidence)
            def close(self):
                threads.append(threading.get_ident())
        self.manager.close()
        self.manager = JobManager(self.config, self.temporary.name, detector_factory=ThreadDetector)
        self.app.manager = self.manager
        _, created = self.call("POST", "/v1/detect/image", self.image)
        self.wait_job(created["id"])
        self.manager.close()
        self.assertEqual(len(threads), 3)
        self.assertEqual(len(set(threads)), 1)


if __name__ == "__main__":
    unittest.main()
