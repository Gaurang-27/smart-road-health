"""A single inference thread owns the models and CUDA context."""

import json
import logging
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .detector import Detector
from .media import process_image, process_video, write_json

LOG = logging.getLogger(__name__)


class JobManager:
    def __init__(self, config, output, capacity=2, detector_factory=Detector):
        self.config = config
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.slots = threading.BoundedSemaphore(capacity)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.detector = None
        # Mark interrupted jobs explicitly; don't pretend they finished after a restart.
        for path in self.output.glob("*/job.json"):
            with path.open() as stream:
                job = json.load(stream)
            if job.get("status") in ("queued", "running"):
                job.update(status="failed", error="Server restarted before completion")
                write_json(path, job)
                for upload in path.parent.glob("input.*"):
                    upload.unlink()
        try:
            self.detector = self.executor.submit(detector_factory, config).result()
        except Exception:
            self.executor.shutdown(wait=True)
            raise

    def reserve(self):
        return self.slots.acquire(blocking=False)

    def create(self, kind):
        job_id = uuid.uuid4().hex
        directory = self.output / job_id
        directory.mkdir()
        return job_id, directory

    def submit(self, job_id, source, kind, confidence, stride):
        job = {"id": job_id, "kind": kind, "status": "queued", "created_at": time.time()}
        write_json(source.parent / "job.json", job)
        self.executor.submit(self._run, job, source, confidence, stride)
        return dict(job, status="queued")

    def _run(self, job, source, confidence, stride):
        path = source.parent / "job.json"
        try:
            job["status"] = "running"
            write_json(path, job)

            def progress(frame, total):
                job["progress"] = {"frames_processed": frame, "total_frames": total or None}
                write_json(path, job)

            processor = process_image if job["kind"] == "image" else process_video
            result = processor(source, source.parent, self.detector, confidence,
                               stride=stride, max_pixels=self.config["max_pixels"], progress=progress)
            job.update(status="completed", result=result, finished_at=time.time())
        except Exception as exc:
            LOG.exception("Detection job %s failed", job["id"])
            job.update(status="failed", error=str(exc), finished_at=time.time())
            for name in ("annotated.jpg", "annotated.avi", "preview.jpg", "frames.jsonl", "result.json"):
                artifact = source.parent / name
                if artifact.exists():
                    artifact.unlink()
        finally:
            try:
                if source.exists():
                    source.unlink()
                write_json(path, job)
            finally:
                self.slots.release()

    def get(self, job_id):
        if len(job_id) != 32 or any(char not in "0123456789abcdef" for char in job_id):
            raise FileNotFoundError("Unknown job")
        with (self.output / job_id / "job.json").open() as stream:
            return json.load(stream)

    def delete(self, job_id):
        job = self.get(job_id)
        if job["status"] in ("queued", "running"):
            raise ValueError("Cannot delete an active job")
        shutil.rmtree(str(self.output / job_id))

    def close(self):
        # Run destruction on the same worker that constructed the models.
        if self.detector is not None:
            self.executor.submit(self.detector.close).result()
            self.detector = None
        self.executor.shutdown(wait=True)
