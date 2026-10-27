"""Small WSGI API compatible with the Nano's system Python 3.6.

Uploads are raw request bodies, streamed to disk. No multipart buffering,
web framework, PyTorch import, or frontend hosting is needed on the Nano.
"""

import json
import logging
import mimetypes
import shutil
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs
from wsgiref.util import FileWrapper

from .config import probability

LOG = logging.getLogger(__name__)


class APIError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


class Application:
    def __init__(self, manager, origins=(), max_upload_mb=256, min_free_mb=512):
        self.manager = manager
        self.origins = set(origins)
        self.max_bytes = max_upload_mb * 1024 * 1024
        self.min_free = min_free_mb * 1024 * 1024

    def __call__(self, environ, start_response):
        headers = [("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff")]
        origin = environ.get("HTTP_ORIGIN")
        if origin and origin in self.origins:
            headers += [("Access-Control-Allow-Origin", origin), ("Vary", "Origin"),
                        ("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"),
                        ("Access-Control-Allow-Headers", "Content-Type"),
                        ("Access-Control-Max-Age", "600")]
        try:
            if origin and origin not in self.origins:
                raise APIError(403, "Origin is not allowed; configure --cors-origin")
            if environ["REQUEST_METHOD"] == "OPTIONS":
                status, value = 204, None
            else:
                status, value = self.route(environ)
            if isinstance(value, Path):
                stream = value.open("rb")
                headers.extend([("Content-Type", mimetypes.guess_type(value.name)[0] or "application/octet-stream"),
                                ("Content-Length", str(value.stat().st_size)),
                                ("Content-Disposition", 'inline; filename="{}"'.format(value.name))])
                start_response("200 OK", headers)
                return FileWrapper(stream, 65536)
        except APIError as exc:
            status, value = exc.status, {"error": exc.message}
        except FileNotFoundError:
            status, value = 404, {"error": "Job or artifact not found"}
        except (ValueError, KeyError) as exc:
            status, value = 400, {"error": str(exc)}
        except Exception:
            LOG.exception("API request failed")
            status, value = 500, {"error": "Internal server error; check server logs"}
        body = b"" if value is None else json.dumps(value, allow_nan=False).encode("utf-8")
        headers.extend([("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
        start_response("{} {}".format(status, HTTPStatus(status).phrase), headers)
        return [body]

    def route(self, environ):
        method, path = environ["REQUEST_METHOD"], environ.get("PATH_INFO", "")
        if method == "GET" and path == "/health":
            return 200, {"status": "ready", "backend": self.manager.config["backend"],
                         "models": [item["name"] for item in self.manager.config["models"]],
                         "max_upload_mb": self.max_bytes // (1024 * 1024)}
        if method == "POST" and path in ("/v1/detect/image", "/v1/detect/video"):
            return self.upload(environ, path.rsplit("/", 1)[1])
        parts = path.strip("/").split("/")
        if len(parts) >= 3 and parts[:2] == ["v1", "jobs"]:
            job = self.manager.get(parts[2])
            if len(parts) == 3 and method == "GET":
                return 200, job
            if len(parts) == 3 and method == "DELETE":
                if job["status"] in ("queued", "running"):
                    raise APIError(409, "Cannot delete an active job")
                self.manager.delete(parts[2])
                return 204, None
            if len(parts) == 5 and parts[3] == "files" and method == "GET":
                if job["status"] != "completed":
                    raise APIError(409, "Job has not completed")
                if parts[4] not in job["result"]["artifacts"].values():
                    raise APIError(404, "Unknown artifact")
                return 200, self.manager.output / parts[2] / parts[4]
        raise APIError(404, "Unknown endpoint")

    def upload(self, environ, kind):
        query = parse_qs(environ.get("QUERY_STRING", ""))
        confidence = probability(query.get("confidence", [self.manager.config["confidence"]])[0])
        stride = int(query.get("stride", [1])[0])
        if stride < 1 or stride > 10000:
            raise APIError(400, "stride must be between 1 and 10000")
        if environ.get("HTTP_TRANSFER_ENCODING"):
            raise APIError(411, "Send a Content-Length; chunked uploads are not supported")
        try:
            length = int(environ.get("CONTENT_LENGTH", ""))
        except ValueError:
            raise APIError(411, "Content-Length is required")
        if length <= 0:
            raise APIError(400, "Upload is empty")
        if length > self.max_bytes:
            raise APIError(413, "Upload exceeds configured size limit")
        if shutil.disk_usage(str(self.manager.output)).free < length + self.min_free:
            raise APIError(507, "Not enough free disk space; delete old jobs")
        if not self.manager.reserve():
            raise APIError(429, "Detection queue is full; retry after the current job completes")
        directory = None
        try:
            job_id, directory = self.manager.create(kind)
            source = directory / ("input.image" if kind == "image" else "input.video")
            with source.open("wb") as stream:
                remaining = length
                while remaining:
                    chunk = environ["wsgi.input"].read(min(65536, remaining))
                    if not chunk:
                        raise APIError(400, "Upload ended before Content-Length bytes were received")
                    stream.write(chunk)
                    remaining -= len(chunk)
            job = self.manager.submit(job_id, source, kind, confidence, stride)
            return 202, dict(job, status_url="/v1/jobs/" + job_id)
        except Exception:
            if directory is not None:
                shutil.rmtree(str(directory), ignore_errors=True)
            self.manager.slots.release()
            raise
