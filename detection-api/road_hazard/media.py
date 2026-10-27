"""Streaming image/video processing shared by CLI and API."""

import json
import math
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import cv2

from .config import probability


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def annotate(image, detections):
    image = image.copy()
    colors = {"pothole": (70, 70, 235), "waterlogging": (230, 160, 30), "obstacle": (30, 185, 245)}
    for item in detections:
        x1, y1, x2, y2 = [int(value) for value in item["bbox_xyxy"]]
        color = colors.get(item["detection_type"], (70, 210, 80))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = "{} {:.0%}".format(item["label"], item["confidence"])
        cv2.putText(image, label, (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return image


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError("Cannot write image: {}".format(path))


def check_frame(frame, max_pixels):
    if frame is None or frame.size == 0:
        raise ValueError("Input is not a decodable image/video")
    if frame.shape[0] * frame.shape[1] > max_pixels:
        raise ValueError("Input exceeds configured max_pixels ({})".format(max_pixels))


def process_image(source, output, detector, confidence, max_pixels=24000000, **unused):
    probability(confidence)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
    check_frame(frame, max_pixels)
    detections = detector.predict(frame, confidence)
    save_image(output / "annotated.jpg", annotate(frame, detections))
    result = {"kind": "image", "width": frame.shape[1], "height": frame.shape[0],
              "detections": detections, "counts": dict(Counter(item["detection_type"] for item in detections)),
              "elapsed_seconds": round(time.monotonic() - started, 3),
              "artifacts": {"image": "annotated.jpg", "json": "result.json"}}
    write_json(output / "result.json", result)
    return result


def process_video(source, output, detector, confidence, stride=1, max_pixels=24000000, progress=None):
    probability(confidence)
    if type(stride) is not int or stride < 1:
        raise ValueError("stride must be a positive integer")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    writer = None
    started = time.monotonic()
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open video")
        declared_pixels = cap.get(cv2.CAP_PROP_FRAME_WIDTH) * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if declared_pixels > max_pixels:
            raise ValueError("Input exceeds configured max_pixels ({})".format(max_pixels))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0 or fps > 1000:
            raise ValueError("Video has invalid FPS; transcode to a constant-frame-rate video first")
        total_value = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total = int(total_value) if math.isfinite(total_value) and total_value > 0 else 0
        count, inferred = 0, 0
        observations = Counter()
        with (output / "frames.jsonl").open("w") as records:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                check_frame(frame, max_pixels)
                height, width = frame.shape[:2]
                if writer is None:
                    dimensions = (width, height)
                    # MJPEG is supported by JetPack's system OpenCV without extra codecs.
                    writer = cv2.VideoWriter(str(output / "annotated.avi"),
                                             cv2.VideoWriter_fourcc(*"MJPG"), fps, dimensions)
                    if not writer.isOpened():
                        raise RuntimeError("OpenCV cannot create MJPEG/AVI output; check video I/O support")
                if (width, height) != dimensions:
                    raise ValueError("Video resolution changed mid-stream")
                if count % stride == 0:
                    detections = detector.predict(frame, confidence)
                    inferred += 1
                    observations.update(item["detection_type"] for item in detections)
                    records.write(json.dumps({"frame_index": count, "timestamp_seconds": count / fps,
                                              "detections": detections}, allow_nan=False) + "\n")
                    frame = annotate(frame, detections)
                    if count == 0:
                        save_image(output / "preview.jpg", frame)
                if count % 30 == 0 and shutil.disk_usage(str(output)).free < 64 * 1024 * 1024:
                    raise RuntimeError("Insufficient disk space while writing video; free space and retry")
                # Skipped frames stay unannotated: never hold stale boxes on moving footage.
                writer.write(frame)
                count += 1
                if progress and (count == 1 or count % 30 == 0):
                    progress(count, total)
        if count == 0:
            raise ValueError("Video contains no decodable frames")
        # Known frame counts let us detect most truncated/corrupt input files.
        if total and count < total - 1:
            raise ValueError("Video decoding stopped early ({}/{} frames)".format(count, total))
        writer.release()
        writer = None
        if not (output / "annotated.avi").stat().st_size:
            raise RuntimeError("Video encoder produced an empty file")
        result = {"kind": "video", "width": dimensions[0], "height": dimensions[1], "fps": fps,
                  "frames": count, "inferred_frames": inferred, "stride": stride,
                  "detection_observations": dict(observations),
                  "elapsed_seconds": round(time.monotonic() - started, 3),
                  "artifacts": {"video": "annotated.avi", "preview": "preview.jpg",
                                "frames": "frames.jsonl", "json": "result.json"}}
        write_json(output / "result.json", result)
        return result
    finally:
        cap.release()
        if writer is not None:
            writer.release()
