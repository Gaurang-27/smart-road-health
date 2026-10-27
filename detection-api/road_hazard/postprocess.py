"""YOLOv8 raw output decoding and class-aware NMS (no torch dependency)."""

import numpy as np


def nms(detections, threshold=0.45, limit=100):
    pending = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    kept = []
    while pending and len(kept) < limit:
        best = pending.pop(0)
        kept.append(best)
        a = np.asarray(best["bbox_xyxy"], dtype=np.float32)
        if not pending:
            break
        b = np.asarray([item["bbox_xyxy"] for item in pending], dtype=np.float32)
        wh = np.maximum(0, np.minimum(a[2:], b[:, 2:]) - np.maximum(a[:2], b[:, :2]))
        intersection = wh[:, 0] * wh[:, 1]
        area_a = np.prod(np.maximum(0, a[2:] - a[:2]))
        area_b = np.prod(np.maximum(0, b[:, 2:] - b[:, :2]), axis=1)
        iou = intersection / np.maximum(area_a + area_b - intersection, 1e-9)
        pending = [item for item, overlap in zip(pending, iou)
                   if (item["detection_type"], item["label"]) !=
                   (best["detection_type"], best["label"]) or overlap <= threshold]
    return kept


def decode_yolov8(output, labels, kind, confidence, scale, padding, image_shape):
    expected = 4 + len(labels)
    raw = np.asarray(output)
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != expected:
        raise ValueError("Expected YOLOv8 output [1, {}, anchors], got {}. Export with nms=False.".format(expected, raw.shape))
    rows = raw[0].T
    rows = rows[np.isfinite(rows).all(axis=1)]
    classes = rows[:, 4:].argmax(axis=1)
    scores = rows[np.arange(len(rows)), classes + 4]
    indices = np.flatnonzero(scores >= confidence)
    # Bound CPU NMS cost for noisy/low-threshold predictions.
    indices = indices[np.argsort(scores[indices])[::-1][:3000]]
    height, width = image_shape[:2]
    detections = []
    for index in indices:
        cx, cy, w, h = rows[index, :4]
        box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        box -= np.array([padding[0], padding[1], padding[0], padding[1]])
        box /= scale
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        class_id = int(classes[index])
        detections.append({"class_id": class_id, "label": labels[class_id],
                           "detection_type": kind, "confidence": float(scores[index]),
                           "bbox_xyxy": [float(value) for value in box]})
    return detections
