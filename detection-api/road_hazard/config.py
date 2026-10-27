"""Explicit configuration, with model paths relative to the config file."""

import json
import math
from pathlib import Path


def probability(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("Confidence must be greater than 0 and at most 1")
    return value


def load_config(path, models=None):
    path = Path(path).resolve()
    with path.open() as stream:
        config = json.load(stream)
    config.setdefault("image_size", 320)
    config.setdefault("device", "cpu")
    config.setdefault("confidence", 0.25)
    config.setdefault("iou", 0.45)
    config.setdefault("max_detections", 100)
    config.setdefault("max_pixels", 24000000)
    probability(config["confidence"])
    probability(config["iou"])
    for key in ("image_size", "max_detections", "max_pixels"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError("{} must be a positive integer".format(key))
    if config["image_size"] % 32:
        raise ValueError("image_size must be a multiple of 32")
    if config.get("backend") not in ("pytorch", "tensorrt"):
        raise ValueError("backend must be pytorch or tensorrt")
    definitions = config.get("models", [])
    names = [item["name"] for item in definitions]
    if len(names) != len(set(names)):
        raise ValueError("Model names must be unique")
    selected = set(models.split(",")) if models else None
    if selected and selected - set(names):
        raise ValueError("Unknown models: {}".format(", ".join(sorted(selected - set(names)))))
    active = []
    for item in definitions:
        enabled = item["name"] in selected if selected is not None else item.get("enabled", True)
        if enabled:
            item = dict(item)
            item["path"] = str((path.parent / item["path"]).resolve())
            if not Path(item["path"]).is_file():
                raise ValueError("Missing model: {}. See README.md for export/build steps.".format(item["path"]))
            if not item.get("labels") or not all(isinstance(label, str) for label in item["labels"]):
                raise ValueError("Each model needs an ordered labels list")
            active.append(item)
    if not active:
        raise ValueError("Enable at least one model")
    config["models"] = active
    return config
