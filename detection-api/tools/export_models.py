"""Desktop-only YOLOv8 export to static ONNX opset 12 for TensorRT 8.2."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/desktop.json")
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--models", help="Comma-separated model names")
    args = parser.parse_args()

    import onnx
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect
    from road_hazard.config import load_config

    config = load_config(args.config, args.models)
    if config["backend"] != "pytorch":
        parser.error("Export needs a pytorch config pointing to .pt weights")
    args.output.mkdir(parents=True, exist_ok=True)
    size = config["image_size"]
    for specification in config["models"]:
        yolo = YOLO(specification["path"], task="detect")
        names = [yolo.names[index] for index in range(len(yolo.names))]
        if names != specification["labels"]:
            raise ValueError("Incorrect labels for " + specification["name"])
        model = yolo.model.cpu().float().eval().fuse()
        for module in model.modules():
            if isinstance(module, Detect):
                module.export, module.format, module.dynamic = True, "onnx", False
        dummy = torch.zeros(1, 3, size, size)
        destination = args.output / (specification["name"] + ".onnx")
        with torch.no_grad():
            model(dummy)
            model(dummy)
            # Explicit legacy exporter: newer torch releases default to Dynamo,
            # which can change the opset and the old Nano parser compatibility.
            torch.onnx.export(model, dummy, str(destination), opset_version=12,
                              input_names=["images"], output_names=["output0"],
                              dynamic_axes=None, do_constant_folding=True, dynamo=False)
        exported = onnx.load(str(destination))
        onnx.checker.check_model(exported)
        dimensions = [dim.dim_value for dim in exported.graph.output[0].type.tensor_type.shape.dim]
        if len(dimensions) != 3 or dimensions[:2] != [1, 4 + len(names)] or dimensions[2] <= 0:
            raise ValueError("Unexpected YOLOv8 output: {}".format(dimensions))
        with destination.with_suffix(".json").open("w") as stream:
            json.dump({"name": specification["name"], "labels": names, "image_size": size,
                       "opset": 12, "input_shape": [1, 3, size, size],
                       "output_shape": dimensions}, stream, indent=2)
        print("Exported {} {}".format(destination, dimensions))


if __name__ == "__main__":
    main()
