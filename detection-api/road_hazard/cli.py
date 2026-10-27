import argparse
from contextlib import redirect_stdout
import json
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path

from .config import load_config, probability


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parser():
    root = argparse.ArgumentParser(description="Offline road hazard detection and HTTP API")
    commands = root.add_subparsers(dest="command")
    for name in ("image", "video", "serve", "check"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, help="Path to a desktop or Jetson JSON config")
        command.add_argument("--models", help="Comma-separated model names to enable (overrides config)")
        if name in ("image", "video"):
            command.add_argument("input", type=Path)
            command.add_argument("--output", "-o", type=Path, default=Path("outputs"))
            command.add_argument("--confidence", type=float)
            if name == "video":
                command.add_argument("--stride", type=positive, default=1, help="Infer every Nth frame; other frames remain unannotated")
        if name == "serve":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=positive, default=8000)
            command.add_argument("--output", "-o", type=Path, default=Path("outputs/api"))
            command.add_argument("--cors-origin", action="append", default=[])
            command.add_argument("--max-upload-mb", type=positive, default=256)
            command.add_argument("--queue-size", type=positive, default=2)
    return root


def main(argv=None):
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    if args.command is None:
        arg_parser.print_help()
        return 2
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        config = load_config(args.config, args.models)
        # OpenCV reads this at import time; enforce the pixel budget before decode.
        os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(config["max_pixels"])
        if args.command == "serve":
            from .api import Application
            from .jobs import JobManager
            from .server import serve

            if args.port > 65535:
                raise ValueError("port must be at most 65535")
            manager = JobManager(config, args.output, args.queue_size)
            try:
                serve(Application(manager, args.cors_origin, args.max_upload_mb), args.host, args.port)
            finally:
                manager.close()
            return 0
        from .detector import Detector

        confidence = probability(args.confidence if getattr(args, "confidence", None) is not None else config["confidence"])
        if args.command != "check" and not args.input.is_file():
            raise ValueError("Input file does not exist: {}".format(args.input))
        with redirect_stdout(sys.stderr):
            detector = Detector(config)
        try:
            if args.command == "check":
                import numpy as np

                detector.predict(np.zeros((config["image_size"], config["image_size"], 3), dtype=np.uint8), confidence)
                print(json.dumps({"status": "ready", "backend": config["backend"],
                                  "models": [item["name"] for item in config["models"]]}))
                return 0
            from .media import process_image, process_video

            destination = args.output.resolve() / uuid.uuid4().hex
            destination.mkdir(parents=True)
            try:
                processor = process_image if args.command == "image" else process_video
                result = processor(args.input, destination, detector, confidence,
                                   stride=getattr(args, "stride", 1), max_pixels=config["max_pixels"])
            except BaseException:
                shutil.rmtree(str(destination))
                raise
            print(json.dumps({"output_dir": str(destination), "result": result}, indent=2))
        finally:
            detector.close()
        return 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        return 1
