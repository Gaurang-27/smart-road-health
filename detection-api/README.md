# Road hazard detection API and CLI

An independent Python project for **offline** pothole, waterlogging, and obstacle
detection. The CLI and API share the same models, postprocessing, and media
pipeline. No Node.js, frontend build, cloud account, or API server is needed to
run the CLI.

## Desktop setup (Python 3.10–3.12)

Run these commands from this directory:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-desktop.txt
python -m pip install --no-deps -e .
road-hazard check --config configs/desktop.json
```

Place the three original `.pt` files in `models/` if they were not transferred
with this directory. The existing weights have been preserved there locally.
`desktop.json` uses CPU and all three models. Change `device` to `"0"` for a
desktop CUDA GPU, or use `--models pothole` to select a subset. Config paths are
resolved relative to their JSON file, not the current directory.

## Image and video CLI

```sh
road-hazard image /path/to/road.jpg --config configs/desktop.json --output outputs
road-hazard video /path/to/road.mp4 --config configs/desktop.json --output outputs --stride 5
```

`python -m road_hazard` is equivalent to the installed `road-hazard` command.
Each run creates a unique subdirectory and prints its absolute path and summary
as JSON. Existing results are never overwritten. Use `--confidence 0.35` to change
the score threshold and `--models pothole,waterlogging,obstacle` to select models.

| Input | Saved outputs |
| --- | --- |
| Image | `annotated.jpg`, `result.json` with all boxes, labels and confidence |
| Video | `annotated.avi`, `preview.jpg`, `frames.jsonl`, `result.json` |

Videos are decoded, inferred, annotated, and written one frame at a time. The
default stride is 1; stride 5 infers frames 0, 5, 10, etc. Other frames remain
unmarked, rather than carrying stale boxes. Video outputs retain the input FPS
and frame order, use MJPEG/AVI, and **do not include audio**. The browser shows a
first-frame preview and offers the video as a download; AVI playback is not
assumed. For variable-frame-rate sources, transcode to constant frame rate first.

`frames.jsonl` has one JSON object per inferred frame with its zero-based
`frame_index`, `timestamp_seconds`, and detections. Summary video counts are
**detection observations**, not unique hazards or tracked objects. CLI failures
exit nonzero and remove the incomplete output directory.

## Original Jetson Nano: JetPack 4 / TensorRT 8.2

This is the original Nano (2GB/4GB), **not Orin Nano**. NVIDIA lists JetPack 4.6.6
as the final JetPack 4 release for this hardware. JetPack 4.6.1 introduced
TensorRT 8.2 and CUDA 10.2. Use the vendor runtime installed with JetPack 4.6.x,
including its system Python 3.6 bindings. Do not install desktop PyTorch wheels,
current pip TensorRT, or the desktop requirements on this device.

References: [NVIDIA JetPack archive](https://developer.nvidia.com/embedded/jetpack-archive),
[JetPack 4.6.1 components](https://developer.nvidia.com/embedded/jetpack-sdk-461),
[NVIDIA JetPack 4 end-of-life announcement](https://forums.developer.nvidia.com/t/announcing-end-of-life-for-nvidia-jetpack-4-with-the-release-of-jetpack-4-6-6/314409).

### 1. Export models on the desktop

In the desktop environment above:

```sh
python -m pip install -r requirements-export.txt
python tools/export_models.py --config configs/desktop.json --output models
```

The exporter creates `.onnx` and metadata `.json` files. It uses YOLOv8 detection,
batch 1, static 320×320 input, FP32 input/output, opset 12, and **no embedded NMS**.
The outputs are `[1, 4 + number_of_classes, 2100]`. These restrictions are
intentional for the older TensorRT parser. Do not substitute dynamic exports,
YOLOv5 objectness outputs, segmentation models, or Ultralytics metadata-prefixed
`.engine` files. Export parameters follow the
[Ultralytics export interface](https://docs.ultralytics.com/modes/export/), with
an explicit legacy PyTorch exporter to retain opset 12 on newer desktop PyTorch.

The current local `models/` directory already contains these validated exports
for all three supplied weights; rerun export when weights or image size change.

The supplied weights are YOLOv8m-sized (~25.8M parameters **each**), not tiny
models. Running three concurrently may exhaust a 2GB Nano. Smaller retrained
weights are likely necessary for high throughput; there is no real-time FPS claim.

### 2. Transfer only the backend to the Nano

Copy this directory to the Nano, for example `/opt/detection-api`. Include the
ONNX files and configs, exclude desktop `.venv/`, `.pt` weights, and outputs.
Neither `frontend/` nor `samples/` is required.

Install the system packages on the Nano:

```sh
sudo apt-get update
sudo apt-get install python3-numpy python3-opencv python3-pycuda python3-libnvinfer
cd /opt/detection-api
/usr/bin/python3 -c 'import cv2, numpy, tensorrt, pycuda.driver; print(tensorrt.__version__)'
```

The core and HTTP API use Python 3.6-compatible syntax and the standard library.
You can run them directly from this directory without pip or a virtualenv. A
virtualenv, if used, must use the system Python and `--system-site-packages` so
the JetPack bindings remain visible.

### 3. Build engines on that Nano

```sh
sh tools/build_engine.sh models/pothole.onnx 256
# Build these only if you intend to enable them:
sh tools/build_engine.sh models/waterlogging.onnx 256
sh tools/build_engine.sh models/obstacle.onnx 256
```

This invokes JetPack's `/usr/src/tensorrt/bin/trtexec`, enables FP16 and caps the
builder workspace at 256 MiB. Total build/runtime memory is higher than that
workspace limit. Build one engine at a time with other applications closed.
Engines are specific to the hardware and TensorRT version: rebuild after a
runtime change, and do not build them on a desktop GPU. Existing engines are
never overwritten by the script.

### 4. Check and run

```sh
/usr/bin/python3 -m road_hazard check --config configs/jetson.json
/usr/bin/python3 -m road_hazard image /data/road.jpg --config configs/jetson.json --output /data/outputs
/usr/bin/python3 -m road_hazard video /data/road.mp4 --config configs/jetson.json --output /data/outputs --stride 5
/usr/bin/python3 -m road_hazard serve --config configs/jetson.json --host 0.0.0.0 --output /data/api-results --cors-origin http://localhost:5173
```

The Nano config enables **potholes only** by default at 320px, with a 1080p input
pixel limit. After checking memory with `tegrastats`, enable more models using
`--models pothole,waterlogging,obstacle` or their `enabled` fields. The `/health`
response lists the models actually loaded. Changing `image_size` requires
re-exporting/rebuilding **every** enabled engine at the same size. The runtime
rejects mismatched shapes or labels rather than returning invalid boxes.

Validate a representative image and short video on the physical Nano before a
long run. Compare outputs with the desktop and measure memory/latency. The
TensorRT path cannot be hardware-tested from the development Mac.

## HTTP API

Desktop example:

```sh
road-hazard serve --config configs/desktop.json --host 0.0.0.0 --output outputs/api --cors-origin http://localhost:5173
```

The default bind address is `127.0.0.1`; `0.0.0.0` enables network access.
Set `--cors-origin` to the frontend's exact **browser origin** (scheme, host,
port; no trailing slash), repeating the flag for multiple allowed origins.
If a different computer opens `http://frontend-host:5173`, allow that origin,
not `localhost`. Reverse proxies preserve the browser origin too.

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/health` | Readiness after models load; active models and upload limit |
| POST | `/v1/detect/image?confidence=0.25` | Raw image body → HTTP 202 job |
| POST | `/v1/detect/video?confidence=0.25&stride=5` | Raw video body → HTTP 202 job |
| GET | `/v1/jobs/{id}` | `queued`, `running`, `completed`, or `failed` and progress/result |
| GET | `/v1/jobs/{id}/files/{filename}` | Download a completed job's advertised artifact |
| DELETE | `/v1/jobs/{id}` | Delete a completed/failed job and its saved files |

Uploads use the **raw file body, not multipart/form-data**. Send Content-Length
(browsers set it automatically for File/Blob bodies). Chunked requests are not
accepted directly; use a buffering reverse proxy if your client needs them.

```sh
curl http://JETSON_IP:8000/health
curl -H 'Content-Type: image/jpeg' --data-binary @road.jpg 'http://JETSON_IP:8000/v1/detect/image?confidence=0.25'
curl -H 'Content-Type: video/mp4' --data-binary @road.mp4 'http://JETSON_IP:8000/v1/detect/video?stride=5'
# Replace JOB_ID with the id returned above:
curl http://JETSON_IP:8000/v1/jobs/JOB_ID
curl -o annotated.avi http://JETSON_IP:8000/v1/jobs/JOB_ID/files/annotated.avi
curl -X DELETE http://JETSON_IP:8000/v1/jobs/JOB_ID
```

A completed image job includes:

```json
{
  "id": "<32-character job id>",
  "kind": "image",
  "status": "completed",
  "result": {
    "kind": "image",
    "width": 1280,
    "height": 720,
    "detections": [{"class_id": 0, "label": "pothole", "detection_type": "pothole", "confidence": 0.91, "bbox_xyxy": [100, 200, 300, 350]}],
    "counts": {"pothole": 1},
    "elapsed_seconds": 1.2,
    "artifacts": {"image": "annotated.jpg", "json": "result.json"}
  }
}
```

Every input uses the same confidence threshold. NMS suppresses overlapping boxes
only for the same hazard/class. Coordinates refer to the original decoded image.
Physical dimensions, depth, severity, and GPS cannot be inferred reliably from
these boxes and are not fabricated.

One worker owns all inference/CUDA state. Default capacity is **two outstanding
uploads/jobs total**, not two GPU workers; additional submissions receive 429.
Use `--queue-size` to adjust. Uploads stream to disk, default to a 256 MiB maximum
(`--max-upload-mb`), and require 512 MiB of free disk headroom. Videos can produce
much larger MJPEG outputs; leave ample disk space. The writer stops if free disk
falls below 64 MiB. Uploaded inputs are removed after each job. Finished outputs
persist until deleted; there is no automatic retention policy. Interrupted jobs
are marked failed on restart, and jobs cannot be cancelled through the API.

This small WSGI server is intended for a trusted LAN. It has no authentication
or TLS and is not an Internet-facing server. For remote use, put an authenticated
HTTPS reverse proxy/VPN in front. The old Nano software stack is end-of-life.
Do not run multiple server processes against the same output directory; each
process would load its own models and restart reconciliation would conflict.

For boot startup, customize the user, paths, and CORS origin in
`deploy/road-hazard.service`, then install it with systemd. Frontend hosting is
documented in that project's own README.

## Tests

From a desktop environment with NumPy/OpenCV:

```sh
python -m unittest discover -s tests -v
```

Tests cover image/video outputs, frame sampling/timing, NMS/letterboxing, config
validation, jobs, queue capacity, CORS, invalid media, downloads/deletion,
restart handling, and inference thread ownership. The syntax compatibility test
runs on modern desktop Python; GPU validation uses `road-hazard check` on Nano.
