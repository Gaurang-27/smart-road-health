# Road hazard detection

Two independent projects, with no shared runtime dependencies:

| Project | Runs on | Purpose |
| --- | --- | --- |
| [`detection-api/`](detection-api/README.md) | Original NVIDIA Jetson Nano, or a desktop | Offline image/video CLI, shared inference, HTTP API |
| [`frontend/`](frontend/README.md) | A separate computer or web host | Upload media, follow jobs, preview and download results |

Each directory is its own local Git repository and can be deployed alone.
No commits or remote repositories have been created. This supplied workspace did
not have Git history. Model binaries are ignored by Git and must be transferred separately.
`samples/` contains original input images and a video for manual testing; neither project
depends on that directory.

Start with [the backend setup](detection-api/README.md) and
[the frontend setup](frontend/README.md). For the original Nano, use the
**JetPack 4 / native TensorRT** instructions, not the desktop pip requirements.

## What changed

- One inference implementation serves the local CLI and asynchronous HTTP jobs.
- The frontend no longer runs on or is served by the Jetson.
- Pothole, waterlogging, and cone/barrel/barrier models run locally. Gemini,
  LangChain, cloud credentials, Streamlit, and training dependencies are gone.
- Removed duplicate API clients, old training/calibration experiments, generated
  output images, the sample GPS dataset, and the tightly coupled map/live-camera
  UI. The focused replacement handles both image and video uploads.
- Results use pixel coordinates and model confidence. Uncalibrated physical-size,
  distance, and severity guesses have been removed. Overlapping *different*
  hazards are preserved; duplicate detections of the same class use NMS.
- Existing `.pt` weights and validated ONNX exports are in `detection-api/models/`. Nano configurations are
  conservative: one active model at 320px, with explicit opt-in to more models.

Legacy source, configuration, experiments, and sample media were backed up before
cleanup to `/private/tmp/capstone-legacy-20261022.tar.gz` (local recovery archive,
outside both projects). That archive contains the old `.env`; keep it private and
copy it somewhere durable if you need it. Temporary storage is not a permanent backup.

The TensorRT engines must be built on the target Nano. Desktop tests cannot certify
GPU compatibility, memory use, accuracy at 320px, or throughput on the board.
