# Model artifacts

The existing `pothole.pt`, `waterlogging.pt`, and `obstacle.pt` were preserved here.
They are ignored by Git. Transfer weights/exports separately when splitting or
deploying this repository; do not assume `git clone` includes them.

Validated static 320px ONNX exports and their JSON metadata are also present
locally. These are ready to transfer for the Nano engine-build step. No TensorRT
engines are included, because those must be built and verified on the target board.

Desktop: `.pt` weights. Nano: native `.engine` files built **on the target Nano**
from the supplied static ONNX export. The Nano does not need the `.pt` files.
See the parent README for export/build commands and model labels.
