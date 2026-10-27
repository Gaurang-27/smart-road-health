"""Native TensorRT 8.x runtime for JetPack 4 / Python 3.6.

Consumes raw trtexec engines, not Ultralytics' metadata-prefixed engine files.
Only static, batch-one, NCHW YOLOv8 detection exports are accepted.
"""

import cv2
import numpy as np

from ..postprocess import decode_yolov8


def letterbox(image, size):
    height, width = image.shape[:2]
    scale = min(size / float(width), size / float(height))
    resized = (int(round(width * scale)), int(round(height * scale)))
    left = (size - resized[0]) // 2
    top = (size - resized[1]) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + resized[1], left:left + resized[0]] = cv2.resize(image, resized)
    tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, scale, (left, top)


class TensorRTModel:
    def __init__(self, specification, config):
        import tensorrt as trt
        import pycuda.driver as cuda

        if not trt.__version__.startswith("8."):
            raise RuntimeError("This backend targets TensorRT 8.x / JetPack 4, found " + trt.__version__)
        self.specification, self.config = specification, config
        self.cuda = cuda
        self.buffers = []
        self.context = self.engine = self.runtime = self.stream = None
        cuda.init()
        self.cuda_context = cuda.Device(int(config["device"])).retain_primary_context()
        self.cuda_context.push()
        try:
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.logger)
            with open(specification["path"], "rb") as stream:
                self.engine = self.runtime.deserialize_cuda_engine(stream.read())
            if self.engine is None:
                raise RuntimeError("Cannot deserialize engine; rebuild with trtexec on this Nano")
            if self.engine.has_implicit_batch_dimension or self.engine.num_bindings != 2:
                raise ValueError("Engine must have one explicit-batch input and one output")
            self.input_index = next(i for i in range(2) if self.engine.binding_is_input(i))
            self.output_index = 1 - self.input_index
            size = config["image_size"]
            if tuple(self.engine.get_binding_shape(self.input_index)) != (1, 3, size, size):
                raise ValueError("Engine input must be [1,3,{0},{0}]; check image_size/export".format(size))
            output_shape = tuple(self.engine.get_binding_shape(self.output_index))
            if len(output_shape) != 3 or output_shape[:2] != (1, 4 + len(specification["labels"])) or output_shape[2] <= 0:
                raise ValueError("Unexpected output shape {}; use the supplied export tool".format(output_shape))
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Cannot allocate TensorRT execution context; reduce active models")
            self.stream = cuda.Stream()
            for index in range(2):
                shape = tuple(self.engine.get_binding_shape(index))
                dtype = trt.nptype(self.engine.get_binding_dtype(index))
                if dtype not in (np.float32, np.float16):
                    raise ValueError("Engine bindings must use FP32 or FP16")
                host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
                device = cuda.mem_alloc(host.nbytes)
                self.buffers.append((host, device, shape))
        except Exception:
            self._release()
            self.cuda_context.pop()
            self.cuda_context.detach()
            self.cuda_context = None
            raise
        self.cuda_context.pop()

    def predict(self, image, confidence):
        tensor, scale, padding = letterbox(image, self.config["image_size"])
        self.cuda_context.push()
        try:
            host, device, _ = self.buffers[self.input_index]
            np.copyto(host, tensor.ravel(), casting="unsafe")
            self.cuda.memcpy_htod_async(device, host, self.stream)
            if not self.context.execute_async_v2([int(buf[1]) for buf in self.buffers], self.stream.handle):
                raise RuntimeError("TensorRT inference failed")
            output, output_device, shape = self.buffers[self.output_index]
            self.cuda.memcpy_dtoh_async(output, output_device, self.stream)
            self.stream.synchronize()
            return decode_yolov8(output.reshape(shape), self.specification["labels"],
                                 self.specification["name"], confidence, scale, padding, image.shape)
        finally:
            self.cuda_context.pop()

    def _release(self):
        self.context = None
        for _, device, _ in self.buffers:
            device.free()
        self.buffers = []
        self.stream = self.engine = self.runtime = None

    def close(self):
        if self.cuda_context is not None:
            self.cuda_context.push()
            try:
                self._release()
            finally:
                self.cuda_context.pop()
                self.cuda_context.detach()
                self.cuda_context = None
