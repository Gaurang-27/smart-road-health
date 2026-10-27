from .postprocess import nms


class Detector:
    def __init__(self, config):
        self.config = config
        self.models = []
        if config["backend"] == "tensorrt":
            from .backends.tensorrt import TensorRTModel as Model
        else:
            from .backends.pytorch import PyTorchModel as Model
        try:
            for specification in config["models"]:
                self.models.append(Model(specification, config))
        except Exception:
            self.close()
            raise

    def predict(self, image, confidence):
        detections = []
        for model in self.models:
            detections.extend(model.predict(image, confidence))
        return nms(detections, self.config["iou"], self.config["max_detections"])

    def close(self):
        for model in self.models:
            model.close()
        self.models = []
