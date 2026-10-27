class PyTorchModel:
    def __init__(self, specification, config):
        from ultralytics import YOLO

        self.specification = specification
        self.config = config
        self.model = YOLO(specification["path"], task="detect")
        names = self.model.names
        actual = [names[index] for index in range(len(names))]
        if actual != specification["labels"]:
            raise ValueError("Configured labels do not match {}: {}".format(specification["name"], actual))

    def predict(self, image, confidence):
        result = self.model.predict(image, conf=confidence, iou=self.config["iou"],
                                    imgsz=self.config["image_size"], device=self.config["device"],
                                    max_det=self.config["max_detections"], verbose=False)[0]
        detections = []
        if result.boxes is not None:
            for box in result.boxes.cpu().numpy():
                class_id = int(box.cls[0])
                detections.append({"class_id": class_id,
                                   "label": self.specification["labels"][class_id],
                                   "detection_type": self.specification["name"],
                                   "confidence": float(box.conf[0]),
                                   "bbox_xyxy": box.xyxy[0].tolist()})
        return detections

    def close(self):
        self.model = None
