from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ultralytics import YOLO


@dataclass
class Detection:
    track_id: int
    cls_id: int
    conf: float
    bbox: Tuple[float, float, float, float]
    label: str


class YoloDetector:
    def __init__(
        self,
        weights: str,
        device: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[List[int]] = None,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.tracker = tracker
        if device:
            self.model.to(device)

    def infer(self, frame) -> List[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            tracker=self.tracker,
            verbose=False,
        )
        if not results:
            return []
        res = results[0]
        boxes = res.boxes
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return []

        detections: List[Detection] = []
        ids = boxes.id
        names = self.model.names if hasattr(self.model, "names") else {}
        for i in range(len(boxes)):
            if ids is None:
                track_id = -1
            else:
                track_id = int(ids[i].item())
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            label = names.get(cls_id, str(cls_id))
            detections.append(
                Detection(
                    track_id=track_id,
                    cls_id=cls_id,
                    conf=conf,
                    bbox=(x1, y1, x2, y2),
                    label=label,
                )
            )
        return detections
