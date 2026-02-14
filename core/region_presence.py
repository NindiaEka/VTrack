from __future__ import annotations

import json
import os
from typing import List, Tuple

import cv2
import numpy as np

from core.detector import Detection
from core.feature_base import FeatureBase
from utils.geometry import bbox_center, point_in_polygon


class RegionPresenceFeature(FeatureBase):
    def __init__(
        self,
        name: str,
        region_id: int | None,
        polygon: List[Tuple[float, float]],
        normalized: bool = False,
    ) -> None:
        self.name = name
        self.region_id = region_id
        self.raw_polygon = polygon
        self.polygon: List[Tuple[float, float]] | None = None
        self.normalized = normalized
        self.current_count = 0

    def process(self, frame, detections: List[Detection], frame_idx: int, fps: float) -> None:
        self._ensure_polygon(frame)
        if not self.polygon:
            return

        count = 0
        for det in detections:
            center = bbox_center(det.bbox)
            if point_in_polygon(center, self.polygon):
                count += 1

        self.current_count = count

    def render(self, frame) -> None:
        self._ensure_polygon(frame)
        if not self.polygon:
            return

        pts = np.array([(int(x), int(y)) for x, y in self.polygon], dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        text = f"Region {self.name}: {self.current_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, scale, thickness)
        pad = 6
        x = max(0, frame.shape[1] - text_size[0] - 10)
        y = max(text_size[1] + pad, frame.shape[0] - 50)
        top_left = (x - pad, y - text_size[1] - pad)
        bottom_right = (x + text_size[0] + pad, y + pad)
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness)

    def finalize(self, output_dir: str, fps: float) -> None:
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, f"{self.name}_summary.json")
        summary = {
            "feature": "region_presence",
            "region": {
                "name": self.name,
                "id": self.region_id,
                "normalized": self.normalized,
            },
            "last_frame_count": self.current_count,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def is_inside(self, point: Tuple[float, float]) -> bool:
        if not self.polygon:
            return False
        return point_in_polygon(point, self.polygon)

    def _ensure_polygon(self, frame) -> None:
        if self.polygon is not None:
            return
        if frame is None:
            return
        if self.normalized:
            height, width = frame.shape[:2]
            self.polygon = [(x * width, y * height) for x, y in self.raw_polygon]
        else:
            self.polygon = list(self.raw_polygon)
