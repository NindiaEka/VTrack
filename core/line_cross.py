from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Tuple

import cv2

from core.detector import Detection
from core.feature_base import FeatureBase
from utils.geometry import bbox_center, line_side


class LineCrossFeature(FeatureBase):
    def __init__(
        self,
        line: Tuple[Tuple[float, float], Tuple[float, float]],
        name: str,
        bidirectional: bool = True,
        normalized: bool = False,
        orientation: str | None = None,
        direction: str | None = None,
    ) -> None:
        self.name = name
        self.raw_p1, self.raw_p2 = line
        self.p1: Tuple[float, float] | None = None
        self.p2: Tuple[float, float] | None = None
        self.bidirectional = bidirectional
        self.normalized = normalized
        self.orientation = orientation
        self.direction = direction
        self.line_axis_value: float | None = None
        self.last_side_by_id: Dict[int, float] = {}
        self.counts = {"A_to_B": 0, "B_to_A": 0}
        self.events: List[dict] = []

    def process(self, frame, detections: List[Detection], frame_idx: int, fps: float) -> None:
        self._ensure_line(frame)
        if self.p1 is None or self.p2 is None:
            return
        for det in detections:
            if det.track_id < 0:
                continue
            center = bbox_center(det.bbox)
            side = self._compute_side(center)
            if abs(side) < 1e-6:
                continue
            prev_side = self.last_side_by_id.get(det.track_id)
            if prev_side is not None and prev_side * side < 0:
                direction = "A_to_B" if prev_side < 0 and side > 0 else "B_to_A"
                if not self.bidirectional and direction != "A_to_B":
                    self.last_side_by_id[det.track_id] = side
                    continue
                self.counts[direction] += 1
                self.events.append(
                    {
                        "frame": frame_idx,
                        "time_sec": frame_idx / fps if fps > 0 else 0.0,
                        "track_id": det.track_id,
                        "direction": direction,
                        "line_direction": self.direction,
                        "line_orientation": self.orientation,
                        "label": det.label,
                    }
                )
            self.last_side_by_id[det.track_id] = side

    def render(self, frame) -> None:
        self._ensure_line(frame)
        if self.p1 is None or self.p2 is None:
            return
        p1 = (int(self.p1[0]), int(self.p1[1]))
        p2 = (int(self.p2[0]), int(self.p2[1]))
        cv2.line(frame, p1, p2, (0, 255, 255), 2)
        text = f"A->B: {self.counts['A_to_B']}  B->A: {self.counts['B_to_A']}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def finalize(self, output_dir: str, fps: float) -> None:
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, f"{self.name}_summary.json")
        events_path = os.path.join(output_dir, f"{self.name}_events.csv")

        summary = {
            "feature": self.name,
            "counts": self.counts,
            "total_crossings": len(self.events),
            "line": {
                "direction": self.direction,
                "orientation": self.orientation,
                "bidirectional": self.bidirectional,
                "normalized": self.normalized,
            },
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        with open(events_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame",
                    "time_sec",
                    "track_id",
                    "direction",
                    "line_direction",
                    "line_orientation",
                    "label",
                ],
            )
            writer.writeheader()
            writer.writerows(self.events)

    def _ensure_line(self, frame) -> None:
        if self.p1 is not None and self.p2 is not None:
            return
        if frame is None:
            return
        if self.normalized:
            height, width = frame.shape[:2]
            self.p1 = (self.raw_p1[0] * width, self.raw_p1[1] * height)
            self.p2 = (self.raw_p2[0] * width, self.raw_p2[1] * height)
        else:
            self.p1 = (self.raw_p1[0], self.raw_p1[1])
            self.p2 = (self.raw_p2[0], self.raw_p2[1])

        if self.orientation == "horizontal":
            self.line_axis_value = (self.p1[1] + self.p2[1]) / 2.0
        elif self.orientation == "vertical":
            self.line_axis_value = (self.p1[0] + self.p2[0]) / 2.0

    def _compute_side(self, center: Tuple[float, float]) -> float:
        if self.p1 is None or self.p2 is None:
            return 0.0
        if self.orientation == "horizontal" and self.line_axis_value is not None:
            diff = center[1] - self.line_axis_value
            if self.direction == "upward":
                diff = -diff
            return diff
        if self.orientation == "vertical" and self.line_axis_value is not None:
            diff = center[0] - self.line_axis_value
            if self.direction == "leftward":
                diff = -diff
            return diff
        return line_side(self.p1, self.p2, center)
