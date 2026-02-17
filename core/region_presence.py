from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Set, Tuple

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
        dwell_threshold_sec: float | None = None,
    ) -> None:
        self.name = name
        self.region_id = region_id
        self.raw_polygon = polygon
        self.polygon: List[Tuple[float, float]] | None = None
        self.normalized = normalized
        self.current_count = 0
        self.dwell_threshold_sec = dwell_threshold_sec
        self.dwell_state: Dict[int, int] = {}
        self.dwell_alerted: Set[int] = set()
        self.dwell_events: List[dict] = []
        self.max_dwell_sec = 0.0

    def process(self, frame, detections: List[Detection], frame_idx: int, fps: float) -> None:
        self._ensure_polygon(frame)
        if not self.polygon:
            return

        count = 0
        inside_tracked_ids: Set[int] = set()
        for det in detections:
            center = bbox_center(det.bbox)
            if point_in_polygon(center, self.polygon):
                count += 1
                if det.track_id >= 0 and self.dwell_threshold_sec is not None:
                    inside_tracked_ids.add(det.track_id)
                    frames_inside = self.dwell_state.get(det.track_id, 0) + 1
                    self.dwell_state[det.track_id] = frames_inside
                    dwell_sec = frames_inside / fps if fps > 0 else 0.0
                    self.max_dwell_sec = max(self.max_dwell_sec, dwell_sec)
                    if dwell_sec >= self.dwell_threshold_sec and det.track_id not in self.dwell_alerted:
                        self.dwell_events.append(
                            {
                                "frame": frame_idx,
                                "time_sec": frame_idx / fps if fps > 0 else 0.0,
                                "track_id": det.track_id,
                                "dwell_time_sec": dwell_sec,
                            }
                        )
                        self.dwell_alerted.add(det.track_id)

        if self.dwell_threshold_sec is not None:
            for track_id in list(self.dwell_state.keys()):
                if track_id not in inside_tracked_ids:
                    self.dwell_state[track_id] = 0
                    self.dwell_alerted.discard(track_id)

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

        if self.dwell_threshold_sec is not None:
            dwell_text = f"Dwell alerts: {len(self.dwell_events)}"
            y2 = max(text_size[1] + pad, y - 26)
            cv2.putText(frame, dwell_text, (x, y2), font, 0.6, (255, 255, 255), 2)

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
        if self.dwell_threshold_sec is not None:
            summary["dwell"] = {
                "threshold_sec": self.dwell_threshold_sec,
                "alerts": len(self.dwell_events),
                "max_dwell_sec": self.max_dwell_sec,
            }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if self.dwell_events:
            events_path = os.path.join(output_dir, f"{self.name}_dwell_events.csv")
            with open(events_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["frame", "time_sec", "track_id", "dwell_time_sec"],
                )
                writer.writeheader()
                writer.writerows(self.dwell_events)

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
