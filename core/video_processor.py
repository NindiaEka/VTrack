from __future__ import annotations

import os
import time
from typing import List

import cv2

from core.detector import Detection, YoloDetector
from core.feature_base import FeatureBase


class VideoProcessor:
    def __init__(
        self,
        detector: YoloDetector,
        output_dir: str,
        output_video: str,
        preview_enabled: bool = False,
        preview_scale: float = 1.0,
        features: List[FeatureBase] | None = None,
    ) -> None:
        self.detector = detector
        self.output_dir = output_dir
        self.output_video = output_video
        self.preview_enabled = preview_enabled
        self.preview_scale = preview_scale
        self.features = features or []

    def process(self, input_path: str) -> str:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, self.output_video)

        ext = os.path.splitext(output_path)[1].lower()
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if ext == ".mp4" else "XVID"))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Failed to create output video writer")

        frame_idx = 0
        last_log = time.time()
        log_every = 30
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detector.infer(frame)
            for feature in self.features:
                feature.process(frame, detections, frame_idx, fps)

            self._draw_detections(frame, detections)
            for feature in self.features:
                feature.render(frame)
            self._maybe_preview(frame)

            writer.write(frame)
            if frame_idx % log_every == 0:
                now = time.time()
                if now - last_log >= 1.0:
                    if total_frames > 0:
                        percent = (frame_idx / total_frames) * 100.0
                        print(f"Processed frame {frame_idx}/{total_frames} ({percent:.1f}%)")
                    else:
                        print(f"Processed frame {frame_idx}")
                    last_log = now
            frame_idx += 1

        cap.release()
        writer.release()

        if self.preview_enabled:
            cv2.destroyAllWindows()

        for feature in self.features:
            feature.finalize(self.output_dir, fps)

        return output_path

    def _draw_detections(self, frame, detections: List[Detection]) -> None:
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            track_text = f"ID {det.track_id}" if det.track_id >= 0 else "ID ?"
            label = f"{det.label} {det.conf:.2f} {track_text}"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        count_text = f"Count: {len(detections)}"
        text_size, _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = max(0, frame.shape[1] - text_size[0] - 10)
        y = max(0, frame.shape[0] - 10)
        cv2.putText(frame, count_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def _maybe_preview(self, frame) -> None:
        if not self.preview_enabled:
            return
        display = frame
        if 0 < self.preview_scale < 1.0:
            display = cv2.resize(
                frame,
                (0, 0),
                fx=self.preview_scale,
                fy=self.preview_scale,
                interpolation=cv2.INTER_AREA,
            )
        cv2.imshow("VTrack Preview", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise RuntimeError("Preview stopped by user")
