from __future__ import annotations

from typing import List

from core.detector import Detection


class FeatureBase:
    name: str

    def process(self, frame, detections: List[Detection], frame_idx: int, fps: float) -> None:
        raise NotImplementedError

    def render(self, frame) -> None:
        return None

    def finalize(self, output_dir: str, fps: float) -> None:
        return None
