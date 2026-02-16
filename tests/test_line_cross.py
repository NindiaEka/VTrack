import numpy as np

from core.detector import Detection
from core.line_cross import LineCrossFeature


def _det(track_id: int, center_x: float, center_y: float) -> Detection:
    size = 2.0
    return Detection(
        track_id=track_id,
        cls_id=0,
        conf=0.9,
        bbox=(center_x - size, center_y - size, center_x + size, center_y + size),
        label="obj",
    )


def test_line_cross_counts_bidirectional():
    feature = LineCrossFeature(((0.0, 0.0), (10.0, 0.0)), name="line")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    feature.process(frame, [_det(1, 5.0, -1.0)], frame_idx=0, fps=30.0)
    feature.process(frame, [_det(1, 5.0, 1.0)], frame_idx=1, fps=30.0)

    assert feature.counts["A_to_B"] == 1
    assert feature.counts["B_to_A"] == 0
    assert len(feature.events) == 1


def test_line_cross_respects_bidirectional_flag():
    feature = LineCrossFeature(
        ((0.0, 0.0), (10.0, 0.0)),
        name="line",
        bidirectional=False,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    feature.process(frame, [_det(2, 5.0, 1.0)], frame_idx=0, fps=30.0)
    feature.process(frame, [_det(2, 5.0, -1.0)], frame_idx=1, fps=30.0)

    assert feature.counts["A_to_B"] == 0
    assert feature.counts["B_to_A"] == 0
