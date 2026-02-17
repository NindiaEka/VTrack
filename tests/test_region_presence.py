import numpy as np

from core.detector import Detection
from core.region_presence import RegionPresenceFeature


def _det(center_x: float, center_y: float) -> Detection:
    size = 2.0
    return Detection(
        track_id=1,
        cls_id=0,
        conf=0.9,
        bbox=(center_x - size, center_y - size, center_x + size, center_y + size),
        label="obj",
    )


def test_region_presence_counts_inside():
    feature = RegionPresenceFeature(
        name="region",
        region_id=1,
        polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        normalized=False,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    detections = [_det(5.0, 5.0), _det(15.0, 5.0)]
    feature.process(frame, detections, frame_idx=0, fps=30.0)

    assert feature.current_count == 1


def test_region_presence_normalized_polygon():
    feature = RegionPresenceFeature(
        name="region",
        region_id=None,
        polygon=[(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)],
        normalized=True,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    feature.process(frame, [_det(20.0, 20.0)], frame_idx=0, fps=30.0)

    assert feature.current_count == 1
    assert feature.is_inside((20.0, 20.0)) is True
    assert feature.is_inside((90.0, 90.0)) is False


def test_region_presence_dwell_alert():
    feature = RegionPresenceFeature(
        name="region",
        region_id=1,
        polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        normalized=False,
        dwell_threshold_sec=1.0,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    fps = 2.0

    feature.process(frame, [_det(5.0, 5.0)], frame_idx=0, fps=fps)
    assert len(feature.dwell_events) == 0

    feature.process(frame, [_det(5.0, 5.0)], frame_idx=1, fps=fps)
    assert len(feature.dwell_events) == 1
