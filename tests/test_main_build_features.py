from main import build_features
from core.line_cross import LineCrossFeature
from core.region_presence import RegionPresenceFeature


def test_build_features_from_config():
    cfg = {
        "feature": ["linecross", "regionpresence"],
        "lines": [
            {
                "id": 1,
                "bidirectional": True,
                "coords": [{"x": 0.1, "y": 0.2}, {"x": 0.9, "y": 0.2}],
            }
        ],
        "regions": [
            {
                "id": 2,
                "name": "area",
                "coords": [
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.2, "y": 0.1},
                    {"x": 0.2, "y": 0.2},
                ],
            }
        ],
    }

    features = build_features(cfg)

    assert len(features) == 2
    assert any(isinstance(f, LineCrossFeature) for f in features)
    assert any(isinstance(f, RegionPresenceFeature) for f in features)
