from __future__ import annotations

import argparse
import os
import sys

import cv2
import yaml

from core.detector import OpenVinoDetector, YoloDetector
from core.line_cross import LineCrossFeature
from core.region_presence import RegionPresenceFeature
from core.video_processor import VideoProcessor
from utils.roi_editor import collect_line, collect_polygon, normalize_points


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_feature_list(cfg: dict) -> list:
    feature_list = cfg.get("feature") or []
    if isinstance(feature_list, str):
        feature_list = [feature_list]
    return feature_list


def save_config(path: str, cfg: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def grab_first_frame(input_video: str):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video}")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("Failed to read first frame for ROI setup")
    return frame


def prepare_roi(cfg: dict, input_video: str, config_path: str) -> dict:
    roi_cfg = cfg.get("roi", {})
    setup_on_start = bool(roi_cfg.get("setup_on_start", False))

    cfg = dict(cfg)
    feature_list = get_feature_list(cfg)
    need_line = "linecross" in feature_list and not cfg.get("lines")
    need_region = "regionpresence" in feature_list and not cfg.get("regions")

    if setup_on_start and (need_line or need_region):
        frame = grab_first_frame(input_video)
        roi_output: dict = {"lines": [], "regions": []}

        if need_line:
            points = collect_line(frame)
            if points is None:
                raise RuntimeError("ROI setup canceled for line")
            norm_points = normalize_points(points, frame.shape)
            roi_output["lines"].append(
                {
                    "id": 1,
                    "bidirectional": True,
                    "coords": [
                        {"x": float(norm_points[0][0]), "y": float(norm_points[0][1])},
                        {"x": float(norm_points[1][0]), "y": float(norm_points[1][1])},
                    ],
                }
            )

        if need_region:
            points = collect_polygon(frame)
            if points is None:
                raise RuntimeError("ROI setup canceled for region")
            norm_points = normalize_points(points, frame.shape)
            roi_output["regions"].append(
                {
                    "id": 1,
                    "name": "region_1",
                    "coords": [{"x": float(x), "y": float(y)} for x, y in norm_points],
                }
            )

        if roi_output.get("lines"):
            cfg["lines"] = roi_output.get("lines")
        if roi_output.get("regions"):
            cfg["regions"] = roi_output.get("regions")

        save_config(config_path, cfg)

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VTrack video inference")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


def build_detector(cfg: dict) -> YoloDetector:
    model_cfg = cfg.get("model", {})
    weights = model_cfg.get("weights")
    if not weights:
        raise ValueError("model.weights is required in config.yaml")
    backend = str(model_cfg.get("backend", "ultralytics")).lower()
    detector_cls = YoloDetector
    if backend in {"openvino", "ov"}:
        detector_cls = OpenVinoDetector
    elif backend not in {"ultralytics", "pytorch", "pt"}:
        raise ValueError(f"Unsupported model.backend: {backend}")

    return detector_cls(
        weights=weights,
        device=model_cfg.get("device"),
        conf=float(model_cfg.get("conf", 0.25)),
        iou=float(model_cfg.get("iou", 0.45)),
        classes=model_cfg.get("classes"),
        tracker=model_cfg.get("tracker", "bytetrack.yaml"),
    )


def build_features(cfg: dict) -> list:
    features = []
    feature_list = get_feature_list(cfg)

    if "linecross" in feature_list:
        for idx, line_cfg in enumerate(cfg.get("lines", []) or []):
            coords = line_cfg.get("coords", [])
            if len(coords) != 2:
                continue
            p1 = (float(coords[0].get("x", 0)), float(coords[0].get("y", 0)))
            p2 = (float(coords[1].get("x", 0)), float(coords[1].get("y", 0)))
            normalized = all(0.0 <= v <= 1.0 for v in [p1[0], p1[1], p2[0], p2[1]])
            name = f"line_{line_cfg.get('id', idx + 1)}"
            bidirectional = bool(line_cfg.get("bidirectional", True))
            orientation = line_cfg.get("orientation")
            direction = line_cfg.get("direction")
            features.append(
                LineCrossFeature(
                    (p1, p2),
                    name=name,
                    bidirectional=bidirectional,
                    normalized=normalized,
                    orientation=orientation,
                    direction=direction,
                )
            )

    if "regionpresence" in feature_list:
        for idx, region_cfg in enumerate(cfg.get("regions", []) or []):
            coords = region_cfg.get("coords", [])
            if len(coords) < 3:
                continue
            polygon = [(float(p.get("x", 0)), float(p.get("y", 0))) for p in coords]
            normalized = all(0.0 <= v <= 1.0 for pt in polygon for v in pt)
            region_id = region_cfg.get("id")
            name = region_cfg.get("name") or f"region_{region_id or idx + 1}"
            features.append(
                RegionPresenceFeature(
                    name=name,
                    region_id=region_id,
                    polygon=polygon,
                    normalized=normalized,
                )
            )
    return features


def build_processor(cfg: dict, detector: YoloDetector) -> VideoProcessor:
    output_dir = cfg.get("output_dir", "outputs")
    output_video = cfg.get("output_video", "annotated.mp4")
    preview_cfg = cfg.get("preview", {})
    preview_enabled = bool(preview_cfg.get("enabled", False))
    preview_scale = float(preview_cfg.get("scale", 1.0))
    features = build_features(cfg)
    return VideoProcessor(
        detector=detector,
        output_dir=output_dir,
        output_video=output_video,
        preview_enabled=preview_enabled,
        preview_scale=preview_scale,
        features=features,
    )


def run_pipeline(cfg: dict, config_path: str) -> str:
    input_video = cfg.get("input_video")
    if not input_video:
        raise ValueError("input_video is required in config.yaml")
    cfg = prepare_roi(cfg, input_video, config_path)
    detector = build_detector(cfg)
    processor = build_processor(cfg, detector)
    print(f"Input video: {input_video}")
    print(f"Output dir: {cfg.get('output_dir', 'outputs')}")
    print(f"Weights: {cfg.get('model', {}).get('weights')}")
    return processor.process(input_video)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    output_path = run_pipeline(cfg, args.config)
    print(f"Annotated video: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
