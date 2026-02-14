from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]


def _draw_instructions(canvas, lines: List[str]) -> None:
    y = 20
    for text in lines:
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 24


def collect_line(frame) -> List[Point] | None:
    points: List[Point] = []
    window = "ROI Setup - Line"

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = frame.copy()
        for pt in points:
            cv2.circle(canvas, pt, 4, (0, 255, 255), -1)
        if len(points) == 2:
            cv2.line(canvas, points[0], points[1], (0, 255, 255), 2)

        _draw_instructions(
            canvas,
            [
                "Left click: set point A then B",
                "Enter/Space: confirm   R: reset   Q/Esc: cancel",
            ],
        )

        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):
            if len(points) == 2:
                break
        elif key in (ord("r"), ord("R")):
            points.clear()
        elif key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(window)
            return None

    cv2.destroyWindow(window)
    return points


def collect_polygon(frame) -> List[Point] | None:
    points: List[Point] = []
    window = "ROI Setup - Region"

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = frame.copy()
        for pt in points:
            cv2.circle(canvas, pt, 3, (255, 0, 0), -1)
        if len(points) >= 2:
            poly = np.array(points, dtype=np.int32)
            cv2.polylines(canvas, [poly], isClosed=False, color=(255, 0, 0), thickness=2)

        _draw_instructions(
            canvas,
            [
                "Left click: add vertex   Right click: undo",
                "Enter/Space: confirm (>=3)   R: reset   Q/Esc: cancel",
            ],
        )

        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):
            if len(points) >= 3:
                break
        elif key in (ord("r"), ord("R")):
            points.clear()
        elif key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(window)
            return None

    cv2.destroyWindow(window)
    return points


def normalize_points(points: List[Point], frame_shape) -> List[Tuple[float, float]]:
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return []
    return [(x / width, y / height) for x, y in points]
