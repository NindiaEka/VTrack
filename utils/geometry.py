from __future__ import annotations

from typing import Tuple


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def line_side(p1: Tuple[float, float], p2: Tuple[float, float], p: Tuple[float, float]) -> float:
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])


def point_in_polygon(point: Tuple[float, float], polygon: Tuple[Tuple[float, float], ...] | list) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
        if intersects:
            inside = not inside
        j = i
    return inside
