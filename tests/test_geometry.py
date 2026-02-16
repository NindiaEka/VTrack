from utils.geometry import bbox_center, line_side, point_in_polygon


def test_bbox_center():
    assert bbox_center((0, 0, 10, 10)) == (5.0, 5.0)


def test_line_side_sign():
    p1 = (0.0, 0.0)
    p2 = (1.0, 0.0)
    assert line_side(p1, p2, (0.0, 1.0)) > 0
    assert line_side(p1, p2, (0.0, -1.0)) < 0


def test_point_in_polygon_basic():
    triangle = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]
    assert point_in_polygon((5.0, 5.0), triangle) is True
    assert point_in_polygon((20.0, 5.0), triangle) is False
