import numpy as np

from mcoc_vision.labeling import (box_edit_mode, box_from_yolo, display_to_image_point, edit_box,
                                  fighter_suggestions, yolo_row)
from mcoc_vision.tracking import Detection


def detection(box, confidence):
    return Detection(np.array(box, dtype=float), confidence)


def test_filters_hud_and_collapses_nested_duplicate_to_larger_box():
    detections = [
        detection([1000, 5, 1080, 90], .9),       # HUD portrait
        detection([990, 170, 1248, 650], .66),    # right fighter
        detection([202, 182, 532, 592], .44),     # full left fighter
        detection([302, 352, 525, 591], .45),     # nested partial duplicate
    ]
    result = fighter_suggestions(detections, (720, 1280, 3))
    assert len(result) == 2
    assert any(np.array_equal(item.box, detections[1].box) for item in result)
    assert any(np.array_equal(item.box, detections[2].box) for item in result)


def test_yolo_conversion_is_normalized():
    row = yolo_row(np.array([100, 200, 300, 600]), (800, 1000, 3))
    assert row == (0, .2, .5, .2, .5)
    assert np.allclose(box_from_yolo(row, (800, 1000, 3)), [100, 200, 300, 600])


def test_mouse_coordinates_scale_with_resizable_review_window():
    assert display_to_image_point((576, 324), (1152, 648), (1080, 1920, 3)) == (960, 540)
    assert display_to_image_point((5000, -10), (1152, 648), (1080, 1920, 3)) == (1919, 0)


def test_box_edit_hit_testing_prefers_edges_and_supports_move():
    box = np.array([100, 200, 400, 600], dtype=float)
    assert box_edit_mode((102, 202), box, 5) == 'top_left'
    assert box_edit_mode((250, 400), box, 5) == 'move'
    assert box_edit_mode((500, 400), box, 5) is None


def test_box_can_move_and_resize_without_leaving_image():
    shape = (720, 1280, 3)
    box = np.array([100, 200, 400, 600], dtype=float)
    assert np.array_equal(edit_box(box, 'move', (200, 300), (250, 350), shape),
                          [150, 250, 450, 650])
    assert np.array_equal(edit_box(box, 'right', (400, 300), (500, 300), shape),
                          [100, 200, 500, 600])
    assert np.array_equal(edit_box(box, 'top_left', (100, 200), (-50, -20), shape),
                          [0, 0, 400, 600])
