from mcoc_vision.screen_capture import intersection_area


def test_intersection_area_for_left_monitor_and_window():
    assert intersection_area((-1920, 0, 0, 1080), (-1920, 0, 0, 1080)) == 1920 * 1080
    assert intersection_area((-100, 50, 100, 150), (0, 0, 1920, 1080)) == 100 * 100
    assert intersection_area((-500, 0, -100, 500), (0, 0, 1920, 1080)) == 0
