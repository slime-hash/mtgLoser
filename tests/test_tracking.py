import numpy as np
import pytest

from mcoc_vision.tracking import Detection, OpponentTracker


def fighter(x, confidence=.9):
    return Detection(np.array([x, 200, x + 180, 600], dtype=float), confidence)


def acquire(tracker):
    for t in (1., 1.02, 1.04):
        result = tracker.update([fighter(100), fighter(800)], t, (720, 1280))
    return result


def test_acquire_and_update_always_select_right_fighter():
    tracker = OpponentTracker()
    assert acquire(tracker).box[0] == 800
    result = tracker.update([fighter(1050), fighter(790)], 1.06, (720, 1280))
    assert result.status == 'detected'
    assert result.box[0] > 1000


def test_single_player_never_initializes_opponent():
    tracker = OpponentTracker()
    for t in range(10):
        assert tracker.update([fighter(100)], float(t), (720, 1280)).box is None


def test_dropout_is_bounded_and_does_not_jump_to_player():
    tracker = OpponentTracker()
    acquire(tracker)
    assert tracker.update([fighter(100)], 1.08, (720, 1280)).status == 'predicted'
    result = tracker.update([fighter(100)], 1.2, (720, 1280))
    assert result.status == 'lost'
    assert result.box is None
    result = tracker.update([fighter(100)], 1.8, (720, 1280))
    assert result.status == 'searching'


def test_ambiguous_candidates_do_not_confirm_detection():
    tracker = OpponentTracker()
    acquire(tracker)
    result = tracker.update([fighter(795), fighter(805)], 1.06, (720, 1280))
    assert result.status == 'ambiguous'


def test_reacquisition_has_new_track_id():
    tracker = OpponentTracker()
    assert acquire(tracker).track_id == 1
    tracker.update([], 2., (720, 1280))
    for t in (2.02, 2.04, 2.06):
        result = tracker.update([fighter(100), fighter(800)], t, (720, 1280))
    assert result.track_id == 2


def test_non_monotonic_time_rejected():
    tracker = OpponentTracker()
    acquire(tracker)
    with pytest.raises(ValueError):
        tracker.update([], 1., (720, 1280))


def test_right_side_rule_prevents_geometric_identity_swap():
    tracker = OpponentTracker()
    assert acquire(tracker).box[0] == 800
    # The player now occupies the old opponent location while the opponent
    # lunges right. Geometry alone would choose the player and swap identity.
    result = tracker.update([fighter(790), fighter(1020)], 1.06, (720, 1280))
    assert result.status == 'detected'
    assert result.box[0] > 950
