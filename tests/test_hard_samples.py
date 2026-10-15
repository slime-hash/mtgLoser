from types import SimpleNamespace

import numpy as np

from mcoc_vision.capture import Frame
from mcoc_vision.hard_samples import HardSampleCollector
from mcoc_vision.tracking import Detection, Track


def result(seq, timestamp, detections, status='detected', value=0):
    image = np.full((720, 1280, 3), value, dtype=np.uint8)
    track = Track(detections[-1].box.copy() if detections else None,
                  status, .9, 1, 0.)
    return SimpleNamespace(frame=Frame(seq, timestamp, image), detections=detections, track=track)


def fighter(x, confidence=.9):
    return Detection(np.array([x, 200, x + 180, 600], dtype=float), confidence)


def test_collector_saves_failures_in_review_format_and_skips_duplicate_results(tmp_path):
    collector = HardSampleCollector(tmp_path, 'model.pt', min_interval=.1, max_gap=.5,
                                    novelty=2., hard_novelty=.5, max_frames=10)
    collector.set_enabled(True)
    sample = result(1, 1., [fighter(100)], status='searching')
    assert collector.consider(sample)
    assert not collector.consider(sample)
    assert (tmp_path / 'images' / 'frame_0000_seq_0000001.jpg').is_file()
    assert (tmp_path / 'labels_suggested' / 'frame_0000_seq_0000001.txt').is_file()
    manifest = collector.close()
    assert manifest['total'] == 1
    assert manifest['records'][0]['reasons'] == [
        'fewer_than_two_fighters', 'tracker_searching', 'visual_change'
    ]


def test_collector_keeps_changed_action_pose_even_when_both_fighters_detected(tmp_path):
    collector = HardSampleCollector(tmp_path, 'model.pt', min_interval=.1, max_gap=5.,
                                    novelty=2., hard_novelty=.5, max_frames=10)
    collector.set_enabled(True)
    detections = [fighter(100), fighter(800)]
    assert collector.consider(result(1, 1., detections, value=0))
    assert collector.consider(result(2, 1.2, detections, value=20))
    assert collector.records[-1]['reasons'] == ['visual_change']


def test_collector_paused_and_frame_limit(tmp_path):
    collector = HardSampleCollector(tmp_path, 'model.pt', max_frames=1)
    sample = result(1, 1., [fighter(100)], status='searching')
    assert not collector.consider(sample)
    collector.set_enabled(True)
    assert collector.consider(sample)
    assert not collector.consider(result(2, 2., [fighter(100)], status='searching', value=20))
