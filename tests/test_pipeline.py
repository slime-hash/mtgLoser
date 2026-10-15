import time
import numpy as np
import pytest

from mcoc_vision.capture import Frame
from mcoc_vision.pipeline import Pipeline
from mcoc_vision.tracking import Detection


def test_worker_propagates_capture_errors():
    class BrokenCapture:
        done = False
        info = {}

        def get(self, *args, **kwargs):
            raise RuntimeError('Disconnected')

    pipeline = Pipeline(BrokenCapture(), None).start()
    pipeline.thread.join(timeout=2)
    with pytest.raises(RuntimeError, match='Inference worker failed'):
        pipeline.get()
    pipeline.close()


def test_result_keeps_exact_frame_and_reports_skips():
    class Capture:
        done = False
        info = {}
        seqs = iter([0, 3, 7])

        def get(self, *args, **kwargs):
            seq = next(self.seqs, None)
            if seq is None:
                self.done = True
                return None
            return Frame(seq, time.perf_counter(), np.zeros((720, 1280, 3), dtype=np.uint8))

    class Detector:
        def detect(self, image):
            return [Detection(np.array([100., 200, 280, 600]), .9),
                    Detection(np.array([800., 200, 980, 600]), .9)], .1

    pipeline = Pipeline(Capture(), Detector()).start()
    pipeline.thread.join(timeout=2)
    result = pipeline.get()
    assert result.frame.seq == 7
    assert result.track.status == 'detected'
    assert result.application_ms >= 0
    report = pipeline.report()
    assert report['processed_frames'] == 3
    assert report['skipped_for_freshness'] == 5
    assert report['source_to_result_latency_ms'] is None
    pipeline.close()
