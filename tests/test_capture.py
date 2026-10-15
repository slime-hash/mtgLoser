import threading
import time

import cv2
import numpy as np

from mcoc_vision.capture import LatestCapture


def test_slow_consumer_gets_latest_and_capture_releases(monkeypatch):
    released = threading.Event()

    class FakeCamera:
        def __init__(self, *args):
            self.seq = 0

        def isOpened(self):
            return True

        def set(self, *args):
            return False

        def get(self, prop):
            return 60

        def getBackendName(self):
            return 'fake'

        def read(self):
            time.sleep(.002)
            self.seq += 1
            return True, np.full((8, 8, 3), self.seq % 255, dtype=np.uint8)

        def release(self):
            released.set()

    monkeypatch.setattr(cv2, 'VideoCapture', FakeCamera)
    capture = LatestCapture(0).start()
    try:
        first = capture.get(timeout=2)
        time.sleep(.06)
        last = capture.get(first.seq, timeout=2)
        assert last.seq > first.seq + 1
        assert last.received_at > first.received_at
        assert not capture.info['buffer_size_request_accepted']
    finally:
        capture.close()
    assert released.is_set()
