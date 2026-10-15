"""Single-slot capture: inference never drains an accumulated Python queue."""
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    seq: int
    received_at: float  # monotonic, after read(); NOT source/render timestamp
    image: np.ndarray
    source_time: float | None = None


class LatestCapture:
    def __init__(self, source, backend=cv2.CAP_DSHOW, width=1280, height=720, fps=60):
        self.source = source
        self.is_file = isinstance(source, str) and Path(source).is_file()
        self.backend = cv2.CAP_ANY if self.is_file else backend
        self.width, self.height, self.fps = width, height, fps
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.latest = None
        self.error = None
        self.done = False
        self.info = {}
        self.thread = threading.Thread(target=self._run, daemon=True, name='capture')

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.source, self.backend)
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open video source: {self.source}')
            if not self.is_file:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
            buffer_supported = cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            rate = cap.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(rate) or rate <= 0:
                rate = self.fps
            self.info = {
                'backend': cap.getBackendName(),
                'reported_fps': rate,
                'requested_size': [self.width, self.height],
                'buffer_size_request_accepted': bool(buffer_supported),
                'timestamp_origin': 'after VideoCapture.read; upstream latency unknown',
            }
            seq, failures = 0, 0
            origin = time.perf_counter()
            while not self.stop_event.is_set():
                ok, image = cap.read()
                if not ok:
                    if self.is_file:
                        break
                    failures += 1
                    if failures >= 30:
                        raise RuntimeError('Camera stopped delivering frames')
                    self.stop_event.wait(.02)
                    continue
                failures = 0
                source_time = seq / rate if self.is_file else None
                if self.is_file:
                    # Playback at source FPS, not maximum decoder throughput.
                    self.stop_event.wait(max(0, origin + source_time - time.perf_counter()))
                if self.stop_event.is_set():
                    break
                frame = Frame(seq, time.perf_counter(), image, source_time)
                self.info['actual_size'] = [image.shape[1], image.shape[0]]
                with self.condition:
                    self.latest = frame
                    self.condition.notify_all()
                seq += 1
        except Exception as exc:
            self.error = exc
        finally:
            if cap is not None:
                cap.release()
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def get(self, after=-1, timeout=1.0):
        with self.condition:
            self.condition.wait_for(
                lambda: self.error or self.done or (self.latest is not None and self.latest.seq > after),
                timeout,
            )
            if self.error:
                raise RuntimeError('Capture failed') from self.error
            if self.latest is not None and self.latest.seq > after:
                return self.latest
            return None

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=3)
        # A driver can block read(). Do not release its handle from another thread.
        if self.thread.is_alive():
            print('Warning: camera driver is still blocked; capture thread is daemonized.')
