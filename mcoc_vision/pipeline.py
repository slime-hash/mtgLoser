from collections import deque
from dataclasses import dataclass
import threading
import time

import numpy as np

from .capture import Frame
from .tracking import OpponentTracker, Track


@dataclass(frozen=True)
class Result:
    frame: Frame
    detections: list
    track: Track
    detector_ms: float
    application_ms: float
    completed_at: float


class Pipeline:
    def __init__(self, capture, detector):
        self.capture, self.detector = capture, detector
        self.tracker = OpponentTracker()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.reset_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name='inference')
        self.latest = None
        self.error = None
        self.done = False
        self.samples = deque(maxlen=3600)
        self.processed = self.skipped = 0
        self.started_at = None
        self.first_frame_at = self.last_frame_at = None
        self.first_seq = self.last_seq = None
        self.last_completed_at = None
        self.detected = self.predicted = self.absent = 0

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        seq = -1
        try:
            while not self.stop_event.is_set():
                frame = self.capture.get(seq, timeout=.2)
                if frame is None:
                    if self.capture.done:
                        break
                    continue
                if self.started_at is None:
                    self.started_at = time.perf_counter()
                if self.reset_event.is_set():
                    self.tracker.reset()
                    self.reset_event.clear()
                detections, detector_ms = self.detector.detect(frame.image)
                track = self.tracker.update(detections, frame.received_at, frame.image.shape)
                completed = time.perf_counter()
                application_ms = (completed - frame.received_at) * 1000
                result = Result(frame, detections, track, detector_ms, application_ms, completed)
                with self.lock:
                    self.latest = result
                    self.samples.append((detector_ms, application_ms, completed))
                    self.processed += 1
                    self.skipped += max(0, frame.seq - seq - 1) if seq >= 0 else 0
                    self.detected += int(track.status == 'detected')
                    self.predicted += int(track.box is not None and track.status != 'detected')
                    self.absent += int(track.box is None)
                    if self.first_seq is None:
                        self.first_seq, self.first_frame_at = frame.seq, frame.received_at
                    self.last_seq, self.last_frame_at = frame.seq, frame.received_at
                    self.last_completed_at = completed
                seq = frame.seq
        except Exception as exc:
            self.error = exc
        finally:
            self.done = True

    def get(self):
        if self.error:
            raise RuntimeError('Inference worker failed') from self.error
        with self.lock:
            return self.latest

    def report(self):
        with self.lock:
            rows = np.array(self.samples)
            elapsed = (self.last_completed_at - self.started_at) if self.last_completed_at else 0
            report = dict(processed_frames=self.processed, skipped_for_freshness=self.skipped,
                          inference_fps=self.processed / max(elapsed, 1e-6),
                          detected_frames=self.detected, predicted_frames=self.predicted,
                          absent_frames=self.absent, capture=self.capture.info.copy(),
                          source_to_result_latency_ms=None,
                          note='Application latency starts AFTER read(); source/OBS/driver buffering is not measured. Tracking coverage is not accuracy.')
            if self.first_seq is not None and self.last_frame_at > self.first_frame_at:
                report['capture_delivery_fps'] = (self.last_seq - self.first_seq) / (self.last_frame_at - self.first_frame_at)
            if len(rows):
                for i, name in enumerate(('detector_ms', 'application_ms')):
                    report[name] = {f'p{p}': float(np.percentile(rows[:, i], p)) for p in (50, 95, 99)}
                recent = rows[rows[:, 2] >= rows[-1, 2] - 2]
                report['recent_inference_fps'] = (len(recent) - 1) / max(recent[-1, 2] - recent[0, 2], 1e-6)
            return report

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            print('Warning: inference has not stopped within 5 seconds.')
