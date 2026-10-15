"""Opponent association with short bounded prediction and explicit lost state.

Initial acquisition assumes the opponent is the right-most of two fighters.
This is not a general re-identification system; crossing/occlusion needs evaluation.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Detection:
    box: np.ndarray
    confidence: float


@dataclass(frozen=True)
class Track:
    box: np.ndarray | None
    status: str
    confidence: float
    track_id: int
    since_detection_ms: float


def center(box):
    return (box[:2] + box[2:]) * .5


def iou(a, b):
    wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(wh))
    union = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection)
    return intersection / max(union, 1e-6)


class OpponentTracker:
    def __init__(self, acquire_conf=.45, hold_seconds=.12, reset_seconds=.6):
        self.acquire_conf = acquire_conf
        self.hold_seconds = hold_seconds
        self.reset_seconds = reset_seconds
        self.track_id = 0
        self.reset()

    def reset(self):
        self.box = None
        self.velocity = np.zeros(4, dtype=float)
        self.last_time = None
        self.last_seen = None
        self.confidence = 0.
        self.pending = None
        self.pending_count = 0

    def update(self, detections, timestamp, shape):
        h, w = shape[:2]
        # Exclude HUD-sized detections. All thresholds are relative to actual image.
        candidates = [d for d in detections if
                      d.box[3] - d.box[1] >= .15 * h and
                      d.box[2] - d.box[0] >= .025 * w and
                      (d.box[2] - d.box[0]) < .85 * w]
        if self.last_time is not None and timestamp <= self.last_time:
            raise ValueError('Tracker requires strictly increasing timestamps')
        if self.box is not None and timestamp - self.last_seen > self.reset_seconds:
            self.reset()
        if self.box is None:
            strong = [d for d in candidates if d.confidence >= self.acquire_conf]
            # A lone detection could be the player: do not silently assign it.
            if len(strong) >= 2:
                candidate = max(strong, key=lambda d: center(d.box)[0])
                others = [d for d in strong if d is not candidate]
                separated = center(candidate.box)[0] - max(center(d.box)[0] for d in others) > .12 * w
                consistent = self.pending is not None and iou(candidate.box, self.pending) > .25
                self.pending_count = self.pending_count + 1 if consistent and separated else int(separated)
                self.pending = candidate.box.copy()
                if self.pending_count >= 3:
                    self.box = candidate.box.copy().astype(float)
                    self.last_time = self.last_seen = timestamp
                    self.confidence = candidate.confidence
                    self.track_id += 1
                    return Track(self.box.copy(), 'detected', self.confidence, self.track_id, 0.)
            else:
                self.pending = None
                self.pending_count = 0
            return Track(None, 'searching', 0., self.track_id, 0.)

        dt = timestamp - self.last_time
        prediction = self.box + self.velocity * min(dt, self.hold_seconds)
        ranked = []
        for d in candidates:
            displacement = np.linalg.norm((center(d.box) - center(prediction)) / np.array([w, h]))
            overlap = iou(d.box, prediction)
            ratio = np.prod(d.box[2:] - d.box[:2]) / max(np.prod(prediction[2:] - prediction[:2]), 1)
            if .35 < ratio < 2.8 and displacement < .18 and (overlap > .1 or displacement < .07):
                ranked.append((overlap + .15 * d.confidence - displacement, d))
        ranked.sort(key=lambda item: item[0], reverse=True)
        detection = None
        ambiguous = False
        if len(candidates) >= 2:
            by_side = sorted(candidates, key=lambda d: center(d.box)[0])
            rightmost, runner_up = by_side[-1], by_side[-2]
            separated = center(rightmost.box)[0] - center(runner_up.box)[0] > .035 * w
            if separated:
                # Game invariant: the opponent is always the fighter on the right.
                detection = rightmost
            else:
                ambiguous = True
        elif ranked:
            # With one visible fighter, retain it only through temporal matching;
            # never reacquire a lone box as the opponent.
            detection = ranked[0][1]
        if detection is not None:
            # Correct most of the displacement immediately; avoid sticky smoothing.
            corrected = prediction + .85 * (detection.box - prediction)
            instant_velocity = (corrected - self.box) / max(dt, 1e-3)
            self.velocity = .7 * self.velocity + .3 * instant_velocity
            self.velocity = np.clip(self.velocity, -2 * max(w, h), 2 * max(w, h))
            self.box = corrected
            self.last_seen = timestamp
            self.confidence = detection.confidence
            status = 'detected'
        else:
            self.box = prediction
            self.velocity *= .7
            status = 'ambiguous' if ambiguous else 'predicted'
        self.last_time = timestamp
        age = timestamp - self.last_seen
        self.box[[0, 2]] = np.clip(self.box[[0, 2]], 0, w - 1)
        self.box[[1, 3]] = np.clip(self.box[[1, 3]], 0, h - 1)
        visible = age <= self.hold_seconds
        return Track(self.box.copy() if visible else None,
                     status if visible else 'lost', self.confidence if visible else 0.,
                     self.track_id, age * 1000)
