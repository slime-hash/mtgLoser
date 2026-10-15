"""Human-in-the-loop collection of diverse detector failures.

The collector consumes the same local pixels and predictions already produced by
the preview pipeline.  It never promotes suggestions to training labels: every
saved sample is emitted in the existing review format.
"""
from pathlib import Path
import json

import cv2
import numpy as np

from .labeling import fighter_suggestions, yolo_row


class HardSampleCollector:
    def __init__(self, output, model, min_interval=.10, max_gap=.50,
                 novelty=2.0, hard_novelty=.65, max_frames=240, quality=95):
        self.root = Path(output)
        self.images = self.root / 'images'
        self.labels = self.root / 'labels_suggested'
        self.images.mkdir(parents=True, exist_ok=True)
        self.labels.mkdir(parents=True, exist_ok=True)
        self.model = str(model)
        self.min_interval = float(min_interval)
        self.max_gap = float(max_gap)
        self.novelty = float(novelty)
        self.hard_novelty = float(hard_novelty)
        self.max_frames = int(max_frames)
        self.quality = int(quality)
        self.enabled = False
        self.records = []
        self.last_seq = -1
        self.last_saved_at = None
        self.last_signature = None
        self.started_at = None
        self._write_manifest()

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        if self.enabled:
            # Compare the first active frame against nothing, not against a frame
            # from an earlier collection interval.
            self.last_signature = None
            self.last_saved_at = None

    @staticmethod
    def _signature(image):
        # Ignore most of the HUD, whose changing bars/numbers are not useful pose
        # novelty.  Low resolution deliberately removes tiny particle noise.
        h = image.shape[0]
        playfield = image[round(.12 * h):]
        gray = cv2.cvtColor(playfield, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (96, 48), interpolation=cv2.INTER_AREA)

    def consider(self, result):
        if not self.enabled or len(self.records) >= self.max_frames:
            return False
        if result.frame.seq <= self.last_seq:
            return False
        self.last_seq = result.frame.seq
        now = result.frame.received_at
        if self.started_at is None:
            self.started_at = now
        if self.last_saved_at is not None and now - self.last_saved_at < self.min_interval:
            return False

        suggestions = fighter_suggestions(result.detections, result.frame.image.shape)
        confidences = [item.confidence for item in suggestions]
        reasons = []
        if len(suggestions) < 2:
            reasons.append('fewer_than_two_fighters')
        if confidences and min(confidences) < .45:
            reasons.append('low_confidence')
        if result.track.status != 'detected':
            reasons.append(f'tracker_{result.track.status}')

        signature = self._signature(result.frame.image)
        novelty = 255. if self.last_signature is None else float(
            np.mean(cv2.absdiff(signature, self.last_signature)))
        forced = self.last_saved_at is None or now - self.last_saved_at >= self.max_gap
        hard = bool(reasons)
        if not ((hard and (novelty >= self.hard_novelty or forced)) or
                (novelty >= self.novelty)):
            return False
        if novelty >= self.novelty:
            reasons.append('visual_change')

        index = len(self.records)
        stem = f'frame_{index:04d}_seq_{result.frame.seq:07d}'
        image_path = self.images / f'{stem}.jpg'
        label_path = self.labels / f'{stem}.txt'
        if not cv2.imwrite(str(image_path), result.frame.image,
                           [cv2.IMWRITE_JPEG_QUALITY, self.quality]):
            raise RuntimeError(f'Cannot write hard sample {image_path}')
        rows = [yolo_row(item.box, result.frame.image.shape) for item in suggestions]
        label_path.write_text(''.join(
            ' '.join(f'{value:.6f}' if column else str(value)
                     for column, value in enumerate(row)) + '\n'
            for row in rows
        ), encoding='utf-8')
        record = {
            'image': str(image_path.resolve()),
            'label_suggestion': str(label_path.resolve()),
            'status': 'needs_review',
            'box_count': len(suggestions),
            'confidences': confidences,
            'reasons': reasons,
            'novelty': novelty,
            'sequence': result.frame.seq,
            'elapsed_seconds': now - self.started_at,
            'tracker_status': result.track.status,
        }
        self.records.append(record)
        self.last_signature = signature
        self.last_saved_at = now
        self._write_manifest()
        return True

    def _manifest(self):
        return {
            'format': 'mcoc-hard-samples-v1',
            'model': self.model,
            'total': len(self.records),
            'counts': {str(n): sum(row['box_count'] == n for row in self.records)
                       for n in range(3)},
            'review_required': True,
            'note': 'Suggestions are untrusted and must be reviewed before training.',
            'selection': {
                'minimum_interval_seconds': self.min_interval,
                'maximum_gap_seconds': self.max_gap,
                'visual_novelty_threshold': self.novelty,
                'hard_case_novelty_threshold': self.hard_novelty,
                'maximum_frames': self.max_frames,
            },
            'records': self.records,
        }

    def _write_manifest(self):
        target = self.root / 'manifest.json'
        temporary = self.root / 'manifest.json.tmp'
        temporary.write_text(json.dumps(self._manifest(), indent=2, ensure_ascii=False),
                             encoding='utf-8')
        temporary.replace(target)

    def close(self):
        self._write_manifest()
        return self._manifest()
