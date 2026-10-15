"""Automatic, external-only fight session capture and background pre-labeling."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from queue import Queue
import threading
import time
from uuid import uuid4

import cv2
import numpy as np

from .battle_state import BattleState


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


@dataclass(frozen=True)
class ScreenObservation:
    state: str
    confidence: float
    pause_green_ratio: float
    bar_saturation: float
    bottom_green_ratio: float
    mean_brightness: float


class ScreenClassifier:
    """Recognize the fight HUD without using fighter detections.

    The normalized regions make this independent of capture resolution.  A
    sustained HUD absence is treated as a boundary; short cinematic effects do
    not split a fight.
    """
    def __init__(self, absence_seconds=4.0):
        self.absence_seconds = float(absence_seconds)
        self.last_combat_at = None

    @staticmethod
    def _roi(image, x1, y1, x2, y2):
        h, w = image.shape[:2]
        return image[round(y1 * h):round(y2 * h), round(x1 * w):round(x2 * w)]

    @staticmethod
    def _ratio(mask):
        return float(np.mean(mask)) if mask.size else 0.0

    @staticmethod
    def _pause_symbol_score(region):
        """Return a score only for the compact green control with white II bars."""
        if region.size == 0:
            return 0.0, 0.0
        height, width = region.shape[:2]
        hue, saturation, value = cv2.split(region)
        green = ((hue >= 35) & (hue <= 90) &
                 (saturation >= 85) & (value >= 65))
        green_ratio = float(np.mean(green))
        white = ((saturation <= 75) & (value >= 185)).astype(np.uint8) * 255
        # The II glyph is in the green upper part; exclude the Esc badge below.
        white[round(.62 * height):, :] = 0
        white[:, :round(.25 * width)] = 0
        white[:, round(.75 * width):] = 0
        count, _, stats, centers = cv2.connectedComponentsWithStats(white, 8)
        bars = []
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            center_x, center_y = centers[index]
            if (area >= 8 and .025 * width <= w <= .14 * width and
                    .13 * height <= h <= .40 * height and h >= 1.7 * w and
                    .28 * width <= center_x <= .72 * width and
                    center_y <= .48 * height):
                bars.append((center_x, center_y, w, h))
        for first_index, first in enumerate(bars):
            for second in bars[first_index + 1:]:
                separation = abs(first[0] - second[0]) / width
                height_ratio = min(first[3], second[3]) / max(first[3], second[3])
                y_gap = abs(first[1] - second[1]) / height
                if .07 <= separation <= .28 and height_ratio >= .70 and y_gap <= .08:
                    glyph_score = min(1.0, height_ratio * (1.0 - y_gap))
                    return green_ratio, glyph_score
        return green_ratio, 0.0

    @staticmethod
    def _health_bar_score(region):
        """Measure a long bright horizontal health fill, not arbitrary colour."""
        if region.size == 0:
            return 0.0
        hue, saturation, value = cv2.split(region)
        pale = (value >= 145) & (saturation <= 115)
        yellow = ((hue >= 10) & (hue <= 38) &
                  (saturation >= 90) & (value >= 135))
        mask = pale | yellow
        row_coverage = np.mean(mask, axis=1)
        strong_rows = row_coverage[row_coverage >= .42]
        return float(np.mean(strong_rows)) if strong_rows.size >= 3 else 0.0

    def observe(self, image, timestamp):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        center = self._roi(hsv, .475, .015, .525, .105)
        left = self._roi(hsv, .20, .060, .38, .120)
        right = self._roi(hsv, .62, .060, .80, .120)
        broad_left = self._roi(hsv, .16, .035, .39, .13)
        broad_right = self._roi(hsv, .61, .035, .84, .13)
        bottom = self._roi(hsv, .20, .78, .80, .98)
        green = lambda region: self._ratio(
            (region[:, :, 0] >= 35) & (region[:, :, 0] <= 90) &
            (region[:, :, 1] >= 80) & (region[:, :, 2] >= 60))
        pause_green, pause_glyph = self._pause_symbol_score(center)
        left_bar = self._health_bar_score(left)
        right_bar = self._health_bar_score(right)
        saturated = lambda region: self._ratio(
            (region[:, :, 1] >= 70) & (region[:, :, 2] >= 50))
        bar_saturation = (saturated(broad_left) + saturated(broad_right)) * .5
        bottom_green = green(bottom)
        brightness = float(np.mean(image))

        combat_hud = (pause_green >= .055 and pause_glyph >= .65 and
                      left_bar >= .42 and right_bar >= .42)
        if combat_hud:
            self.last_combat_at = timestamp
            confidence = min(.99, .82 + .08 * pause_glyph +
                             .05 * min(left_bar, right_bar))
            state = 'combat'
        elif brightness < 27 and bar_saturation >= .10:
            state, confidence = 'paused', .92
        elif bottom_green >= .018 and bar_saturation < .12:
            state, confidence = 'result', .90
        elif self.last_combat_at is not None and timestamp - self.last_combat_at >= self.absence_seconds:
            state, confidence = 'loading', .90
        else:
            state, confidence = 'unknown', .0
        return ScreenObservation(state, confidence, pause_green, bar_saturation,
                                 bottom_green, brightness)


class BackgroundProcessor:
    """Bounded single worker: finished fights cannot block fresh capture."""
    def __init__(self, root, model=None, imgsz=640, confidence=.12, prompts=None):
        self.root = Path(root)
        self.model_path = str(model) if model else None
        self.imgsz = int(imgsz)
        self.confidence = float(confidence)
        self.prompts = list(prompts) if prompts else None
        self.queue = Queue()
        self.thread = threading.Thread(target=self._run, name='fight-postprocess', daemon=True)
        self.error = None
        self.detector = None
        self.thread.start()

    def submit(self, fight):
        job = {'schema': 1, 'fight_id': fight['fight_id'], 'state': 'queued',
               'submitted_at': datetime.now(timezone.utc).isoformat()}
        _atomic_json(self.root / 'jobs' / f"{fight['fight_id']}.json", job)
        self.queue.put(fight)

    def _run(self):
        while True:
            fight = self.queue.get()
            if fight is None:
                self.queue.task_done()
                return
            try:
                try:
                    self._process(fight)
                except Exception as exc:
                    self.error = exc
                    fight_id = fight.get('fight_id', 'unknown')
                    _atomic_json(self.root / 'jobs' / f'{fight_id}.json',
                                 {'schema': 1, 'fight_id': fight_id, 'state': 'failed',
                                  'error': f'{type(exc).__name__}: {exc}'})
            finally:
                self.queue.task_done()

    def _process(self, fight):
        from .avatars import AvatarCatalog
        from .labeling import fighter_suggestions, yolo_row

        fight_id = fight['fight_id']
        job_path = self.root / 'jobs' / f'{fight_id}.json'
        _atomic_json(job_path, {'schema': 1, 'fight_id': fight_id, 'state': 'processing'})
        if self.model_path and self.detector is None:
            from .detector import Detector
            self.detector = Detector(self.model_path, self.imgsz, self.confidence,
                                     classes=None if self.prompts else (0,),
                                     prompts=self.prompts)
        records = []
        labels = Path(fight['directory']) / 'labels_suggested'
        labels.mkdir(parents=True, exist_ok=True)
        for image_name in fight['images']:
            path = Path(fight['directory']) / 'images' / image_name
            image = cv2.imread(str(path))
            if image is None:
                records.append({'image': str(path.resolve()), 'status': 'unreadable'})
                continue
            suggestions = []
            detector_ms = None
            if self.detector is not None:
                detections, detector_ms = self.detector.detect(image)
                suggestions = fighter_suggestions(detections, image.shape)
            label = labels / f'{path.stem}.txt'
            rows = [yolo_row(item.box, image.shape) for item in suggestions]
            label.write_text(''.join(
                ' '.join(f'{value:.6f}' if column else str(value)
                         for column, value in enumerate(row)) + '\n'
                for row in rows), encoding='utf-8')
            records.append({'image': str(path.resolve()),
                            'label_suggestion': str(label.resolve()),
                            'status': 'needs_review', 'box_count': len(rows),
                            'confidences': [item.confidence for item in suggestions],
                            'detector_ms': detector_ms})
        manifest = {key: value for key, value in fight.items() if key != 'images'}
        manifest.update(format='mcoc-fight-v1', total=len(records), records=records,
                        review_required=True,
                        note='Suggested boxes are never training truth before review.')
        _atomic_json(Path(fight['directory']) / 'manifest.json', manifest)
        catalog = AvatarCatalog(self.root / 'avatars')
        proposals = {}
        for side in ('left', 'right'):
            crop = fight.get('identity_crops', {}).get(side)
            proposals[side] = (catalog.identify(crop) if crop else
                               {'character_id': None, 'accepted': False,
                                'reason': 'identity crop is unavailable'})
        # The explicit session value is trusted only as a temporary fallback;
        # the right side is always visual-catalog based.
        for side in ('left', 'right'):
            character = fight.get(f'{side}_character')
            if character and character != 'unknown':
                proposals[side] = {'character_id': character,
                                   'candidate': character, 'score': 1.0,
                                   'second_score': -1.0, 'margin': 2.0,
                                   'accepted': True,
                                   'source': fight.get('identity_source', 'explicit')}
        identity = {'fight_id': fight_id, 'left': proposals['left'],
                    'right': proposals['right'],
                    'status': ('needs_review' if all(item.get('accepted')
                                                     for item in proposals.values())
                               else 'unresolved')}
        _atomic_json(Path(fight['directory']) / 'identity.json', identity)
        if identity['status'] == 'unresolved':
            _atomic_json(self.root / 'unresolved' / f'{fight_id}.json', identity)
        if fight['left_character'] != 'unknown':
            player_index = dict(fight_id=fight_id, character_id=fight['left_character'],
                                source_side='left', transform='flip_horizontal',
                                source_manifest=str((Path(fight['directory']) /
                                                     'manifest.json').resolve()),
                                status='pending_identity_and_review')
            _atomic_json(self.root / 'characters' / fight['left_character'] / 'pending' /
                         f'{fight_id}.json', player_index)
        _atomic_json(job_path, {'schema': 1, 'fight_id': fight_id, 'state': 'complete',
                               'frames': len(records),
                               'completed_at': datetime.now(timezone.utc).isoformat()})

    def close(self):
        self.queue.put(None)
        self.queue.join()
        self.thread.join()
        if self.error:
            raise RuntimeError('Background fight processing failed') from self.error


class SessionRecorder:
    def __init__(self, capture, root, player, sample_fps=10, quality=92,
                 model=None, imgsz=640, confidence=.12, absence_seconds=4.0,
                 prompts=None):
        self.capture = capture
        self.root = Path(root).resolve()
        self.player = player
        self.sample_interval = 1 / float(sample_fps)
        self.quality = int(quality)
        self.classifier = ScreenClassifier(absence_seconds)
        self.state = BattleState(confirm_seconds=.35, min_confidence=.8)
        self.processor = BackgroundProcessor(self.root, model, imgsz, confidence, prompts)
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.fights = []
        self.current = None
        self.next_sample_at = 0.0
        self.last_status_at = 0.0
        self.queue_screen = None
        self.lineup = None
        self.next_lineup_check_at = 0.0
        self.lineup_departures = 0
        self.lineup_departed = False
        self.lineup_title_box = None
        self.lineup_title_signature = None
        self.lineup_start_fight = 0
        self.series_index = 0

    def _start(self, event, frame):
        fight_id = event['fight_id']
        directory = self.root / 'fights' / fight_id
        (directory / 'images').mkdir(parents=True, exist_ok=False)
        round_index = len(self.fights) - self.lineup_start_fight
        lineup_round = (self.lineup['rounds'][round_index]
                        if self.lineup and round_index < len(self.lineup['rounds']) else None)
        left_character = self.player
        right_character = None
        identity_source = 'left_explicit_right_unresolved'
        if lineup_round:
            left_character = lineup_round['left']['character_id'] or 'unknown'
            right_character = lineup_round['right']['character_id']
            identity_source = 'queue_screen_ocr'
        self.current = {'schema': 1, 'fight_id': fight_id,
                        'session_id': self.session_id, 'round': round_index + 1,
                        'left_character': left_character,
                        'right_character': right_character,
                        'identity_source': identity_source,
                        'directory': str(directory.resolve()), 'started_at': frame.received_at,
                        'capture': dict(self.capture.info), 'images': [], 'partial': False}
        print(f'[fight {len(self.fights) + 1}] started: {fight_id}', flush=True)

    def _save(self, frame):
        if frame.received_at < self.next_sample_at:
            return
        self.next_sample_at = frame.received_at + self.sample_interval
        index = len(self.current['images'])
        name = f'frame_{index:05d}_seq_{frame.seq:09d}.jpg'
        path = Path(self.current['directory']) / 'images' / name
        if not cv2.imwrite(str(path), frame.image, [cv2.IMWRITE_JPEG_QUALITY, self.quality]):
            raise RuntimeError(f'Cannot write {path}')
        self.current['images'].append(name)
        if index == 0:
            identity = Path(self.current['directory']) / 'identity_frame.jpg'
            cv2.imwrite(str(identity), frame.image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            self.current['identity_frame'] = str(identity.resolve())

    def _end(self, event):
        if self.current is None:
            return
        self.current['ended_at'] = self.state.last_at
        self.current['partial'] = bool(event['partial'])
        self.current['end_reason'] = event['reason']
        self.current['frame_count'] = len(self.current['images'])
        fight = self.current
        self.current = None
        if not fight['images']:
            return
        self.fights.append(fight['fight_id'])
        self.processor.submit(fight)
        print(f"[fight {len(self.fights)}] queued: {fight['frame_count']} frames, "
              f"reason={fight['end_reason']}", flush=True)
        if len(self.fights) - self.lineup_start_fight >= 3:
            self.lineup = None
            self.lineup_departed = False
            self.lineup_departures = 0
            self.lineup_title_box = None
            self.lineup_title_signature = None
            self.next_lineup_check_at = 0.0
            self.lineup_start_fight = len(self.fights)
            print('Series complete; waiting for the next SERIES MATCH - SET LINEUP.',
                  flush=True)

    def _session_manifest(self, stopped=False):
        return {'schema': 1, 'session_id': self.session_id, 'player': self.player,
                'started_at': self.started_at, 'stopped': bool(stopped),
                'fights': self.fights, 'queue_screen': self.queue_screen,
                'safety': 'external screen pixels only; no game process or input interaction'}

    def run(self, seconds=0):
        session_path = self.root / 'sessions' / self.session_id / 'session.json'
        _atomic_json(session_path, self._session_manifest())
        seq = -1
        started = time.perf_counter()
        print(f'Session {self.session_id} is watching the screen. Ctrl+C stops safely.', flush=True)
        try:
            while not seconds or time.perf_counter() - started < seconds:
                frame = self.capture.get(seq, .5)
                if frame is None:
                    if self.capture.done:
                        break
                    continue
                seq = frame.seq
                if not self.lineup_departed:
                    if self.lineup is None:
                        if frame.received_at < self.next_lineup_check_at:
                            continue
                        self.next_lineup_check_at = frame.received_at + 1.0
                        candidate_path = session_path.parent / '.queue_candidate.jpg'
                        if not cv2.imwrite(str(candidate_path), frame.image,
                                           [cv2.IMWRITE_JPEG_QUALITY, 96]):
                            raise RuntimeError(f'Cannot write queue candidate: {candidate_path}')
                        try:
                            from .lineup import read_lineup
                            candidate = read_lineup(candidate_path)
                        except Exception as exc:
                            print(f'Waiting for lineup title: {exc}', flush=True)
                            continue
                        if not candidate.get('is_lineup_screen'):
                            print('Waiting for SERIES MATCH - SET LINEUP...', flush=True)
                            continue
                        self.series_index += 1
                        queue_path = session_path.parent / f'queue_screen_{self.series_index:03d}.jpg'
                        candidate_path.replace(queue_path)
                        candidate['source'] = str(queue_path.resolve())
                        self.queue_screen = str(queue_path.resolve())
                        self.lineup = candidate
                        self.lineup_title_box = candidate.get('title_box')
                        if self.lineup_title_box:
                            x1, y1, x2, y2 = self.lineup_title_box
                            title = frame.image[y1:y2, x1:x2]
                            self.lineup_title_signature = cv2.cvtColor(
                                title, cv2.COLOR_BGR2GRAY).copy()
                        _atomic_json(session_path.parent / 'lineup.json', candidate)
                        _atomic_json(session_path, self._session_manifest())
                        print('Lineup read once: ' + ', '.join(
                            f"{row['left']['display_name']} vs {row['right']['display_name']}"
                            for row in candidate['rounds']), flush=True)
                        continue
                    box = self.lineup_title_box
                    signature = self.lineup_title_signature
                    similarity = 0.0
                    if box and signature is not None:
                        x1, y1, x2, y2 = box
                        current = frame.image[y1:y2, x1:x2]
                        if current.shape[:2] == signature.shape:
                            current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
                            similarity = float(cv2.matchTemplate(
                                current, signature, cv2.TM_CCOEFF_NORMED)[0, 0])
                    if similarity >= .65:
                        self.lineup_departures = 0
                        continue
                    self.lineup_departures += 1
                    if self.lineup_departures < 5:
                        continue
                    self.lineup_departed = True
                    print('Lineup closed; waiting for the combat pause button.', flush=True)
                    continue
                observation = self.classifier.observe(frame.image, frame.received_at)
                events = self.state.observe(observation.state, observation.confidence,
                                            frame.received_at)
                for event in events:
                    if event['event'] == 'start':
                        self._start(event, frame)
                    elif event['event'] == 'end':
                        self._end(event)
                if self.current is not None and observation.state == 'combat':
                    self._save(frame)
                if frame.received_at - self.last_status_at >= 2:
                    active = len(self.current['images']) if self.current else 0
                    print(f'screen={observation.state} active_frames={active} '
                          f'queued={self.processor.queue.qsize()}', flush=True)
                    self.last_status_at = frame.received_at
        except KeyboardInterrupt:
            print('Stopping session safely...', flush=True)
        finally:
            for event in self.state.stop():
                self._end(event)
            _atomic_json(session_path, self._session_manifest(stopped=True))
            self.capture.close()
            self.processor.close()
        print(f'Session complete: {len(self.fights)} fight(s), {session_path}', flush=True)
        return self._session_manifest(stopped=True)
