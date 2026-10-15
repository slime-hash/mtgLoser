import json
from pathlib import Path

import cv2
import numpy as np

from mcoc_vision.session import BackgroundProcessor, ScreenClassifier


def test_screen_classifier_combat_pause_and_sustained_absence():
    classifier = ScreenClassifier(absence_seconds=4)
    combat = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Compact green pause control with two white vertical bars.
    combat[25:80, 929:981] = (0, 150, 0)
    combat[41:68, 944:951] = (245, 245, 245)
    combat[41:68, 959:966] = (245, 245, 245)
    # Both long health fills are mandatory corroborating evidence.
    combat[75:105, 395:700] = (235, 235, 235)
    combat[75:105, 1205:1510] = (0, 210, 255)
    assert classifier.observe(combat, 1).state == 'combat'

    pause = np.full_like(combat, 12)
    pause[40:140, 310:750] = (0, 0, 150)
    pause[40:140, 1170:1610] = (0, 0, 150)
    assert classifier.observe(pause, 2).state == 'paused'

    blank = np.full_like(combat, 80)
    assert classifier.observe(blank, 4).state == 'unknown'
    assert classifier.observe(blank, 6).state == 'loading'


def test_screen_classifier_rejects_wide_green_menu_header():
    classifier = ScreenClassifier(absence_seconds=4)
    menu = np.full((1080, 1920, 3), 80, dtype=np.uint8)
    menu[0:70, 500:1400] = (0, 150, 0)
    assert classifier.observe(menu, 1).state == 'unknown'


def test_background_processor_creates_recoverable_indices(tmp_path):
    fight_dir = tmp_path / 'fights' / 'fight1'
    images = fight_dir / 'images'
    images.mkdir(parents=True)
    assert cv2.imwrite(str(images / 'frame.jpg'), np.zeros((80, 120, 3), dtype=np.uint8))
    fight = {'schema': 1, 'fight_id': 'fight1', 'session_id': 'session1',
             'left_character': 'juggernaut', 'right_character': None,
             'identity_source': 'left_explicit_right_unresolved',
             'directory': str(fight_dir.resolve()), 'images': ['frame.jpg'],
             'partial': False}
    worker = BackgroundProcessor(tmp_path, model=None)
    worker.submit(fight)
    worker.close()

    manifest = json.loads((fight_dir / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['total'] == 1
    assert manifest['records'][0]['box_count'] == 0
    assert (tmp_path / 'unresolved' / 'fight1.json').is_file()
    pending = tmp_path / 'characters' / 'juggernaut' / 'pending' / 'fight1.json'
    assert json.loads(pending.read_text(encoding='utf-8'))['transform'] == 'flip_horizontal'
    job = json.loads((tmp_path / 'jobs' / 'fight1.json').read_text(encoding='utf-8'))
    assert job['state'] == 'complete'
