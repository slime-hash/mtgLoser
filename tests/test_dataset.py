import json

import cv2
import numpy as np
import pytest

from mcoc_vision.dataset import export_reviewed_fight
from mcoc_vision.dataset import _read_rows


def test_single_right_fighter_keeps_opponent_role(tmp_path):
    label = tmp_path / 'right_only.txt'
    label.write_text('1 0.750000 0.500000 0.200000 0.600000\n', encoding='utf-8')
    rows = _read_rows(label)
    assert len(rows) == 1
    assert rows[0][0] == 1


def _fight(root, reviewed=True):
    fight = root / 'fights' / 'fight1'
    images = fight / 'images'
    suggestions = fight / 'labels_suggested'
    images.mkdir(parents=True)
    suggestions.mkdir()
    image = images / 'frame.jpg'
    cv2.imwrite(str(image), np.zeros((40, 100, 3), dtype=np.uint8))
    label = suggestions / 'frame.txt'
    label.write_text('0 0.25 0.5 0.2 0.8\n0 0.75 0.5 0.2 0.8\n', encoding='utf-8')
    if reviewed:
        checked = fight / 'labels_reviewed'
        checked.mkdir()
        (checked / 'frame.txt').write_text(label.read_text(encoding='utf-8'), encoding='utf-8')
    manifest = {'records': [{'image': str(image.resolve()),
                             'label_suggestion': str(label.resolve())}]}
    (fight / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')


def test_export_has_two_generic_boxes_and_two_target_views(tmp_path):
    _fight(tmp_path)
    result = export_reviewed_fight(tmp_path, 'fight1', 'juggernaut', 'hulk', 'test')
    assert result['unseen_pair_evaluation'] is True
    detector = (tmp_path / 'detector/test/labels/fight1_frame.txt').read_text().splitlines()
    assert [line.split()[0] for line in detector] == ['0', '0']
    hulk = (tmp_path / 'characters/hulk/test/labels/fight1_frame.txt').read_text().splitlines()
    assert [line.split()[0] for line in hulk] == ['1', '0']
    juggernaut = (tmp_path / 'characters/juggernaut/test/labels/fight1_frame.txt').read_text().splitlines()
    assert [line.split()[0] for line in juggernaut] == ['1', '0']
    assert (tmp_path / 'detector/data.yaml').is_file()


def test_export_refuses_unreviewed_or_changed_identity(tmp_path):
    _fight(tmp_path, reviewed=False)
    with pytest.raises(ValueError, match='not fully reviewed'):
        export_reviewed_fight(tmp_path, 'fight1', 'juggernaut', 'hulk')
    (tmp_path / 'fights/fight1/labels_reviewed').mkdir()
    (tmp_path / 'fights/fight1/labels_reviewed/frame.txt').write_text(
        '0 .25 .5 .2 .8\n0 .75 .5 .2 .8\n', encoding='utf-8')
    export_reviewed_fight(tmp_path, 'fight1', 'juggernaut', 'hulk')
    with pytest.raises(ValueError, match='already exported'):
        export_reviewed_fight(tmp_path, 'fight1', 'spider_man', 'hulk')


def test_same_pair_cannot_leak_between_train_and_test(tmp_path):
    _fight(tmp_path)
    export_reviewed_fight(tmp_path, 'fight1', 'juggernaut', 'hulk', 'train')
    source = tmp_path / 'fights/fight1'
    target = tmp_path / 'fights/fight2'
    target.mkdir(parents=True)
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    (target / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    (target / 'labels_reviewed').mkdir()
    (target / 'labels_reviewed/frame.txt').write_text(
        '0 .25 .5 .2 .8\n0 .75 .5 .2 .8\n', encoding='utf-8')
    with pytest.raises(ValueError, match='cannot leak'):
        export_reviewed_fight(tmp_path, 'fight2', 'hulk', 'juggernaut', 'test')
