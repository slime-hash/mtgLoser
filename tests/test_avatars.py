import cv2
import numpy as np

from mcoc_vision.avatars import AvatarCatalog


def _pattern(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (30, 24, 3), dtype=np.uint8)


def test_one_static_avatar_is_enough_for_exact_ui_match(tmp_path):
    refs = tmp_path / 'avatars'
    hulk = _pattern(1)
    juggernaut = _pattern(2)
    cv2.imwrite(str(tmp_path / 'hulk.png'), hulk)
    cv2.imwrite(str(tmp_path / 'juggernaut.png'), juggernaut)
    catalog = AvatarCatalog(refs)
    catalog.add('hulk', tmp_path / 'hulk.png')
    catalog.add('juggernaut', tmp_path / 'juggernaut.png')
    scene = np.zeros((100, 140, 3), dtype=np.uint8)
    scene[40:70, 50:74] = hulk
    cv2.imwrite(str(tmp_path / 'scene.png'), scene)
    result = catalog.identify(tmp_path / 'scene.png')
    assert result['accepted'] is True
    assert result['character_id'] == 'hulk'


def test_unknown_avatar_is_not_forced_into_catalog(tmp_path):
    refs = tmp_path / 'avatars'
    cv2.imwrite(str(tmp_path / 'hulk.png'), _pattern(1))
    catalog = AvatarCatalog(refs)
    catalog.add('hulk', tmp_path / 'hulk.png')
    cv2.imwrite(str(tmp_path / 'unknown.png'), _pattern(3))
    assert catalog.identify(tmp_path / 'unknown.png')['accepted'] is False
