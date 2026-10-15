import json

import pytest

from mcoc_vision.library import BattleLibrary


def source(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    image = root / 'frame.jpg'
    image.write_bytes(b'test')
    label = root / 'frame.txt'
    label.write_text('0 .2 .5 .2 .4\n0 .8 .5 .2 .4\n')
    (root / 'manifest.json').write_text(json.dumps({'records': [
        {'image': str(image), 'label_suggestion': str(label)}]}))
    return root


def test_reimport_and_live_corrections(tmp_path):
    src = source(tmp_path)
    library = BattleLibrary(tmp_path / 'library')
    library.register(src, 'fight-1', 'juggernaut', 'hulk')
    library.register(src, 'fight-1', 'juggernaut', 'hulk')
    views = list((library.root / 'characters/juggernaut/samples').glob('*.json'))
    assert len(views) == 1
    result = library.resolve('juggernaut', views[0].name)
    assert result['labels'][0][1] == pytest.approx(.8)
    assert len(result['labels']) == 2
    (src / 'labels_reviewed').mkdir()
    (src / 'labels_reviewed/frame.txt').write_text('0 .25 .5 .2 .4\n0 .75 .5 .2 .4\n')
    updated = library.resolve('juggernaut', views[0].name)
    assert updated['label_status'] == 'reviewed'
    assert updated['labels'][0][1] == .75
    with pytest.raises(ValueError, match='different'):
        library.register(src, 'fight-1', 'hulk', 'juggernaut')


def test_lone_box_does_not_acquire_character_identity(tmp_path):
    src = source(tmp_path)
    (src / 'frame.txt').write_text('0 .7 .5 .2 .4\n')
    library = BattleLibrary(tmp_path / 'library')
    library.register(src, 'fight-1', 'juggernaut', 'hulk')
    view = next((library.root / 'characters/hulk/samples').glob('*.json'))
    assert library.resolve('hulk', view.name)['identity_needs_review']


def test_reject_path_escape(tmp_path):
    library = BattleLibrary(tmp_path)
    with pytest.raises(ValueError):
        library.register(tmp_path, '../escape', 'hulk', 'hulk')
