"""Verified export from one reviewed fight into universal and character datasets."""
import hashlib
import json
from pathlib import Path
import shutil

import cv2

from .library import identifier


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _read_rows(path):
    rows = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if (len(values) != 5 or values[0] not in (0, 1) or
                not all(0 <= value <= 1 for value in values[1:]) or
                values[3] <= 0 or values[4] <= 0):
            raise ValueError(f'Invalid fighter label: {path}')
        rows.append(values)
    if not 1 <= len(rows) <= 2:
        raise ValueError(f'Expected one or two reviewed fighters: {path}')
    if len(rows) == 2 and rows[0][0] == rows[1][0]:
        rows.sort(key=lambda row: row[1])
        rows[0][0], rows[1][0] = 0, 1
    rows.sort(key=lambda row: row[0])
    if len(rows) == 2 and rows[0][0] == rows[1][0]:
        raise ValueError(f'Reviewed fighter roles must be unique: {path}')
    return rows


def _write_rows(path, rows):
    Path(path).write_text(''.join(
        ' '.join(str(int(value)) if column == 0 else f'{value:.6f}'
                 for column, value in enumerate(row)) + '\n'
        for row in rows), encoding='utf-8')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _copy_verified(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.tmp')
    shutil.copy2(source, temporary)
    if _sha256(source) != _sha256(temporary):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f'Copy verification failed: {source}')
    temporary.replace(target)


def _character_rows(rows, source_side, flipped):
    target = 0 if source_side == 'left' else 1
    output = []
    for row in rows:
        converted = list(row)
        converted[0] = 0 if int(row[0]) == target else 1
        if flipped:
            converted[1] = 1 - converted[1]
        output.append(converted)
    output.sort(key=lambda row: row[1])
    return output


def export_reviewed_fight(root, fight_id, left, right, split='train'):
    """Export only after every frame has a human-reviewed two-box annotation.

    The universal detector receives both boxes as class 0 (fighter). Character
    views retain the full frame and mark target as class 0, other fighter as 1.
    Left views are mirrored so the target is canonicalized to the right side.
    """
    root = Path(root).resolve()
    identifier(fight_id)
    identifier(left)
    identifier(right)
    if left == right:
        raise ValueError('Left and right character IDs must differ')
    if split not in {'train', 'val', 'test'}:
        raise ValueError('Split must be train, val, or test')
    fight_dir = root / 'fights' / fight_id
    manifest_path = fight_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    export_path = root / 'exports' / f'{fight_id}.json'
    identity = {'left': left, 'right': right}
    if export_path.exists():
        previous = json.loads(export_path.read_text(encoding='utf-8'))
        if previous['identity'] != identity or previous['split'] != split:
            raise ValueError('Fight was already exported with different identity or split')
        return previous
    pair = frozenset((left, right))
    for other_path in (root / 'exports').glob('*.json'):
        other = json.loads(other_path.read_text(encoding='utf-8'))
        other_pair = frozenset((other['identity']['left'], other['identity']['right']))
        if other_pair == pair and other['split'] != split:
            raise ValueError(f'Character pair is already assigned to {other["split"]}; '
                             'one pair cannot leak across dataset splits')

    prepared = []
    for record in manifest['records']:
        image = Path(record['image']).resolve()
        reviewed = fight_dir / 'labels_reviewed' / f'{image.stem}.txt'
        if not image.is_file() or not reviewed.is_file():
            raise ValueError(f'Fight is not fully reviewed: {image.name}')
        rows = _read_rows(reviewed)
        sample_id = f'{fight_id}_{image.stem}'
        prepared.append((sample_id, image, reviewed, rows))

    exported = []
    for sample_id, image, reviewed, rows in prepared:
        suffix = image.suffix.lower()
        detector_image = root / 'detector' / split / 'images' / f'{sample_id}{suffix}'
        detector_label = root / 'detector' / split / 'labels' / f'{sample_id}.txt'
        _copy_verified(image, detector_image)
        detector_label.parent.mkdir(parents=True, exist_ok=True)
        _write_rows(detector_label, [[0, *row[1:]] for row in rows])

        views = []
        for side, character, flipped in (
                ('left', left, True), ('right', right, False)):
            role = 0 if side == 'left' else 1
            if not any(int(row[0]) == role for row in rows):
                continue
            target_dir = root / 'characters' / character / split
            character_image = target_dir / 'images' / f'{sample_id}{suffix}'
            character_label = target_dir / 'labels' / f'{sample_id}.txt'
            metadata = target_dir / 'metadata' / f'{sample_id}.json'
            character_image.parent.mkdir(parents=True, exist_ok=True)
            if flipped:
                source_image = cv2.imread(str(image))
                if source_image is None:
                    raise ValueError(f'Unreadable image: {image}')
                temporary = character_image.with_suffix(character_image.suffix + '.tmp.jpg')
                if not cv2.imwrite(str(temporary), cv2.flip(source_image, 1),
                                   [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f'Cannot write mirrored image: {character_image}')
                temporary.replace(character_image)
            else:
                _copy_verified(image, character_image)
            character_label.parent.mkdir(parents=True, exist_ok=True)
            _write_rows(character_label, _character_rows(rows, side, flipped))
            view = {'fight_id': fight_id, 'sample_id': sample_id,
                    'character_id': character, 'source_side': side,
                    'transform': 'flip_horizontal' if flipped else 'identity',
                    'image': str(character_image), 'label': str(character_label),
                    'classes': {'0': 'target_character', '1': 'other_fighter'}}
            _atomic_json(metadata, view)
            views.append(view)
        exported.append({'sample_id': sample_id,
                         'detector_image': str(detector_image),
                         'detector_label': str(detector_label),
                         'reviewed_source': str(reviewed), 'character_views': views})

    result = {'schema': 1, 'fight_id': fight_id, 'identity': identity,
              'split': split, 'frames': len(exported),
              'unseen_pair_evaluation': split == 'test', 'samples': exported,
              'source_manifest': str(manifest_path),
              'source_cleanup_allowed': True}
    _atomic_json(export_path, result)
    _write_dataset_yaml(root)
    return result


def _write_dataset_yaml(root):
    root = Path(root).resolve()
    content = (f'path: {root.as_posix()}/detector\n'
               'train: train/images\n'
               'val: val/images\n'
               'test: test/images\n\n'
               'names:\n  0: fighter\n')
    target = root / 'detector' / 'data.yaml'
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.yaml.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(target)
