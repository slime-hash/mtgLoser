"""Append-only battle registration and character views of canonical annotations."""
import hashlib
import json
from pathlib import Path
import re


def identifier(value):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,95}', value):
        raise ValueError('IDs must use lowercase ASCII letters, digits, _ or -')
    return value


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


class BattleLibrary:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def register(self, source, fight_id, left, right):
        """Register ONE battle manifest. Repeat calls repair missing indices safely.

        Character identity is supplied explicitly until the visual identity module
        exists. An input containing multiple battles must first be segmented.
        """
        for value in (fight_id, left, right):
            identifier(value)
        source = Path(source).resolve()
        manifest_path = source / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        entries = []
        for row in manifest['records']:
            image = Path(row['image']).resolve()
            label = Path(row['label_suggestion']).resolve()
            if not image.is_file() or not label.is_file():
                raise ValueError(f'Missing source image or suggestion: {image}')
            sample_id = hashlib.sha256(str(image).encode('utf-8')).hexdigest()[:24]
            entries.append(dict(sample_id=sample_id, image=str(image),
                                suggested=str(label),
                                reviewed=str(source / 'labels_reviewed' / (image.stem + '.txt'))))
        if len({r['sample_id'] for r in entries}) != len(entries):
            raise ValueError('Duplicate image in source manifest')
        battle = dict(schema=1, fight_id=fight_id, left=left, right=right,
                      identity_source='explicit', source_manifest=str(manifest_path),
                      samples=entries)
        target = self.root / 'fights' / fight_id / 'fight.json'
        if target.exists():
            if json.loads(target.read_text(encoding='utf-8')) != battle:
                raise ValueError('Battle ID already registered with different sources or identities')
        else:
            save_new(target, battle)
        for side, character, transform in [('left', left, 'flip_horizontal'),
                                           ('right', right, 'identity')]:
            for entry in entries:
                view = dict(fight_id=fight_id, sample_id=entry['sample_id'],
                            character_id=character, source_side=side, transform=transform)
                index = self.root / 'characters' / character / 'samples' / (
                    f"{fight_id}_{entry['sample_id']}_{side}.json")
                if not index.exists():
                    save_new(index, view)
        return dict(fight_id=fight_id, frames=len(entries), character_views=2 * len(entries))

    def resolve(self, character, view_name):
        """Read current corrections on demand; never copy stale reviewed labels.

        Both boxes are retained. Lone/merged boxes are not assigned an identity.
        """
        identifier(character)
        if Path(view_name).name != view_name or not view_name.endswith('.json'):
            raise ValueError('Expected a view JSON filename')
        view = json.loads((self.root / 'characters' / character / 'samples' /
                           view_name).read_text(encoding='utf-8'))
        fight = json.loads((self.root / 'fights' / view['fight_id'] /
                            'fight.json').read_text(encoding='utf-8'))
        entry = next(r for r in fight['samples'] if r['sample_id'] == view['sample_id'])
        reviewed = Path(entry['reviewed'])
        label = reviewed if reviewed.is_file() else Path(entry['suggested'])
        rows = []
        for line in label.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            values = [float(v) for v in line.split()]
            if (len(values) != 5 or values[0] != 0 or
                    not all(0 <= v <= 1 for v in values[1:]) or
                    values[3] <= 0 or values[4] <= 0):
                raise ValueError(f'Invalid fighter label: {label}')
            rows.append(values)
        rows.sort(key=lambda row: row[1])
        target_index = None
        if len(rows) == 2 and rows[1][1] - rows[0][1] > .035:
            target_index = 0 if view['source_side'] == 'left' else 1
        if view['transform'] == 'flip_horizontal':
            rows = [[r[0], 1 - r[1], *r[2:]] for r in rows]
        return dict(**view, image=entry['image'], labels=rows,
                    label_status='reviewed' if reviewed.is_file() else 'suggested',
                    target_index=target_index, identity_needs_review=target_index is None)
