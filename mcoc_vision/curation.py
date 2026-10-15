"""Create a small temporally diverse review set from hard-sample manifests."""
from pathlib import Path
import json
import shutil

import numpy as np


def parse_index_ranges(value, total):
    """Parse inclusive zero-based ranges such as ``1-99,132-239``."""
    if not value:
        return [(0, total - 1)] if total else []
    ranges = []
    for item in value.split(','):
        parts = item.strip().split('-', 1)
        start = int(parts[0])
        end = int(parts[-1])
        if start < 0 or end < start or end >= total:
            raise ValueError(f'Invalid record range {item!r} for {total} records')
        ranges.append((start, end))
    return ranges


def _hardness(record):
    count = int(record.get('box_count', 0))
    confidences = record.get('confidences') or []
    minimum = min(confidences) if confidences else 0.
    status = record.get('tracker_status', '')
    # Missing boxes dominate, then low confidence and tracker loss.  Novelty is
    # only a tie-breaker because it may also come from particles or UI text.
    return ((2 - min(count, 2)) * 10 + (1 - minimum) * 3 +
            (2 if status in {'lost', 'searching'} else 0) +
            min(float(record.get('novelty', 0)), 50) / 100)


def _allocate(lengths, limit):
    if not lengths or limit <= 0:
        return []
    total = sum(lengths)
    raw = [limit * length / total for length in lengths]
    quotas = [min(length, int(value)) for length, value in zip(lengths, raw)]
    for index, length in enumerate(lengths):
        if length and not quotas[index] and sum(quotas) < limit:
            quotas[index] = 1
    while sum(quotas) < min(limit, total):
        choices = [i for i, length in enumerate(lengths) if quotas[i] < length]
        if not choices:
            break
        index = max(choices, key=lambda i: raw[i] - quotas[i])
        quotas[index] += 1
    return quotas


def select_records(records, limit, ranges=None):
    ranges = ranges or ([(0, len(records) - 1)] if records else [])
    groups = [[(index, records[index]) for index in range(start, end + 1)]
              for start, end in ranges]
    quotas = _allocate([len(group) for group in groups], min(limit, sum(map(len, groups))))
    selected = []
    for group, quota in zip(groups, quotas):
        # One hardest frame per temporal bin gives action-phase coverage without
        # keeping ten adjacent frames from the same animation.
        for chunk in np.array_split(np.arange(len(group)), quota) if quota else []:
            choices = [group[int(position)] for position in chunk]
            selected.append(max(choices, key=lambda item: _hardness(item[1])))
    return sorted(selected, key=lambda item: item[0])


def curate_manifest(input_dir, output_dir, limit=64, index_ranges=None):
    source = Path(input_dir)
    output = Path(output_dir)
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    records = manifest['records']
    ranges = parse_index_ranges(index_ranges, len(records))
    selected = select_records(records, int(limit), ranges)
    images = output / 'images'
    labels = output / 'labels_suggested'
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    curated = []
    for source_index, record in selected:
        image_source = Path(record['image'])
        label_source = Path(record['label_suggestion'])
        image_target = images / image_source.name
        label_target = labels / label_source.name
        shutil.copy2(image_source, image_target)
        shutil.copy2(label_source, label_target)
        copied = dict(record)
        copied.update(image=str(image_target.resolve()),
                      label_suggestion=str(label_target.resolve()),
                      source_record_index=source_index)
        curated.append(copied)
    result = {
        'format': 'mcoc-curated-hard-samples-v1',
        'source_manifest': str((source / 'manifest.json').resolve()),
        'source_total': len(records),
        'total': len(curated),
        'index_ranges': index_ranges,
        'review_required': True,
        'note': 'Suggestions are untrusted and must be reviewed before training.',
        'counts': {str(n): sum(row.get('box_count') == n for row in curated)
                   for n in range(3)},
        'records': curated,
    }
    (output / 'manifest.json').write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                          encoding='utf-8')
    return result
