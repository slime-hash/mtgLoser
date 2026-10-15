import json
from pathlib import Path

from mcoc_vision.curation import curate_manifest, parse_index_ranges, select_records


def record(index, boxes=2, confidence=.9):
    return {'box_count': boxes, 'confidences': [confidence] * boxes,
            'tracker_status': 'detected', 'novelty': index}


def test_ranges_are_inclusive_and_validated():
    assert parse_index_ranges('1-3,7-9', 10) == [(1, 3), (7, 9)]
    try:
        parse_index_ranges('2-10', 10)
    except ValueError as error:
        assert 'Invalid record range' in str(error)
    else:
        raise AssertionError('out-of-bounds range accepted')


def test_selection_spans_time_and_prefers_hard_frame_in_each_bin():
    rows = [record(index) for index in range(12)]
    rows[2] = record(2, boxes=0)
    rows[8] = record(8, boxes=1, confidence=.2)
    selected = select_records(rows, 4, [(0, 11)])
    assert [index for index, _ in selected] == [2, 5, 8, 11]


def test_curated_manifest_is_directly_reviewable(tmp_path):
    source = tmp_path / 'source'
    (source / 'images').mkdir(parents=True)
    (source / 'labels_suggested').mkdir()
    rows = []
    for index in range(4):
        image = source / 'images' / f'{index}.jpg'
        label = source / 'labels_suggested' / f'{index}.txt'
        image.write_bytes(b'image')
        label.write_text('', encoding='utf-8')
        rows.append({**record(index), 'image': str(image.resolve()),
                     'label_suggestion': str(label.resolve()), 'status': 'needs_review'})
    (source / 'manifest.json').write_text(json.dumps({'records': rows}), encoding='utf-8')
    result = curate_manifest(source, tmp_path / 'curated', limit=2, index_ranges='0-3')
    assert result['total'] == 2
    assert all(Path(row['image']).is_file() for row in result['records'])
    assert all(Path(row['label_suggestion']).is_file() for row in result['records'])
