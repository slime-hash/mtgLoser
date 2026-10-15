"""Local per-fight refinement from human-reviewed boxes only."""
import json
from pathlib import Path
import shutil

import cv2


def _rows(path):
    if not Path(path).is_file():
        return []
    return [[float(value) for value in line.split()]
            for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def _target(rows, side):
    if not rows:
        return None
    return (min if side == 'left' else max)(rows, key=lambda row: row[1])


def _write(path, rows):
    Path(path).write_text(''.join(' '.join(str(int(row[0])) if i == 0 else f'{v:.6f}'
                                           for i, v in enumerate(row)) + '\n'
                                  for row in rows), encoding='utf-8')


def refine_fight(root, fight_id, side, epochs=30):
    """Train a temporary one-character detector; never overwrite reviewed labels."""
    from ultralytics import YOLO
    import torch

    if side not in {'left', 'right'}:
        raise ValueError('side must be left or right')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for local refinement')
    root, fight = Path(root).resolve(), Path(root).resolve() / 'fights' / fight_id
    manifest = json.loads((fight / 'manifest.json').read_text(encoding='utf-8'))
    reviewed, work = fight / 'labels_reviewed', fight / 'refinement' / side
    dataset = work / 'dataset'
    if dataset.exists():
        shutil.rmtree(dataset)
    samples = []
    for record in manifest['records']:
        image = Path(record['image'])
        row = _target(_rows(reviewed / f'{image.stem}.txt'), side)
        if row is not None:
            samples.append((image, row))
    if len(samples) < 12:
        raise ValueError(f'Need at least 12 reviewed {side}-side frames; found {len(samples)}')
    for index, (image, row) in enumerate(samples):
        split = 'val' if index % 10 == 0 else 'train'
        image_out = dataset / split / 'images' / image.name
        label_out = dataset / split / 'labels' / f'{image.stem}.txt'
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, image_out)
        _write(label_out, [[0, *row[1:]]])
    yaml = work / 'data.yaml'
    yaml.parent.mkdir(parents=True, exist_ok=True)
    yaml.write_text(f'path: {dataset.as_posix()}\ntrain: train/images\nval: val/images\n'
                    'names:\n  0: target\n', encoding='utf-8')
    run = work / 'run'
    base_model = Path('models/fighter_bootstrap.pt').resolve()
    if not base_model.is_file():
        raise FileNotFoundError(f'Missing base detector: {base_model}')
    model = YOLO(str(base_model))
    model.train(data=str(yaml), epochs=int(epochs), imgsz=640, batch=8, device=0,
                workers=2, project=str(run.parent), name=run.name, exist_ok=True,
                fliplr=0., close_mosaic=0, patience=10, verbose=False)
    refined = YOLO(str(run / 'weights' / 'best.pt'))
    changed = predicted = 0
    for record in manifest['records']:
        image = Path(record['image'])
        reviewed_label = reviewed / f'{image.stem}.txt'
        if reviewed_label.exists():
            continue
        result = refined.predict(str(image), imgsz=640, conf=.12, iou=.5, device=0,
                                 verbose=False, max_det=4)[0]
        candidates = result.boxes.data.detach().cpu().numpy()
        choice = [row for row in candidates
                  if (row[0] + row[2]) * .5 < cv2.imread(str(image)).shape[1] * .5
                  ] if side == 'left' else [row for row in candidates
                  if (row[0] + row[2]) * .5 >= cv2.imread(str(image)).shape[1] * .5]
        suggested = Path(record['label_suggestion'])
        before = _rows(suggested)
        kept = [row for row in before if (row[1] >= .5 if side == 'left' else row[1] < .5)]
        if choice:
            row = max(choice, key=lambda item: item[4])
            h, w = cv2.imread(str(image)).shape[:2]
            kept.append([0, (row[0] + row[2]) / (2 * w), (row[1] + row[3]) / (2 * h),
                         (row[2] - row[0]) / w, (row[3] - row[1]) / h])
            predicted += 1
        _write(suggested, sorted(kept, key=lambda row: row[1]))
        changed += 1
    report = {'fight_id': fight_id, 'side': side, 'reviewed_examples': len(samples),
              'unreviewed_recomputed': changed, 'target_predictions': predicted,
              'model': str((run / 'weights' / 'best.pt').resolve())}
    (work / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
