"""Conservative filtering for suggested fighter boxes.

Suggestions are never promoted to training labels without review.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import json

import cv2
import numpy as np

from .tracking import Detection, iou


def intersection_over_smaller(a, b):
    wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(wh))
    smaller = min(float(np.prod(a[2:] - a[:2])), float(np.prod(b[2:] - b[:2])))
    return intersection / max(smaller, 1e-6)


def fighter_suggestions(detections, shape, max_boxes=2):
    """Filter HUD-sized boxes and collapse nested/overlapping duplicates."""
    h, w = shape[:2]
    candidates = []
    for detection in detections:
        box = detection.box
        bw, bh = box[2] - box[0], box[3] - box[1]
        cy = (box[1] + box[3]) * .5
        if bh < .15 * h or bw < .025 * w or bw > .85 * w:
            continue
        if cy < .22 * h:  # portraits and health-bar UI live at the top
            continue
        candidates.append(detection)

    groups = []
    for detection in sorted(candidates, key=lambda d: np.prod(d.box[2:] - d.box[:2]), reverse=True):
        duplicate = next((group for group in groups if
                          iou(detection.box, group[0].box) > .45 or
                          intersection_over_smaller(detection.box, group[0].box) > .75), None)
        if duplicate is None:
            groups.append([detection])
        else:
            duplicate.append(detection)

    selected = []
    for group in groups:
        largest = max(group, key=lambda d: np.prod(d.box[2:] - d.box[:2]))
        strongest = max(group, key=lambda d: d.confidence)
        selected.append(largest if strongest.confidence - largest.confidence < .15 else strongest)

    selected.sort(key=lambda d: d.confidence, reverse=True)
    return selected[:max_boxes]


def yolo_row(box, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    return (0, ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h,
            (x2 - x1) / w, (y2 - y1) / h)


def box_from_yolo(row, shape):
    h, w = shape[:2]
    _, cx, cy, bw, bh = row
    return np.array([(cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h], dtype=float)


def display_to_image_point(point, display_size, image_shape):
    """Map a mouse position in a resized OpenCV window to source pixels."""
    display_w, display_h = display_size
    image_h, image_w = image_shape[:2]
    if display_w <= 0 or display_h <= 0:
        return tuple(int(value) for value in point)
    x = int(np.clip(round(point[0] * image_w / display_w), 0, image_w - 1))
    y = int(np.clip(round(point[1] * image_h / display_h), 0, image_h - 1))
    return x, y


def box_edit_mode(point, box, tolerance):
    """Return the edge/corner being grabbed, or ``move`` for the interior."""
    x, y = point
    x1, y1, x2, y2 = box
    near_left = abs(x - x1) <= tolerance and y1 - tolerance <= y <= y2 + tolerance
    near_right = abs(x - x2) <= tolerance and y1 - tolerance <= y <= y2 + tolerance
    near_top = abs(y - y1) <= tolerance and x1 - tolerance <= x <= x2 + tolerance
    near_bottom = abs(y - y2) <= tolerance and x1 - tolerance <= x <= x2 + tolerance
    horizontal = 'left' if near_left else ('right' if near_right else '')
    vertical = 'top' if near_top else ('bottom' if near_bottom else '')
    if horizontal and vertical:
        return f'{vertical}_{horizontal}'
    if horizontal or vertical:
        return horizontal or vertical
    if x1 < x < x2 and y1 < y < y2:
        return 'move'
    return None


def edit_box(original, mode, start, current, shape, minimum=10):
    """Move or resize one box while keeping it inside the source image."""
    h, w = shape[:2]
    box = np.asarray(original, dtype=float).copy()
    dx, dy = current[0] - start[0], current[1] - start[1]
    if mode == 'move':
        width, height = box[2] - box[0], box[3] - box[1]
        x1 = np.clip(box[0] + dx, 0, max(0, w - width))
        y1 = np.clip(box[1] + dy, 0, max(0, h - height))
        return np.array([x1, y1, x1 + width, y1 + height], dtype=float)
    if 'left' in mode:
        box[0] = np.clip(box[0] + dx, 0, box[2] - minimum)
    if 'right' in mode:
        box[2] = np.clip(box[2] + dx, box[0] + minimum, w - 1)
    if 'top' in mode:
        box[1] = np.clip(box[1] + dy, 0, box[3] - minimum)
    if 'bottom' in mode:
        box[3] = np.clip(box[3] + dy, box[1] + minimum, h - 1)
    return box


def draw_suggestions(image, suggestions):
    preview = image.copy()
    for index, detection in enumerate(suggestions, 1):
        x1, y1, x2, y2 = detection.box.astype(int)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 200, 255), 3)
        cv2.putText(preview, f'{index}: {detection.confidence:.2f}',
                    (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (0, 0, 0), 4)
        cv2.putText(preview, f'{index}: {detection.confidence:.2f}',
                    (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (0, 200, 255), 2)
    return preview


def write_prelabels(input_dir, output_dir, detector):
    source = Path(input_dir)
    output = Path(output_dir)
    labels = output / 'labels_suggested'
    previews = output / 'previews'
    labels.mkdir(parents=True, exist_ok=True)
    previews.mkdir(parents=True, exist_ok=True)
    records = []
    paths = sorted(p for p in source.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    for index, path in enumerate(paths, 1):
        image = cv2.imread(str(path))
        if image is None:
            records.append({'image': str(path), 'status': 'unreadable', 'boxes': []})
            continue
        detections, detector_ms = detector.detect(image)
        suggestions = fighter_suggestions(detections, image.shape)
        rows = [yolo_row(d.box, image.shape) for d in suggestions]
        label_path = labels / f'{path.stem}.txt'
        label_path.write_text(''.join(' '.join(f'{value:.6f}' if i else str(value)
                                                for i, value in enumerate(row)) + '\n'
                                      for row in rows), encoding='utf-8')
        cv2.imwrite(str(previews / f'{path.stem}.jpg'), draw_suggestions(image, suggestions),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({'image': str(path.resolve()), 'label_suggestion': str(label_path.resolve()),
                        'preview': str((previews / f'{path.stem}.jpg').resolve()),
                        'status': 'needs_review', 'box_count': len(suggestions),
                        'confidences': [d.confidence for d in suggestions],
                        'detector_ms': detector_ms})
        if index % 10 == 0 or index == len(paths):
            print(f'Pre-labeled {index}/{len(paths)}', flush=True)
    manifest = {'input': str(source.resolve()), 'total': len(paths),
                'counts': {str(n): sum(r.get('box_count') == n for r in records) for n in range(3)},
                'review_required': True,
                'note': 'labels_suggested are untrusted and must not be used for training before review.',
                'records': records}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return manifest


def review_prelabels(prelabels_dir, window_width=1152, window_height=648):
    """Interactive local box review using only OpenCV."""
    root = Path(prelabels_dir)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    records = [record for record in manifest['records'] if record.get('status') != 'unreadable']
    reviewed_dir = root / 'labels_reviewed'
    reviewed_dir.mkdir(exist_ok=True)
    state_path = root / 'review_state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'reviewed': {}}
    index = min(state.get('next_index', 0), max(len(records) - 1, 0))
    window = 'MCOC label review'
    # A normal resizable window lets the reviewer maximize it.  The rendered
    # frame is fitted to the current client area on every iteration below.
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, window_width, window_height)
    drag = {'mode': None, 'index': None, 'start': None, 'current': None, 'original': None}

    def read_boxes(record, image):
        reviewed = reviewed_dir / (Path(record['image']).stem + '.txt')
        source = reviewed if reviewed.exists() else Path(record['label_suggestion'])
        boxes = []
        if source.exists():
            for line in source.read_text(encoding='utf-8').splitlines():
                values = [float(value) for value in line.split()]
                if len(values) == 5:
                    boxes.append(box_from_yolo(values, image.shape))
        if len(boxes) > 2:
            # Old review files could contain a third accidental/background box.
            # Keep the two substantial candidates visible instead of assigning
            # duplicate fighter numbers.
            boxes = sorted(boxes, key=lambda box: np.prod(box[2:] - box[:2]), reverse=True)[:2]
        return boxes

    image = None
    boxes = []
    display_size = [window_width, window_height]

    def image_point(x, y):
        return display_to_image_point((x, y), display_size, image.shape)

    def reset_drag():
        drag.update(mode=None, index=None, start=None, current=None, original=None)

    def mouse(event, x, y, flags, _):
        nonlocal boxes
        x, y = image_point(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            tolerance = max(6, round(10 * image.shape[1] / display_size[0]))
            hits = []
            for box_index, box in enumerate(boxes):
                mode = box_edit_mode((x, y), box, tolerance)
                if mode:
                    hits.append((box_index, mode, np.prod(box[2:] - box[:2])))
            if hits:
                box_index, mode, _ = min(hits, key=lambda item: item[2])
                drag.update(mode=mode, index=box_index, start=(x, y), current=(x, y),
                            original=boxes[box_index].copy())
            else:
                drag.update(mode='new', index=None, start=(x, y), current=(x, y), original=None)
        elif event == cv2.EVENT_MOUSEMOVE and drag['start'] is not None:
            drag['current'] = (x, y)
            if drag['mode'] != 'new':
                boxes[drag['index']] = edit_box(drag['original'], drag['mode'], drag['start'],
                                                drag['current'], image.shape)
        elif event == cv2.EVENT_LBUTTONUP and drag['start'] is not None:
            if drag['mode'] == 'new':
                x1, y1 = drag['start']
                candidate = np.array([min(x1, x), min(y1, y), max(x1, x), max(y1, y)], dtype=float)
                if (len(boxes) < 2 and candidate[2] - candidate[0] >= 10 and
                        candidate[3] - candidate[1] >= 10):
                    boxes.append(candidate)
            else:
                boxes[drag['index']] = edit_box(drag['original'], drag['mode'], drag['start'],
                                                (x, y), image.shape)
            reset_drag()
        elif event == cv2.EVENT_RBUTTONDOWN and boxes:
            containing = [(i, np.prod(box[2:] - box[:2])) for i, box in enumerate(boxes)
                          if box[0] <= x <= box[2] and box[1] <= y <= box[3]]
            if containing:
                boxes.pop(min(containing, key=lambda item: item[1])[0])

    cv2.setMouseCallback(window, mouse)

    def save_current(record):
        target = reviewed_dir / (Path(record['image']).stem + '.txt')
        ordered = sorted(boxes, key=lambda box: (box[0] + box[2]) * .5)
        # With two fighters, their relative order defines the roles even when
        # both happen to stand on the same half of the screen.
        roles = ([0, 1] if len(ordered) == 2 else
                 ([0] if ordered and (ordered[0][0] + ordered[0][2]) * .5 < image.shape[1] * .5
                  else [1]))
        target.write_text(''.join(' '.join(f'{value:.6f}' if i else str(value)
                                           for i, value in enumerate((role, *yolo_row(box, image.shape)[1:]))) + '\n'
                                  for role, box in zip(roles, ordered)), encoding='utf-8')
        state['reviewed'][Path(record['image']).name] = len(boxes)
        state['next_index'] = min(index + 1, len(records) - 1)
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')

    try:
        loaded_index = None
        while records:
            record = records[index]
            if loaded_index != index:
                image = cv2.imread(record['image'])
                if image is None:
                    raise RuntimeError(f"Cannot read {record['image']}")
                boxes = read_boxes(record, image)
                loaded_index = index
            canvas = image.copy()
            ordered_for_display = sorted(boxes, key=lambda box: (box[0] + box[2]) * .5)
            display_roles = ([1, 2] if len(ordered_for_display) == 2 else
                             ([1] if ordered_for_display and
                               (ordered_for_display[0][0] + ordered_for_display[0][2]) * .5 < image.shape[1] * .5
                              else [2]))
            role_by_id = {id(box): role for box, role in zip(ordered_for_display, display_roles)}
            for box in boxes:
                number = role_by_id[id(box)]
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(canvas, str(number), (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 220, 0), 2)
            if drag['mode'] == 'new' and drag['start'] and drag['current']:
                cv2.rectangle(canvas, drag['start'], drag['current'], (0, 200, 255), 2)
            status = 'REVIEWED' if Path(record['image']).name in state['reviewed'] else 'NEEDS REVIEW'
            text = f'{index + 1}/{len(records)} {status} | boxes {len(boxes)}'
            help_text = ('LMB empty:add | inside:move | edge:resize | RMB:delete | +/-:zoom | '
                         'A/D:frames | Space/Enter:save+next | 1:learn left | 2:learn right | Q:menu')
            for line, y in ((text, 25), (help_text, 52)):
                cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 4)
                cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
            image_h, image_w = image.shape[:2]
            try:
                _, _, client_w, client_h = cv2.getWindowImageRect(window)
            except cv2.error:
                client_w, client_h = window_width, window_height
            client_w = max(1, int(client_w))
            client_h = max(1, int(client_h))
            scale = min(client_w / image_w, client_h / image_h)
            display_size[:] = [max(1, round(image_w * scale)), max(1, round(image_h * scale))]
            shown = cv2.resize(canvas, tuple(display_size), interpolation=cv2.INTER_AREA)
            cv2.imshow(window, shown)
            key = cv2.waitKeyEx(20)
            if key in (ord('q'), ord('Q'), 8) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                save_current(record)
                break
            if key in (ord('1'), ord('2')):
                save_current(record)
                return {'action': 'refine', 'side': 'left' if key == ord('1') else 'right'}
            if key in (ord(' '), 10, 13):
                save_current(record)
                if index < len(records) - 1:
                    index += 1
                    loaded_index = None
                else:
                    break
            elif key in (ord('a'), ord('A'), 81, 2424832) and index > 0:
                index -= 1
                loaded_index = None
            elif key in (ord('d'), ord('D'), 83, 2555904) and index < len(records) - 1:
                index += 1
                loaded_index = None
            elif key == ord('c'):
                boxes = []
            elif key == ord('r'):
                suggested = Path(record['label_suggestion'])
                reviewed = reviewed_dir / (Path(record['image']).stem + '.txt')
                if reviewed.exists():
                    reviewed.unlink()
                boxes = []
                for line in suggested.read_text(encoding='utf-8').splitlines():
                    values = [float(value) for value in line.split()]
                    if len(values) == 5:
                        boxes.append(box_from_yolo(values, image.shape))
                if len(boxes) > 2:
                    boxes = sorted(boxes, key=lambda box: np.prod(box[2:] - box[:2]),
                                   reverse=True)[:2]
            elif key in (ord('+'), ord('=')):
                window_width = min(1920, round(window_width * 1.2))
                window_height = min(1080, round(window_height * 1.2))
            elif key in (ord('-'), ord('_')):
                window_width = max(640, round(window_width / 1.2))
                window_height = max(360, round(window_height / 1.2))
    finally:
        cv2.destroyWindow(window)
    print(f"Reviewed {len(state['reviewed'])}/{len(records)}. Verified labels: {reviewed_dir.resolve()}")
    return {'action': 'back'}
