import argparse
import atexit
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import cv2
import numpy as np

from .capture import LatestCapture


def _clear_terminal():
    print('\x1b[2J\x1b[H', end='', flush=True)


def _set_cursor(visible):
    print('\x1b[?25h' if visible else '\x1b[?25l', end='', flush=True)


def _choose_menu(title, options, back_label='Back'):
    """Small dependency-free Windows TUI. Return option index or None for back."""
    import msvcrt
    selected = 0
    entries = list(options) + [back_label]
    while True:
        _clear_terminal()
        print(title + '\n')
        for index, label in enumerate(entries):
            if index == selected:
                print(f'\x1b[93m  {label}\x1b[0m')
            else:
                print(f'  {label}')
        key = msvcrt.getwch()
        if key in ('\x00', '\xe0'):
            key = {'H': 'up', 'P': 'down', 'K': 'back', 'M': 'select'}.get(
                msvcrt.getwch(), '')
        else:
            key = key.lower()
        if key in ('w', 'up'):
            selected = (selected - 1) % len(entries)
        elif key in ('s', 'down'):
            selected = (selected + 1) % len(entries)
        elif key in ('a', 'back', '\x08'):
            return None
        elif key in ('d', 'select', '\r'):
            return None if selected == len(options) else selected


def cameras():
    from cv2_enumerate_cameras import enumerate_cameras
    return list(enumerate_cameras(cv2.CAP_DSHOW))


def resolve_source(value):
    if value.lower() == 'obs':
        found = [c for c in cameras() if 'obs' in c.name.lower()]
        if len(found) != 1:
            raise ValueError(f'Expected one OBS camera, found {len(found)}. Run: python main.py cameras')
        print(f'Camera: {found[0].name}, DirectShow index={found[0].index}')
        return found[0].index
    if value.isdecimal():
        return int(value)
    if Path(value).is_file():
        return str(Path(value).resolve())
    raise ValueError(f'No camera or local video file: {value}')


def save_report(path, report):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'Report: {target.resolve()}')


def open_capture(args):
    if str(args.source).lower() == 'screen':
        from .screen_capture import LatestScreenCapture
        return LatestScreenCapture(args.window, fps=args.fps).start()
    return LatestCapture(resolve_source(args.source), width=args.width,
                         height=args.height, fps=args.fps).start()


def probe(args):
    capture = open_capture(args)
    seq = -1
    first = last = None
    first_seq = None
    last_frame = None
    frames = 0
    startup = time.perf_counter()
    try:
        while time.perf_counter() - (first or startup) < (args.seconds if first else 15):
            frame = capture.get(seq, .5)
            if frame is None:
                if capture.done:
                    break
                continue
            if first is None:
                first, first_seq = frame.received_at, frame.seq
            last, seq = frame.received_at, frame.seq
            last_frame = frame
            frames += 1
        if not frames:
            raise RuntimeError('No frames received within 15 seconds. Start OBS Virtual Camera.')
        if args.snapshot:
            target = Path(args.snapshot)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), last_frame.image):
                raise RuntimeError(f'Cannot write {target}')
        save_report(args.report, dict(capture=capture.info, observed_frames=frames,
                    capture_delivery_fps=(seq - first_seq) / max(last - first, 1e-6),
                    note='Delivery rate is not unique game-frame rate or end-to-end latency.'))
    finally:
        capture.close()


def collect(args):
    """Keep a small, diverse set of raw frames for later review/annotation."""
    capture = open_capture(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seq = -1
    first_at = next_due = None
    last_saved_at = None
    previous_signature = None
    items = []
    try:
        while True:
            frame = capture.get(seq, .5)
            if frame is None:
                if capture.done:
                    break
                continue
            seq = frame.seq
            if first_at is None:
                first_at = next_due = frame.received_at
            if frame.received_at - first_at >= args.seconds:
                break
            if frame.received_at < next_due:
                continue
            next_due = frame.received_at + args.interval
            signature = cv2.resize(cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY), (64, 36))
            novelty = 255. if previous_signature is None else float(
                np.mean(cv2.absdiff(signature, previous_signature)))
            forced = last_saved_at is None or frame.received_at - last_saved_at >= args.max_gap
            if novelty < args.novelty and not forced:
                continue
            name = f'frame_{len(items):04d}_seq_{frame.seq:07d}.jpg'
            target = output / name
            if not cv2.imwrite(str(target), frame.image, [cv2.IMWRITE_JPEG_QUALITY, args.quality]):
                raise RuntimeError(f'Cannot write {target}')
            items.append({'file': name, 'seq': frame.seq,
                          'elapsed_seconds': frame.received_at - first_at,
                          'novelty': novelty})
            previous_signature = signature
            last_saved_at = frame.received_at
        report = {'source': str(args.source), 'duration_seconds': args.seconds,
                  'saved_frames': len(items), 'capture': capture.info,
                  'selection': {'minimum_interval_seconds': args.interval,
                                'maximum_gap_seconds': args.max_gap,
                                'novelty_threshold': args.novelty},
                  'frames': items}
        save_report(output / 'manifest.json', report)
    finally:
        capture.close()


def draw(result, report, show_candidates=False, collection_status=None):
    image = result.frame.image.copy()
    age_ms = (time.perf_counter() - result.frame.received_at) * 1000
    stale = age_ms > 250
    if show_candidates and not stale:
        for detection in result.detections:
            x1, y1, x2, y2 = detection.box.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (140, 140, 140), 1)
            cv2.putText(image, f'{detection.confidence:.2f}', (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (200, 200, 200), 1)
    track = result.track
    if track.box is not None and not stale:
        x1, y1, x2, y2 = track.box.astype(int)
        color = (50, 230, 50) if track.status == 'detected' else (0, 180, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'Opponent #{track.track_id}: {track.status} {track.confidence:.2f}',
                    (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    lines = [
        f"{'STALE SOURCE' if stale else track.status.upper()} | inference {report.get('recent_inference_fps', 0):.1f} fps",
        f'detector {result.detector_ms:.1f} ms | read->result {result.application_ms:.1f} ms | read->display {age_ms:.1f} ms',
        'Q/Esc: quit | R: reacquire | D: candidates | S: one frame' +
        (' | C: hard samples' if collection_status is not None else ''),
    ]
    if collection_status is not None:
        lines.append(collection_status)
    for i, line in enumerate(lines):
        y = 25 + i * 26
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 4)
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    return image


def run(args):
    from .detector import Detector
    from .hard_samples import HardSampleCollector
    from .pipeline import Pipeline

    detector = Detector(args.model, args.imgsz, args.confidence, args.classes, args.prompts)
    capture = open_capture(args)
    pipeline = Pipeline(capture, detector).start()
    started = time.perf_counter()
    first_result = None
    next_report = started
    report = {}
    candidates = args.candidates
    collector = (HardSampleCollector(args.hard_samples, args.model,
                                     min_interval=args.hard_min_interval,
                                     max_gap=args.hard_max_gap,
                                     novelty=args.hard_novelty,
                                     max_frames=args.hard_max_frames)
                 if args.hard_samples else None)
    collecting = False
    last_collected_seq = -1
    window_created = False
    saved = False
    try:
        while True:
            now = time.perf_counter()
            result = pipeline.get()
            if now >= next_report:
                report = pipeline.report()
                if result is not None:
                    print(f"{report['recent_inference_fps']:.1f} fps | detector {result.detector_ms:.1f} ms | "
                          f"application {result.application_ms:.1f} ms | {result.track.status}", flush=True)
                next_report = now + 2
            if result is not None:
                if collector is not None and collecting and result.frame.seq > last_collected_seq:
                    collector.consider(result)
                    last_collected_seq = result.frame.seq
                if first_result is None:
                    first_result = now
                if args.snapshot and not saved and now - first_result >= 1:
                    target = Path(args.snapshot)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(target), draw(result, report, True)):
                        raise RuntimeError(f'Cannot write {target}')
                    saved = True
                if not args.headless:
                    if not window_created:
                        cv2.namedWindow('MCOC Vision', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('MCOC Vision', args.preview_width, args.preview_height)
                        cv2.moveWindow('MCOC Vision', args.preview_x, args.preview_y)
                        window_created = True
                    # Display the exact frame that produced this box, never a newer unrelated frame.
                    collection_status = None
                    if collector is not None:
                        state = 'RECORDING' if collecting else 'PAUSED'
                        collection_status = (f'HARD SAMPLES: {state} | '
                                             f'{len(collector.records)}/{collector.max_frames}')
                    cv2.imshow('MCOC Vision', draw(result, report, candidates,
                                                   collection_status))
            if pipeline.done:
                pipeline.get()  # propagate worker errors even at EOF
                break
            if args.seconds and first_result and now - first_result >= args.seconds:
                break
            if first_result is None and now - started > 20:
                raise RuntimeError('No inference results within 20 seconds; check camera source.')
            if not args.headless:
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord('q')):
                    break
                if window_created and cv2.getWindowProperty('MCOC Vision', cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord('r'):
                    pipeline.reset_event.set()
                if key == ord('d'):
                    candidates = not candidates
                if key == ord('c') and collector is not None:
                    collecting = not collecting
                    collector.set_enabled(collecting)
                    print(f"Hard-sample collection {'started' if collecting else 'paused'}: "
                          f'{len(collector.records)}/{collector.max_frames}', flush=True)
                if key == ord('s') and result is not None:
                    target = Path('artifacts/frames') / f'{time.time_ns()}.png'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if cv2.imwrite(str(target), result.frame.image):
                        print(f'Raw frame saved: {target}')
            time.sleep(.005 if args.headless else .01)
    finally:
        pipeline.close()
        capture.close()
        if window_created:
            cv2.destroyAllWindows()
        report = pipeline.report()
        if collector is not None:
            hard_report = collector.close()
            report['hard_samples'] = {
                'output': str(collector.root.resolve()),
                'saved_frames': hard_report['total'],
                'review_required': True,
            }
        report.update(model=args.model, imgsz=args.imgsz, confidence=args.confidence,
                      classes=args.classes, prompts=args.prompts,
                      preview_enabled=not args.headless)
        save_report(args.report, report)


def train(args):
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA; run setup.ps1.')
    if not Path(args.data).is_file():
        raise ValueError('Supply an existing, manually reviewed YOLO dataset YAML.')
    YOLO(args.model).train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                          batch=args.batch, device=0, workers=2, amp=True,
                          project='runs/train', name=args.name, seed=42,
                          fliplr=args.fliplr, flipud=0., close_mosaic=10)


def export(args):
    from ultralytics import YOLO
    if not Path(args.model).is_file():
        raise ValueError('Export requires an existing trained .pt model.')
    # TensorRT is optional; install it explicitly after benchmarking the baseline.
    YOLO(args.model).export(format='engine', imgsz=args.imgsz, half=True,
                           device=0, batch=1, dynamic=False, workspace=2)


def evaluate(args):
    from ultralytics import YOLO
    if not Path(args.model).is_file():
        raise ValueError('Evaluation requires an existing trained model.')
    if not Path(args.data).is_file():
        raise ValueError('Evaluation requires an existing dataset YAML.')
    metrics = YOLO(args.model).val(data=args.data, split=args.split,
                                   imgsz=args.imgsz, batch=args.batch,
                                   device=0, workers=2, plots=True)
    result = {'split': args.split, 'map50': float(metrics.box.map50),
              'map50_95': float(metrics.box.map),
              'precision': float(metrics.box.mp), 'recall': float(metrics.box.mr)}
    print(json.dumps(result, indent=2))


def prelabel(args):
    from .detector import Detector
    from .labeling import write_prelabels
    prompts = None if args.closed_set else args.prompts
    detector = Detector(args.model, args.imgsz, args.confidence, None, prompts,
                        augment=args.tta)
    manifest = write_prelabels(args.input, args.output, detector)
    print(json.dumps({key: manifest[key] for key in ('input', 'total', 'counts', 'review_required')},
                     indent=2, ensure_ascii=False))


def review(args):
    from .labeling import review_prelabels
    review_prelabels(args.input, args.window_width, args.window_height)


def curate(args):
    from .curation import curate_manifest
    result = curate_manifest(args.input, args.output, args.limit, args.ranges)
    print(json.dumps({key: result[key] for key in
                      ('source_total', 'total', 'index_ranges', 'counts')},
                     indent=2, ensure_ascii=False))


def register_battle(args):
    from .library import BattleLibrary
    print(json.dumps(BattleLibrary(args.library).register(
        args.input, args.fight_id, args.left, args.right), indent=2))


def session(args):
    from .library import identifier
    from .session import SessionRecorder
    identifier(args.player)
    capture = open_capture(args)
    recorder = SessionRecorder(capture, args.output, args.player,
                               sample_fps=args.sample_fps, quality=args.quality,
                               model=None if args.defer_labeling else args.model,
                               imgsz=args.imgsz, confidence=args.confidence,
                               absence_seconds=args.end_grace, prompts=args.prompts)
    recorder.run(args.seconds)


def confirm_fight(args):
    from .dataset import export_reviewed_fight
    result = export_reviewed_fight(args.library, args.fight_id,
                                   args.left, args.right, args.split)
    print(json.dumps({key: result[key] for key in
                      ('fight_id', 'identity', 'split', 'frames',
                       'unseen_pair_evaluation')}, indent=2, ensure_ascii=False))


def avatar_add(args):
    from .avatars import AvatarCatalog
    print(json.dumps(AvatarCatalog(Path(args.library) / 'avatars').add(
        args.character, args.image), indent=2, ensure_ascii=False))


def avatar_identify(args):
    from .avatars import AvatarCatalog
    print(json.dumps(AvatarCatalog(Path(args.library) / 'avatars').identify(
        args.image), indent=2, ensure_ascii=False))


def review_fight(args):
    from .labeling import review_prelabels
    fights = Path(args.library) / 'fights'
    if args.fight_id == 'latest':
        candidates = [path.parent for path in fights.glob('*/manifest.json')]
        if not candidates:
            raise ValueError('No processed fights are available for review')
        fight = max(candidates, key=lambda path: (path / 'manifest.json').stat().st_mtime)
    else:
        fight = fights / args.fight_id
    return review_prelabels(fight, args.window_width, args.window_height)


def refine_fight(args):
    from .fewshot import refine_fight as refine
    print(json.dumps(refine(args.library, args.fight_id, args.side, args.epochs), indent=2))


def read_lineup(args):
    from .lineup import read_lineup as recognize
    result = recognize(args.image)
    if args.output:
        save_report(args.output, result)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=True))


def _review_complete(fight):
    manifest = json.loads((fight / 'manifest.json').read_text(encoding='utf-8'))
    reviewed = fight / 'labels_reviewed'
    return bool(manifest['records']) and all(
        (reviewed / f"{Path(record['image']).stem}.txt").is_file()
        for record in manifest['records'])


def _review_swap_report(fight):
    manifest = json.loads((fight / 'manifest.json').read_text(encoding='utf-8'))
    changed = 0
    for record in manifest['records']:
        suggested = Path(record['label_suggestion'])
        rows = [line.split() for line in suggested.read_text(encoding='utf-8').splitlines()
                if line.strip()]
        if len(rows) == 2 and float(rows[0][1]) > float(rows[1][1]):
            changed += 1
    return changed


def _review_progress(fight, manifest):
    total = len(manifest.get('records', []))
    state_path = fight / 'review_state.json'
    reviewed = 0
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding='utf-8'))
        reviewed = len(state.get('reviewed', {}))
    return reviewed, total


def _run_refinement(root, fight, manifest, side):
    character_id = manifest['left_character'] if side == 'left' else manifest['right_character']
    character = character_id.replace('_', ' ').title()
    decision = _choose_menu(
        f'Learn {character}\nOnly unreviewed {side}-side boxes will be recalculated.',
        ['Start recalculation'],
    )
    if decision is None:
        return
    _clear_terminal()
    print(f'Learning {character} from your saved edits...\n', flush=True)
    from .fewshot import refine_fight as refine
    try:
        result = refine(root, fight.name, side, epochs=30)
        print(f"Recalculated: {result['unreviewed_recomputed']}")
        print(f"Boxes found: {result['target_predictions']}")
        print(f"Reviewed examples kept unchanged: {result['reviewed_examples']}")
    except Exception as error:
        print(f'Recalculation was not completed: {error}')
    print('\nPress any key to return.', end='', flush=True)
    import msvcrt
    msvcrt.getwch()


def _menu_one_fight(root, fight, manifest):
    left = manifest['left_character']
    right = manifest['right_character']
    left_name = left.replace('_', ' ').title()
    right_name = right.replace('_', ' ').title()
    while True:
        reviewed, total = _review_progress(fight, manifest)
        choice = _choose_menu(
            f'{left_name} vs {right_name}  [{reviewed}/{total}]',
            ['Open frame review', f'Learn {left_name}', f'Learn {right_name}'],
        )
        if choice is None:
            return
        if choice in (1, 2):
            _run_refinement(root, fight, manifest, 'left' if choice == 1 else 'right')
            continue

        outcome = review_fight(SimpleNamespace(library=str(root), fight_id=fight.name,
                                               window_width=1152, window_height=648))
        if outcome and outcome.get('action') == 'refine':
            _run_refinement(root, fight, manifest, outcome['side'])
            continue
        if not _review_complete(fight):
            continue

        swaps = _review_swap_report(fight)
        decision = _choose_menu(
            f'Review complete\nRole order corrected in {swaps} frames.',
            [f'Export to {right_name} dataset', 'Keep in Review'],
        )
        if decision == 0:
            __import__('mcoc_vision.dataset', fromlist=['export_reviewed_fight']) \
                .export_reviewed_fight(root, fight.name, left, right, 'train')
            return


def _menu_review(root=Path('data')):
    while True:
        fights = []
        for manifest_path in (root / 'fights').glob('*/manifest.json'):
            fight = manifest_path.parent
            if not (root / 'exports' / f'{fight.name}.json').exists():
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                fights.append((fight, manifest))
        fights.sort(key=lambda item: item[0].stat().st_mtime)
        labels = []
        for index, (fight, manifest) in enumerate(fights, 1):
            opponent = (manifest.get('right_character') or 'unknown').replace('_', ' ').title()
            reviewed, total = _review_progress(fight, manifest)
            suffix = (f' [{reviewed}/{total}]' if 0 < reviewed < total else
                      (' [ready]' if total and reviewed >= total else ''))
            labels.append(f'{opponent}{suffix}')
        choice = _choose_menu('Review', labels)
        if choice is None:
            return
        fight, manifest = fights[choice]
        _menu_one_fight(root, fight, manifest)


def interactive_menu():
    _set_cursor(False)
    atexit.register(_set_cursor, True)
    common = dict(source='screen', window='Champions', width=1280, height=720,
                  fps=60, seconds=0)
    while True:
        choice = _choose_menu('MCOC Vision', [
            'Review recorded fights',
            'Record session from SET LINEUP',
            'Live preview (no recording)',
        ], back_label='Exit')
        if choice is None:
            _clear_terminal()
            return
        if choice == 0:
            _menu_review()
        elif choice == 1:
            session(SimpleNamespace(**common, output='data', player='unknown', sample_fps=10,
                quality=92, end_grace=4., model='models/yolov8s-worldv2.pt', imgsz=640,
                confidence=.12, prompts=['video game fighter', 'superhero',
                'human character', 'person'], defer_labeling=False))
        elif choice == 2:
            run(SimpleNamespace(**common, report='artifacts/run-report.json', snapshot=None,
                model='models/yolov8s-worldv2.pt', imgsz=640, confidence=.12, classes=[0],
                prompts=['video game fighter', 'superhero', 'human character', 'person'],
                headless=False, candidates=True, preview_x=50, preview_y=50,
                preview_width=1280, preview_height=720, hard_samples=None,
                hard_min_interval=.1, hard_max_gap=.5, hard_novelty=2., hard_max_frames=240))


def main():
    if len(sys.argv) == 1:
        interactive_menu()
        return
    parser = argparse.ArgumentParser(description='MCOC opponent detection, CUDA only')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('cameras', help='List named DirectShow cameras')
    for name in ('probe', 'run', 'collect'):
        p = commands.add_parser(name)
        p.add_argument('--source', default='obs', help='obs, DirectShow index, or video path')
        p.add_argument('--window', default='Champions', help='Window title for --source screen')
        p.add_argument('--width', type=int, default=1280)
        p.add_argument('--height', type=int, default=720)
        p.add_argument('--fps', type=float, default=60)
        default_seconds = 10 if name == 'probe' else (30 if name == 'collect' else 0)
        p.add_argument('--seconds', type=float, default=default_seconds)
        p.add_argument('--report', default=f'artifacts/{name}-report.json')
        if name in ('probe', 'run'):
            p.add_argument('--snapshot', help='Save one local diagnostic image')
        if name == 'run':
            p.add_argument('--model', default='models/yolo11n.pt')
            p.add_argument('--imgsz', type=int, default=640)
            p.add_argument('--confidence', type=float, default=.25)
            p.add_argument('--classes', type=int, nargs='+', default=[0])
            p.add_argument('--prompts', nargs='+', help='Enable YOLO-World with these class prompts')
            p.add_argument('--headless', action='store_true')
            p.add_argument('--candidates', action='store_true')
            p.add_argument('--preview-x', type=int, default=50,
                           help='Screen X coordinate for the separate preview window')
            p.add_argument('--preview-y', type=int, default=50,
                           help='Screen Y coordinate for the separate preview window')
            p.add_argument('--preview-width', type=int, default=1280)
            p.add_argument('--preview-height', type=int, default=720)
            p.add_argument('--hard-samples',
                           help='Review-ready output directory; press C to start/pause collection')
            p.add_argument('--hard-max-frames', type=int, default=240)
            p.add_argument('--hard-min-interval', type=float, default=.10)
            p.add_argument('--hard-max-gap', type=float, default=.50)
            p.add_argument('--hard-novelty', type=float, default=2.0)
        if name == 'collect':
            p.add_argument('--output', default='artifacts/collected')
            p.add_argument('--interval', type=float, default=.5)
            p.add_argument('--max-gap', type=float, default=2.)
            p.add_argument('--novelty', type=float, default=6.)
            p.add_argument('--quality', type=int, default=92)
    p = commands.add_parser('train')
    p.add_argument('--data', required=True)
    p.add_argument('--model', default='models/yolo11n.pt')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--name', default='fighters')
    p.add_argument('--fliplr', type=float, default=.5,
                   help='Horizontal flip probability; safe because both sides use class fighter')
    p = commands.add_parser('export')
    p.add_argument('--model', required=True)
    p.add_argument('--imgsz', type=int, default=640)
    p = commands.add_parser('evaluate', help='Evaluate without adding the split to training')
    p.add_argument('--model', required=True)
    p.add_argument('--data', default='data/detector/data.yaml')
    p.add_argument('--split', choices=('val', 'test'), default='test')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=8)
    p = commands.add_parser('prelabel')
    p.add_argument('--input', required=True)
    p.add_argument('--output', default='artifacts/prelabels')
    p.add_argument('--model', default='models/yolov8s-worldv2.pt')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--confidence', type=float, default=.15)
    p.add_argument('--prompts', nargs='+', default=['video game fighter', 'superhero',
                                                    'human character', 'person'])
    p.add_argument('--closed-set', action='store_true',
                   help='Use classes already stored in a trained model; ignore text prompts')
    p.add_argument('--tta', action='store_true',
                   help='Slower test-time augmentation for higher-quality offline suggestions')
    p = commands.add_parser('review')
    p.add_argument('--input', default='artifacts/prelabels-live')
    p.add_argument('--window-width', type=int, default=1152)
    p.add_argument('--window-height', type=int, default=648)
    p = commands.add_parser('read-lineup', help='Read three English matchup rows')
    p.set_defaults(command='read_lineup')
    p.add_argument('--image', required=True)
    p.add_argument('--output')
    p = commands.add_parser('curate')
    p.add_argument('--input', required=True, help='Hard-sample directory with manifest.json')
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=64)
    p.add_argument('--ranges', help='Inclusive zero-based record ranges, e.g. 1-99,132-239')
    p = commands.add_parser('register-battle')
    p.set_defaults(command='register_battle')
    p.add_argument('--input', required=True, help='Manifest containing exactly one battle')
    p.add_argument('--library', default='data')
    p.add_argument('--fight-id', required=True)
    p.add_argument('--left', required=True, help='Canonical character ID')
    p.add_argument('--right', required=True, help='Canonical character ID')
    p = commands.add_parser('session', help='Automatically segment and record fights')
    p.add_argument('--source', default='screen', help='screen, obs, camera index, or video path')
    p.add_argument('--window', default='Champions', help='Window title for --source screen')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fps', type=float, default=60)
    p.add_argument('--seconds', type=float, default=0, help='0 runs until Ctrl+C')
    p.add_argument('--output', default='data')
    p.add_argument('--player', default='unknown',
                   help='Optional fixed left character ID; omit when it changes between fights')
    p.add_argument('--sample-fps', type=float, default=10)
    p.add_argument('--quality', type=int, default=92)
    p.add_argument('--end-grace', type=float, default=4.0,
                   help='HUD absence before closing a fight')
    p.add_argument('--model', default='models/yolov8s-worldv2.pt')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--confidence', type=float, default=.12)
    p.add_argument('--prompts', nargs='+', default=['video game fighter', 'superhero',
                                                    'human character', 'person'])
    p.add_argument('--defer-labeling', action='store_true',
                   help='Record and queue only; do not run background GPU pre-labeling')
    p = commands.add_parser('confirm-fight', help='Export one fully reviewed fight')
    p.set_defaults(command='confirm_fight')
    p.add_argument('--library', default='data')
    p.add_argument('--fight-id', required=True)
    p.add_argument('--left', required=True, help='Character shown on the left in source frames')
    p.add_argument('--right', required=True, help='Character shown on the right in source frames')
    p.add_argument('--split', choices=('train', 'val', 'test'), default='train',
                   help='Assign the whole fight to one split; use test for unseen pairs')
    p = commands.add_parser('avatar-add', help='Add one static avatar reference')
    p.set_defaults(command='avatar_add')
    p.add_argument('--library', default='data')
    p.add_argument('--character', required=True)
    p.add_argument('--image', required=True)
    p = commands.add_parser('avatar-identify', help='Test avatar recognition on an image')
    p.set_defaults(command='avatar_identify')
    p.add_argument('--library', default='data')
    p.add_argument('--image', required=True)
    p = commands.add_parser('review-fight', help='Review one processed fight or the latest one')
    p.set_defaults(command='review_fight')
    p.add_argument('--library', default='data')
    p.add_argument('--fight-id', default='latest')
    p.add_argument('--window-width', type=int, default=1152)
    p.add_argument('--window-height', type=int, default=648)
    p = commands.add_parser('refine-fight', help='Learn one side from reviewed frames only')
    p.set_defaults(command='refine_fight')
    p.add_argument('--library', default='data')
    p.add_argument('--fight-id', required=True)
    p.add_argument('--side', choices=('left', 'right'), required=True)
    p.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    if hasattr(args, 'seconds') and args.seconds < 0:
        parser.error('--seconds must be nonnegative')
    if hasattr(args, 'imgsz') and (args.imgsz < 32 or args.imgsz % 32):
        parser.error('--imgsz must be a positive multiple of 32')
    if hasattr(args, 'limit') and args.limit <= 0:
        parser.error('--limit must be positive')
    if hasattr(args, 'sample_fps') and args.sample_fps <= 0:
        parser.error('--sample-fps must be positive')
    if hasattr(args, 'quality') and not 1 <= args.quality <= 100:
        parser.error('--quality must be between 1 and 100')
    if hasattr(args, 'end_grace') and args.end_grace <= 0:
        parser.error('--end-grace must be positive')
    if args.command == 'cameras':
        for camera in cameras():
            print(f'{camera.index}: {camera.name}')
    else:
        globals()[args.command](args)
