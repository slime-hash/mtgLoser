"""Read the English series-lineup screen with built-in Windows OCR."""
import asyncio
from pathlib import Path
import re

import cv2


def character_id(text):
    value = re.sub(r'[^A-Z0-9]+', '_', text.upper()).strip('_').lower()
    return value or None


async def _ocr_async(path):
    from winrt.windows.globalization import Language
    from winrt.windows.graphics.imaging import BitmapDecoder
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.storage import StorageFile

    file = await StorageFile.get_file_from_path_async(str(Path(path).resolve()))
    stream = await file.open_read_async()
    decoder = await BitmapDecoder.create_async(stream)
    bitmap = await decoder.get_software_bitmap_async()
    engine = OcrEngine.try_create_from_language(Language('en-US'))
    if engine is None:
        raise RuntimeError('Built-in Windows English OCR is unavailable')
    result = await engine.recognize_async(bitmap)
    lines = []
    for line in result.lines:
        words = list(line.words)
        if not words:
            continue
        left = min(word.bounding_rect.x for word in words)
        top = min(word.bounding_rect.y for word in words)
        right = max(word.bounding_rect.x + word.bounding_rect.width for word in words)
        bottom = max(word.bounding_rect.y + word.bounding_rect.height for word in words)
        lines.append({'text': line.text, 'x': left, 'y': top,
                      'width': right - left, 'height': bottom - top})
    return lines


def _ocr(path):
    return asyncio.run(_ocr_async(path))


def _normalized(text):
    return re.sub(r'[^A-Z]+', ' ', text.upper()).strip()


def _is_title(lines):
    combined = ' '.join(_normalized(item['text']) for item in lines)
    # Whole words in the correct order: SET inside SETTINGS must never match.
    return re.search(r'\bSERIES\s+MATCH\s+SET\s+LINEUP\b', combined) is not None


def _title_box(lines):
    direct = [item for item in lines
              if 'SERIES' in _normalized(item['text']) and
              'LINEUP' in _normalized(item['text'])]
    title_lines = direct or [item for item in lines if any(
        token in _normalized(item['text'])
        for token in ('SERIES', 'MATCH', 'LINEUP'))]
    if not title_lines:
        return None
    left = min(item['x'] for item in title_lines)
    top = min(item['y'] for item in title_lines)
    right = max(item['x'] + item['width'] for item in title_lines)
    bottom = max(item['y'] + item['height'] for item in title_lines)
    return [round(left), round(top), round(right), round(bottom)]


def _is_character_name(text):
    cleaned = text.strip()
    normalized = _normalized(cleaned)
    if len(normalized.replace(' ', '')) < 3 or any(ch.isdigit() for ch in cleaned):
        return False
    if cleaned != cleaned.upper():
        return False
    excluded = {'SERIES MATCH SET LINEUP', 'ACCEPT', 'SETTINGS', 'RANK', 'LEVEL', 'VS'}
    return normalized not in excluded and not normalized.startswith(('RANK ', 'LEVEL '))


def read_lineup(image_path):
    image_path = Path(image_path).resolve()
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f'Unreadable lineup image: {image_path}')
    h, w = image.shape[:2]
    lines = _ocr(image_path)
    if not _is_title(lines):
        return {'schema': 1, 'source': str(image_path), 'language': 'en-US',
                'is_lineup_screen': False, 'rounds': [], 'needs_review': True,
                'ocr_lines': lines}

    candidates = {'left': [], 'right': []}
    for item in lines:
        text = item['text'].strip()
        center_x = (item['x'] + item['width'] * .5) / w
        center_y = (item['y'] + item['height'] * .5) / h
        if center_y < .20 or center_y > .88 or not _is_character_name(text):
            continue
        side = 'left' if center_x < .5 else 'right'
        candidates[side].append((center_y, text))

    names = {}
    for side in ('left', 'right'):
        chosen = sorted(candidates[side])[-3:]
        chosen.sort()
        names[side] = [{'display_name': text, 'character_id': character_id(text),
                        'needs_review': False, 'ocr_lines': [text]}
                       for _, text in chosen]
        while len(names[side]) < 3:
            names[side].append({'display_name': None, 'character_id': None,
                                'needs_review': True, 'ocr_lines': []})

    rounds = [{'round': index + 1, 'left': names['left'][index],
               'right': names['right'][index]} for index in range(3)]
    return {'schema': 1, 'source': str(image_path), 'language': 'en-US',
            'is_lineup_screen': True, 'rounds': rounds,
            'title_box': _title_box(lines),
            'needs_review': any(side['needs_review'] for round_ in rounds
                                for side in (round_['left'], round_['right'])),
            'ocr_lines': lines}
