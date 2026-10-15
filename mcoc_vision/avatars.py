"""Static avatar catalog and conservative post-fight template matching."""
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from .library import identifier


class AvatarCatalog:
    def __init__(self, root, threshold=.84, minimum_margin=.04):
        self.root = Path(root).resolve()
        self.threshold = float(threshold)
        self.minimum_margin = float(minimum_margin)

    def add(self, character, image_path):
        identifier(character)
        source = Path(image_path).resolve()
        image = cv2.imread(str(source))
        if image is None or min(image.shape[:2]) < 16:
            raise ValueError('Avatar must be a readable image at least 16x16 pixels')
        target = self.root / f'{character}.png'
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.png.tmp.png')
        if not cv2.imwrite(str(temporary), image):
            raise RuntimeError(f'Cannot write avatar: {target}')
        temporary.replace(target)
        return {'character_id': character, 'avatar': str(target)}

    @staticmethod
    def _score(scene, template):
        scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        scores = []
        for scale in (.80, .90, 1.0, 1.10, 1.20):
            width = max(8, round(template_gray.shape[1] * scale))
            height = max(8, round(template_gray.shape[0] * scale))
            if width > scene_gray.shape[1] or height > scene_gray.shape[0]:
                continue
            resized = cv2.resize(template_gray, (width, height), interpolation=cv2.INTER_AREA)
            response = cv2.matchTemplate(scene_gray, resized, cv2.TM_CCOEFF_NORMED)
            scores.append(float(np.max(response)))
        return max(scores, default=-1.0)

    def identify(self, image_path):
        scene = cv2.imread(str(Path(image_path)))
        if scene is None:
            raise ValueError(f'Unreadable identity image: {image_path}')
        ranked = []
        for path in sorted(self.root.glob('*.png')):
            template = cv2.imread(str(path))
            if template is not None:
                ranked.append((self._score(scene, template), path.stem))
        ranked.sort(reverse=True)
        best_score, best_id = ranked[0] if ranked else (-1.0, None)
        second_score = ranked[1][0] if len(ranked) > 1 else -1.0
        accepted = (best_id is not None and best_score >= self.threshold and
                    best_score - second_score >= self.minimum_margin)
        return {'character_id': best_id if accepted else None,
                'candidate': best_id, 'score': best_score,
                'second_score': second_score,
                'margin': best_score - second_score,
                'accepted': accepted,
                'threshold': self.threshold,
                'minimum_margin': self.minimum_margin}
