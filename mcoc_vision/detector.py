from pathlib import Path
import time

import cv2
import numpy as np

from .tracking import Detection


class Detector:
    def __init__(self, model='models/yolo11n.pt', imgsz=640, confidence=.25,
                 classes=(0,), prompts=None, augment=False):
        import torch
        from ultralytics import YOLO, YOLOWorld

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Run setup.ps1; CPU fallback is intentionally disabled.')
        cv2.setNumThreads(1)
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = True
        Path(model).parent.mkdir(parents=True, exist_ok=True)
        if prompts:
            self.model = YOLOWorld(model)
            self.model.set_classes(prompts)
        else:
            self.model = YOLO(model, task='detect')
        self.kwargs = dict(device=0, imgsz=imgsz, conf=confidence, iou=.5,
                           quantize=16, verbose=False, max_det=8, rect=False,
                           augment=bool(augment))
        if not prompts and classes is not None:
            self.kwargs['classes'] = list(classes)
        # Warmup before opening the source so it cannot inflate stream latency.
        for _ in range(5):
            self.model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), **self.kwargs)
        torch.cuda.synchronize()
        self.name = str(model)
        self.prompts = list(prompts) if prompts else None
        self.augment = bool(augment)

    def detect(self, image):
        started = time.perf_counter()
        result = self.model.predict(image, **self.kwargs)[0]
        # One synchronized transfer; timing includes preprocessing/NMS/transfer.
        rows = result.boxes.data.detach().cpu().numpy()
        detections = [Detection(row[:4].astype(float), float(row[4])) for row in rows]
        return detections, (time.perf_counter() - started) * 1000
