from typing import List, Optional, Tuple
import numpy as np

class PlateDetector:
    def __init__(self, model, imgsz: int = 640, conf: float = 0.25):
        self.model = model
        self.imgsz = imgsz
        self.conf = conf

    @staticmethod
    def _clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        return x1, y1, x2, y2

    def best_bbox(self, img_bgr: np.ndarray) -> Tuple[Optional[List[int]], float]:
        results = self.model.predict(img_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)
        if not results:
            return None, 0.0

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0 or r0.boxes.conf is None:
            return None, 0.0

        confs = r0.boxes.conf.detach().cpu().numpy()
        xyxy = r0.boxes.xyxy.detach().cpu().numpy()

        idx = int(np.argmax(confs))
        best_conf = float(confs[idx])
        x1, y1, x2, y2 = xyxy[idx].tolist()

        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = self._clamp_bbox(int(x1), int(y1), int(x2), int(y2), w, h)
        return [x1, y1, x2, y2], best_conf
