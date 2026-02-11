from typing import Tuple, List
import numpy as np
import cv2

class PlateOCREngine:
    def __init__(self, ocr, target_h: int = 96, border=(10,10,16,16), max_w: int = 520):
        self.ocr = ocr
        self.target_h = target_h
        self.border_t, self.border_b, self.border_l, self.border_r = border
        self.max_w = max_w

    def prep(self, crop_bgr: np.ndarray) -> np.ndarray:
        h, w = crop_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return crop_bgr

        # Resize to stable height
        scale = self.target_h / h
        new_w = max(1, int(w * scale))
        resized = cv2.resize(crop_bgr, (new_w, self.target_h), interpolation=cv2.INTER_CUBIC)

        # Optional width cap for speed
        if resized.shape[1] > self.max_w:
            resized = cv2.resize(resized, (self.max_w, resized.shape[0]), interpolation=cv2.INTER_AREA)

        # Add white border to avoid edge truncation
        bordered = cv2.copyMakeBorder(
            resized,
            self.border_t, self.border_b, self.border_l, self.border_r,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )
        return bordered

    def run(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        crop_bgr = self.prep(crop_bgr)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        out = self.ocr.ocr(crop_rgb)

        texts: List[str] = []
        confs: List[float] = []

        def add(text, conf):
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
                if isinstance(conf, (int, float)):
                    confs.append(float(conf))

        def parse(item):
            if item is None:
                return

            # (text, conf)
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str):
                add(item[0], item[1])
                return

            # (box, (text, conf)) or [box, text, conf]
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                second = item[1]
                if isinstance(second, (list, tuple)) and len(second) == 2 and isinstance(second[0], str):
                    add(second[0], second[1])
                    return
                if len(item) >= 3 and isinstance(item[1], str):
                    add(item[1], item[2])
                    return

            # nested lists
            if isinstance(item, list):
                for sub in item:
                    parse(sub)

        parse(out)

        if not texts:
            return "", 0.0

        text = " ".join(texts).strip()
        conf = float(np.mean(confs)) if confs else 0.0
        return text, conf
