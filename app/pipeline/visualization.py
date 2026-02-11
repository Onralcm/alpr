import base64
from typing import List, Optional
import cv2
import numpy as np

def clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def crop_with_padding(img_bgr: np.ndarray, bbox: List[int],
                      pad_l_ratio: float, pad_r_ratio: float, pad_y_ratio: float) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = img_bgr.shape[:2]
    bw, bh = (x2 - x1), (y2 - y1)

    pad_l = int(pad_l_ratio * bw)
    pad_r = int(pad_r_ratio * bw)
    pad_y = int(pad_y_ratio * bh)

    x1m, y1m, x2m, y2m = clamp_bbox(x1 - pad_l, y1 - pad_y, x2 + pad_r, y2 + pad_y, w, h)
    return img_bgr[y1m:y2m, x1m:x2m].copy()

def draw_bbox(img_bgr: np.ndarray, bbox: Optional[List[int]], label: str = "") -> np.ndarray:
    out = img_bgr.copy()
    if bbox is None:
        return out
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if label:
        cv2.putText(out, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out

def to_b64(img_bgr: np.ndarray, fmt: str = "jpg", jpg_quality: int = 85) -> str:
    fmt = fmt.lower()
    if fmt == "png":
        ok, buf = cv2.imencode(".png", img_bgr)
    else:
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")

