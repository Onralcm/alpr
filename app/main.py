import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # keep OMP workaround

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from paddleocr import PaddleOCR
from contextlib import asynccontextmanager

from app.config import (
    YOLO_WEIGHTS_PATH, YOLO_IMGSZ, YOLO_CONF,
    OCR_LANG, OCR_USE_ANGLE,
    PAD_L_RATIO, PAD_R_RATIO, PAD_Y_RATIO,
    OCR_TARGET_H, OCR_BORDER_T, OCR_BORDER_B, OCR_BORDER_L, OCR_BORDER_R,
    ENCODE_FORMAT, JPG_QUALITY
)
from app.pipeline.timing import Timer
from app.pipeline.detector import PlateDetector
from app.pipeline.ocr_engine import PlateOCREngine
from app.pipeline.visualization import crop_with_padding, draw_bbox, to_b64
from app.pipeline.postprocess import try_fix_tr_plate

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup (warm-up) ---
    try:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)

        # Warm up detector
        _ = detector.best_bbox(dummy)

        # Warm up OCR (use a plate-like crop size; simple dummy still helps)
        _ = ocr_engine.run(dummy)

        print("✅ Warmup completed")
    except Exception as e:
        print("⚠️ Warmup failed:", e)

    yield

app = FastAPI(title="ALPR API", lifespan=lifespan)

# Load models once
yolo_model = YOLO(YOLO_WEIGHTS_PATH)
detector = PlateDetector(yolo_model, imgsz=YOLO_IMGSZ, conf=YOLO_CONF)

ocr = PaddleOCR(use_angle_cls=OCR_USE_ANGLE, lang=OCR_LANG)
ocr_engine = PlateOCREngine(
    ocr,
    target_h=OCR_TARGET_H,
    border=(OCR_BORDER_T, OCR_BORDER_B, OCR_BORDER_L, OCR_BORDER_R)
)

@app.post("/predict")
async def predict(country: str = "TR", file: UploadFile = File(...)) -> JSONResponse:
    tm = Timer()
    tm.mark("t0")

    content = await file.read()
    np_buf = np.frombuffer(content, dtype=np.uint8)

    tm.mark("decode0")
    img_bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    tm.mark("decode1")

    if img_bgr is None:
        return JSONResponse(status_code=400, content={"error": "Could not decode image."})

    tm.mark("det0")
    bbox, det_conf = detector.best_bbox(img_bgr)
    tm.mark("det1")

    # recognition (crop + ocr + post)
    tm.mark("rec0")

    plate_text_raw, ocr_conf = "", 0.0
    plate_text_clean, is_valid = "", False
    crop = None

    tm.mark("crop0")
    if bbox is not None:
        crop = crop_with_padding(img_bgr, bbox, PAD_L_RATIO, PAD_R_RATIO, PAD_Y_RATIO)
    tm.mark("crop1")

    tm.mark("ocr0")
    if crop is not None and crop.size > 0:
        plate_text_raw, ocr_conf = ocr_engine.run(crop)
    tm.mark("ocr1")

    tm.mark("post0")
    if country.upper() == "TR":
        plate_text_clean, is_valid = try_fix_tr_plate(plate_text_raw)
    else:
        plate_text_clean, is_valid = plate_text_raw, False
    tm.mark("post1")

    tm.mark("rec1")

    # viz + encode
    tm.mark("draw0")
    label = f"{plate_text_clean or 'PLATE'} | det={det_conf:.2f} ocr={ocr_conf:.2f}"
    vis = draw_bbox(img_bgr, bbox, label=label)
    tm.mark("draw1")

    tm.mark("enc0")
    bbox_b64 = to_b64(vis, fmt=ENCODE_FORMAT, jpg_quality=JPG_QUALITY)
    crop_b64 = to_b64(crop, fmt=ENCODE_FORMAT, jpg_quality=JPG_QUALITY) if crop is not None else ""
    tm.mark("enc1")

    timings = {
        "t_decode_ms": tm.ms("decode0", "decode1"),
        "t_detect_ms": tm.ms("det0", "det1"),
        "t_crop_ms": tm.ms("crop0", "crop1"),
        "t_ocr_ms": tm.ms("ocr0", "ocr1"),
        "t_post_ms": tm.ms("post0", "post1"),
        "t_draw_ms": tm.ms("draw0", "draw1"),
        "t_encode_ms": tm.ms("enc0", "enc1"),
        "t_recognition_ms": tm.ms("rec0", "rec1"),
        "t_total_ms": tm.total_ms(),
    }

    return JSONResponse(content={
        "plate_bbox": bbox,
        "det_conf": det_conf,
        "plate_text_raw": plate_text_raw,
        "plate_text_clean": plate_text_clean,
        "is_valid_tr_format": is_valid if country.upper() == "TR" else False,
        "ocr_conf": ocr_conf,
        "bbox_image_b64": bbox_b64,
        "plate_crop_b64": crop_b64,
        "timings": timings,
    })
