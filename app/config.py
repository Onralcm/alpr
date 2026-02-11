import os

YOLO_WEIGHTS_PATH = os.getenv("YOLO_WEIGHTS_PATH", "weights/license_plate.pt")

YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.50"))

OCR_LANG = os.getenv("OCR_LANG", "en")  # Turkish plates -> en is fine
OCR_USE_ANGLE = os.getenv("OCR_USE_ANGLE", "True").lower() in ("1", "true", "yes")

# Crop padding (tune later)
PAD_L_RATIO = float(os.getenv("PAD_L_RATIO", "0.10"))
PAD_R_RATIO = float(os.getenv("PAD_R_RATIO", "0.15"))
PAD_Y_RATIO = float(os.getenv("PAD_Y_RATIO", "0.10"))

# OCR preprocess
OCR_TARGET_H = int(os.getenv("OCR_TARGET_H", "96"))
OCR_BORDER_T = int(os.getenv("OCR_BORDER_T", "10"))
OCR_BORDER_B = int(os.getenv("OCR_BORDER_B", "10"))
OCR_BORDER_L = int(os.getenv("OCR_BORDER_L", "16"))
OCR_BORDER_R = int(os.getenv("OCR_BORDER_R", "16"))

# Encode (use jpg to reduce payload)
ENCODE_FORMAT = os.getenv("ENCODE_FORMAT", "jpg")  # jpg or png
JPG_QUALITY = int(os.getenv("JPG_QUALITY", "85"))
