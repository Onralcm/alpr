# ALPR (Automatic License Plate Recognition) --- Turkey MVP

A lightweight end-to-end **MLOps-style personal project**: - Detect
license plate with **YOLO (Ultralytics)** - Read plate text with
**PaddleOCR** - Return: - plate text (raw + postprocessed) -
bounding-box overlay image - plate crop image - timing breakdown
(detection / OCR / encoding / total) - Simple UI with **Streamlit**
(upload image → get results)

This project is designed to teach the **full development cycle**:
environment setup → inference API → UI → debugging → performance →
future CI/CD + deployment.

------------------------------------------------------------------------

## Features

-   Plate detection (YOLO weights you provide)
-   OCR (PaddleOCR)
-   Turkish plate postprocessing/validation
-   Streamlit web UI (no API URL shown)
-   Detailed timing breakdown
-   Works locally on Windows (CPU)

------------------------------------------------------------------------

## Project Structure

    ALPR/
      app/
        main.py
        config.py
        pipeline/
          detector.py
          ocr_engine.py
          postprocess.py
          viz.py
          timing.py
      web/
        app_streamlit.py
      weights/
        plate.pt
      requirements.txt
      README.md

------------------------------------------------------------------------

## Requirements

-   Windows 10/11
-   Conda (Miniconda / Anaconda)
-   Python 3.10 (recommended)
-   CPU inference (GPU optional later)

------------------------------------------------------------------------

## Installation

### 1) Create environment

``` bash
conda create -n alpr python=3.10
conda activate alpr
```

### 2) Install PyTorch CPU wheels

``` bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
```

### 3) Install remaining dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Configure Weights

Place YOLO weights here:

    weights/license_plate.pt

Optional environment override:

``` powershell
$env:YOLO_WEIGHTS_PATH="weights\your_model.pt"
```

------------------------------------------------------------------------

## Run

### Start API

``` bash
uvicorn app.main:app --reload
```

### Start UI

``` bash
streamlit run web/app_streamlit.py
```

Swagger docs:

    http://127.0.0.1:8000/docs

------------------------------------------------------------------------

## Performance Notes

### Cold Start Behavior

First request may take longer due to: - PaddleOCR initialization - Torch
kernel selection - OpenMP thread initialization - Model loading

The app performs automatic warm-up using FastAPI lifespan handlers.

------------------------------------------------------------------------

## API Response Example

``` json
{
  "plate_bbox": [x1, y1, x2, y2],
  "det_conf": 0.91,
  "plate_text_raw": "06 BK 4487",
  "plate_text_clean": "06BK4487",
  "is_valid_tr_format": true,
  "ocr_conf": 0.88,
  "timings": {
    "t_detect_ms": 300,
    "t_ocr_ms": 750,
    "t_total_ms": 1100
  }
}
```

------------------------------------------------------------------------

## Common Issues

### OpenMP Error

    libiomp5md.dll already initialized

Fix:

``` python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

### pip not recognized

Use:

    python -m pip install ...

------------------------------------------------------------------------

## Roadmap

-   Add German plate support
-   Add unit tests
-   Dockerize project
-   CI/CD pipeline
-   Deployment to cloud
-   Logging and request tracking

------------------------------------------------------------------------

## License

Educational / personal project.
