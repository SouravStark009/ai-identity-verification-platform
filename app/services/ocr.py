"""OCR service using PaddleOCR (pretrained)."""
import logging
import numpy as np
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

# Loaded once at import time — model weights stay in memory across requests
_ocr_engine = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)


def extract_text(image_bgr: np.ndarray) -> dict:
    """Run OCR and return full text plus per-line results with confidence scores."""
    try:
        result = _ocr_engine.ocr(image_bgr, cls=True)
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return {"full_text": "", "lines": []}

    lines = []
    if result and result[0]:
        for line in result[0]:
            text, confidence = line[1]
            lines.append({"text": text, "confidence": round(float(confidence), 3)})

    return {"full_text": " ".join(l["text"] for l in lines), "lines": lines}