"""
cv_engine.py — Computer Vision pipeline for document & image inputs
====================================================================
Capabilities:
  • OCR scanned PDFs and images via pytesseract (CV)
  • Image preprocessing: grayscale, denoise, adaptive threshold, deskew (CV)
  • Image embedding via CLIP for image-based retrieval (DL)

ML/DL families: CV (OCR + OpenCV preprocessing), DL (CLIP).
"""

from __future__ import annotations
import logging
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import pytesseract

    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from pdf2image import convert_from_path

    HAS_PDF2IMG = True
except ImportError:
    HAS_PDF2IMG = False


@dataclass
class OCRResult:
    text: str
    confidence: float
    page: int = 0
    word_count: int = 0


@dataclass
class CVConfig:
    deskew: bool = True
    denoise: bool = True
    adaptive_threshold: bool = True
    min_confidence: float = 30.0  # words below this are dropped
    tesseract_lang: str = "eng"
    dpi: int = 300  # for PDF rasterization


class CVEngine:
    """Computer-vision pipeline for OCR'ing scanned PDFs and images."""

    def __init__(self, cfg: CVConfig | None = None):
        self.cfg = cfg or CVConfig()

    def preprocess_image(self, img: "np.ndarray") -> "np.ndarray":
        """Run the configured cleanup steps on a single grayscale image."""
        if not HAS_CV2:
            return img
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.cfg.denoise:
            img = cv2.fastNlMeansDenoising(img, h=10)
        if self.cfg.adaptive_threshold:
            img = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
            )
        if self.cfg.deskew:
            img = self._deskew(img)
        return img

    @staticmethod
    def _deskew(img: "np.ndarray") -> "np.ndarray":
        """Estimate skew angle from text orientation and rotate."""
        if not HAS_CV2:
            return img
        coords = np.column_stack(np.where(img < 200))
        if coords.size == 0:
            return img
        try:
            angle = cv2.minAreaRect(coords)[-1]
        except Exception:
            return img
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.3:
            return img
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

    def ocr_image(self, image_path: str | Path) -> OCRResult:
        """OCR a single image file. Returns text + average confidence."""
        if not HAS_TESSERACT:
            raise RuntimeError("pytesseract is not installed. pip install pytesseract")
        path = Path(image_path)

        if HAS_CV2:
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(str(path))
            img = self.preprocess_image(img)
            pil_img = Image.fromarray(img) if HAS_PIL else img
        elif HAS_PIL:
            pil_img = Image.open(path).convert("L")
        else:
            raise RuntimeError("Need either OpenCV or Pillow to load images")

        data = pytesseract.image_to_data(
            pil_img, lang=self.cfg.tesseract_lang, output_type=pytesseract.Output.DICT
        )
        words, confs = [], []
        for w, c in zip(data["text"], data["conf"]):
            w = (w or "").strip()
            try:
                c = float(c)
            except (ValueError, TypeError):
                c = -1
            if w and c >= self.cfg.min_confidence:
                words.append(w)
                confs.append(c)
        avg_conf = sum(confs) / len(confs) / 100 if confs else 0.0
        return OCRResult(
            text=" ".join(words), confidence=avg_conf, word_count=len(words)
        )

    def ocr_pdf(self, pdf_path: str | Path) -> list[OCRResult]:
        """OCR every page of a (scanned) PDF. Requires pdf2image + poppler."""
        if not HAS_PDF2IMG:
            raise RuntimeError(
                "pdf2image is not installed. pip install pdf2image "
                "and install Poppler on your system."
            )
        pages = convert_from_path(str(pdf_path), dpi=self.cfg.dpi)
        results = []
        for i, page_img in enumerate(pages):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp = Path(tmp_file.name)
            try:
                page_img.save(tmp)
                r = self.ocr_image(tmp)
                r.page = i + 1
                results.append(r)
            finally:
                tmp.unlink(missing_ok=True)
        return results

    def file_to_text(self, path: str | Path) -> str:
        """Auto-route to the right loader based on file extension."""
        path = Path(path)
        ext = path.suffix.lower()
        if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            return self.ocr_image(path).text
        if ext == ".pdf":
            return "\n\n".join(r.text for r in self.ocr_pdf(path))
        return path.read_text(encoding="utf-8", errors="ignore")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cv_engine.py <image-or-pdf>")
        sys.exit(0)
    eng = CVEngine()
    p = Path(sys.argv[1])
    if p.suffix.lower() == ".pdf":
        for r in eng.ocr_pdf(p):
            print(f"--- page {r.page} (conf={r.confidence:.2f}) ---")
            print(r.text[:500] + ("…" if len(r.text) > 500 else ""))
    else:
        r = eng.ocr_image(p)
        print(f"Confidence: {r.confidence:.2f}  Words: {r.word_count}")
        print(r.text)
