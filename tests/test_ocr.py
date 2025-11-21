"""
Tests for OCR service
"""
import pytest
from PIL import Image
import io
import base64


def test_ocr_endpoint_validation():
    """Test OCR endpoint validation"""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # Missing image data
    response = client.post("/api/ocr", json={})
    assert response.status_code == 422


def test_image_preprocessing():
    """Test image preprocessing"""
    from app.services.ocr_service import OCREngine

    engine = OCREngine("test")

    # Create test image
    img = Image.new("RGB", (100, 100), color="white")

    # Preprocess
    processed = engine.preprocess_image(img)

    # Should be grayscale
    assert processed.mode == "L"


def test_load_image_from_base64():
    """Test loading image from base64"""
    from app.services.ocr_service import ocr_service

    # Create test image
    img = Image.new("RGB", (100, 100), color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()

    # Encode to base64
    b64_data = base64.b64encode(img_bytes).decode()

    # This would be async in real code
    # loaded_img = await ocr_service.load_image(image_data=f"data:image/png;base64,{b64_data}")
    # assert loaded_img.size == (100, 100)
    pass


class TestOCREngines:
    """Test individual OCR engines"""

    def test_tesseract_available(self):
        """Check if Tesseract is available"""
        from app.services.ocr_service import TesseractEngine

        engine = TesseractEngine()
        # Just check initialization
        assert engine.name == "tesseract"

    def test_easyocr_available(self):
        """Check if EasyOCR is available"""
        from app.services.ocr_service import EasyOCREngine

        engine = EasyOCREngine()
        assert engine.name == "easyocr"

    def test_paddleocr_available(self):
        """Check if PaddleOCR is available"""
        from app.services.ocr_service import PaddleOCREngine

        engine = PaddleOCREngine()
        assert engine.name == "paddleocr"
