"""
MedAI – ML Utility Functions
"""

import logging
import os

logger = logging.getLogger("ml_engine")

# Cache EasyOCR reader to avoid re-initializing on every request
_easyocr_reader = None


def get_gpu_info() -> dict:
    """Return GPU information for system diagnostics."""
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "gpu_memory_total_mb": round(
                    torch.cuda.get_device_properties(0).total_mem / 1024 / 1024
                ),
                "torch_version": torch.__version__,
            }
        return {
            "gpu_available": False,
            "torch_version": torch.__version__,
            "note": "Running on CPU",
        }
    except ImportError:
        return {"gpu_available": False, "note": "PyTorch not installed"}


def sanitize_input(text: str) -> str:
    """Sanitize and normalize user symptom input."""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    cleaned = text.strip()
    if len(cleaned) < 3:
        raise ValueError("Input too short. Please describe your symptoms.")
    if len(cleaned) > 5000:
        raise ValueError("Input too long. Please limit to 5000 characters.")
    return cleaned


def extract_text_from_file(uploaded_file) -> str:
    """
    Extract text from uploaded medical reports.
    Supports: PDF, TXT, PNG, JPG, JPEG images (OCR).
    """
    filename = uploaded_file.name.lower()
    content_type = getattr(uploaded_file, "content_type", "")

    # Plain text files
    if filename.endswith(".txt"):
        try:
            raw = uploaded_file.read()
            return raw.decode("utf-8", errors="ignore").strip()
        except Exception as exc:
            logger.error("Failed to read TXT file: %s", exc)
            raise ValueError("Could not read the text file.") from exc

    # PDF files
    if filename.endswith(".pdf") or "pdf" in content_type:
        # Try pdfplumber first (better table extraction)
        try:
            import pdfplumber

            pages_text = []
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    # Extract regular text
                    text = page.extract_text()
                    if text:
                        pages_text.append(text.strip())

                    # Extract tables (medical reports often have tabular data)
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if row:
                                # Filter None cells and join
                                cells = [str(c).strip() for c in row if c]
                                if cells:
                                    pages_text.append(" | ".join(cells))

            combined = "\n".join(pages_text).strip()
            if not combined:
                raise ValueError(
                    "The PDF appears to be scanned/image-based. "
                    "Please upload a text-based PDF or describe your symptoms."
                )
            logger.info("pdfplumber extracted %d chars from PDF", len(combined))
            return combined
        except ImportError:
            logger.warning("pdfplumber not available, trying PyPDF2 fallback")
        except ValueError:
            raise
        except Exception as exc:
            logger.warning("pdfplumber failed: %s — trying PyPDF2 fallback", exc)

        # Fallback: PyPDF2
        try:
            import PyPDF2

            uploaded_file.seek(0)
            reader = PyPDF2.PdfReader(uploaded_file)
            pages_text = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text.strip())
            combined = "\n".join(pages_text).strip()
            if not combined:
                raise ValueError(
                    "The PDF appears to be scanned/image-based. "
                    "Please upload a text-based PDF or describe your symptoms."
                )
            return combined
        except ImportError:
            raise ValueError(
                "PDF processing is not available. Please install pdfplumber."
            )
        except ValueError:
            raise
        except Exception as exc:
            logger.error("Failed to read PDF file: %s", exc)
            raise ValueError("Could not read the PDF file.") from exc

    # Image files (OCR)
    if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        global _easyocr_reader
        # Try EasyOCR first (self-contained, no system binary needed)
        try:
            import easyocr
            import numpy as np
            from PIL import Image

            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)

            if _easyocr_reader is None:
                logger.info("Initializing EasyOCR reader (first request)...")
                _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

            results = _easyocr_reader.readtext(img_array, detail=0)
            text = "\n".join(results).strip()

            if not text:
                raise ValueError(
                    "Could not extract text from the image. "
                    "Please try a clearer image or describe your symptoms manually."
                )
            logger.info("EasyOCR extracted %d chars from image", len(text))
            return text
        except ImportError:
            logger.warning("easyocr not available, trying pytesseract fallback")
        except ValueError:
            raise
        except Exception as exc:
            logger.warning("EasyOCR failed: %s – trying pytesseract fallback", exc)

        # Fallback: pytesseract (requires Tesseract binary installed)
        try:
            from PIL import Image
            import pytesseract

            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            text = pytesseract.image_to_string(image).strip()
            if not text:
                raise ValueError(
                    "Could not extract text from the image. "
                    "Please try a clearer image or describe your symptoms manually."
                )
            logger.info("Pytesseract extracted %d chars from image", len(text))
            return text
        except ImportError:
            raise ValueError(
                "Image OCR libraries are not available. "
                "Please describe your symptoms in text instead."
            )
        except (ValueError, KeyboardInterrupt):
            raise
        except Exception as exc:
            logger.error("Failed OCR on image: %s", exc)
            raise ValueError(
                "Could not process the image. "
                "Please try a clearer image or describe your symptoms in text."
            ) from exc

    raise ValueError(
        f"Unsupported file type: {os.path.splitext(filename)[1]}. "
        "Supported formats: PDF, TXT, PNG, JPG, JPEG."
    )