"""
chartink_ocr.py
---------------
Optical Character Recognition (OCR) and intelligent rule extraction engine
for Chartink scanner screenshots.

Supports:
- PNG, JPG, JPEG, WEBP screenshots
- System tesseract binary or pytesseract library
- Chartink filter syntax parsing (timeframes, indicators, operators, values, multipliers)
- Automatic detection of match logic (ALL vs ANY)
"""

import os
import re
import shutil
import subprocess
import tempfile
import logging
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageOps, ImageEnhance

logger = logging.getLogger(__name__)

# Operators mapping for Chartink phrasing
OPERATOR_MAP = [
    (r"\bcrossed\s+above\b|\bcrosses\s+above\b|\bcrossed\s+over\b", "crossed_above"),
    (r"\bcrossed\s+below\b|\bcrosses\s+below\b|\bcrossed\s+under\b", "crossed_below"),
    (r"\bgreater\s+than\s+or\s+equal\s+to\b|\bgreater\s+than\s+equal\b|>=|=>", ">="),
    (r"\bless\s+than\s+or\s+equal\s+to\b|\bless\s+than\s+equal\b|<=|=<", "<="),
    (r"\bgreater\s+than\b|\bhigher\s+than\b|\babove\b|>", ">"),
    (r"\bless\s+than\b|\blower\s+than\b|\bbelow\b|<", "<"),
    (r"\bequal\s+to\b|\bequals\b|==|=", "=="),
]

# Lines to ignore (UI chrome, buttons, headers)
IGNORE_PHRASES = [
    "passes all of the below",
    "passes any of the below",
    "stock passes",
    "cash segment",
    "future segment",
    "nifty 500 segment",
    "save scan",
    "run scan",
    "filter:",
    "filters:",
    "add filter",
    "delete",
    "duplicate",
    "submit",
    "chartink",
    "created by",
    "scan results",
    "backtest results",
]


def find_tesseract_binary() -> Optional[str]:
    """Finds available tesseract binary on host system (Mac, Linux, Windows)."""
    # 1. PATH lookup
    bin_path = shutil.which("tesseract")
    if bin_path:
        return bin_path

    # 2. Common OS installation paths
    candidate_paths = [
        "/opt/homebrew/bin/tesseract",       # Mac Apple Silicon Homebrew
        "/usr/local/bin/tesseract",          # Mac Intel Homebrew / Linux manual
        "/usr/bin/tesseract",                # Ubuntu / Debian apt-get
        "/usr/local/share/tesseract",
        "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
    ]
    for p in candidate_paths:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p

    return None


def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    Optimizes screenshot image for character recognition:
    - Grayscale conversion
    - Contrast auto-adjustment
    - Sharpness enhancement
    - Optional upscaling for low-res screenshots
    """
    try:
        # Convert RGBA / P to RGB
        if img.mode in ("RGBA", "LA", "P"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            bg.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
            img = bg
        else:
            img = img.convert("RGB")

        # Upscale if image is narrow / small
        w, h = img.size
        if w < 1000:
            scale = 1200 / max(w, 1)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        # Convert to grayscale
        gray = ImageOps.grayscale(img)

        # Contrast enhancement
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(1.8)

        # Sharpness enhancement
        sharpener = ImageEnhance.Sharpness(gray)
        processed = sharpener.enhance(1.5)

        return processed
    except Exception as e:
        logger.warning(f"Error preprocessing image for OCR: {e}")
        return img


def extract_text_from_image(image_input) -> Tuple[str, Optional[str]]:
    """
    Performs OCR on an image file path, bytes, or PIL Image.
    Returns (extracted_text, error_message).
    """
    try:
        if isinstance(image_input, (str, bytes, os.PathLike)):
            if isinstance(image_input, bytes):
                import io
                img = Image.open(io.BytesIO(image_input))
            else:
                img = Image.open(image_input)
        elif hasattr(image_input, "read"):
            # File-like object (e.g. Streamlit UploadedFile)
            image_input.seek(0)
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            return "", f"Unsupported image input type: {type(image_input)}"

        # Preprocess image
        processed_img = preprocess_image_for_ocr(img)

        # Method A: Try pytesseract if available
        try:
            import pytesseract
            tess_bin = find_tesseract_binary()
            if tess_bin:
                pytesseract.pytesseract.tesseract_cmd = tess_bin
            text = pytesseract.image_to_string(processed_img, config="--psm 6")
            if text.strip():
                return text, None
        except (ImportError, Exception):
            pass

        # Method B: Direct CLI subprocess invocation
        tess_bin = find_tesseract_binary()
        if not tess_bin:
            return "", "Tesseract OCR is not installed. Please install tesseract (e.g. `brew install tesseract` or `sudo apt-get install -y tesseract-ocr`)."

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            processed_img.save(tmp_path, format="PNG")

        try:
            cmd = [tess_bin, tmp_path, "stdout", "--psm", "6"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            text = proc.stdout
            if not text.strip():
                # Try fallback PSM mode 3 (fully automatic page segmentation)
                cmd_fallback = [tess_bin, tmp_path, "stdout", "--psm", "3"]
                proc_fallback = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=15)
                text = proc_fallback.stdout
            return text, None
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    except Exception as e:
        logger.exception("OCR extraction failure")
        return "", f"OCR failed: {str(e)}"


def clean_chartink_ocr_text(raw_text: str) -> Tuple[str, List[str], str]:
    """
    Cleans raw OCR text:
    - Identifies logic: 'ALL' or 'ANY'
    - Filters out UI noise / headers / buttons
    - Returns (logic, cleaned_lines, summary)
    """
    logic = "ALL"
    cleaned_lines = []

    lines = raw_text.splitlines()
    for raw_l in lines:
        line = raw_l.strip()
        if not line:
            continue

        l_lower = line.lower()

        # Check for ANY / OR logic in header
        if "passes any" in l_lower or "any of the below" in l_lower or "any (" in l_lower:
            logic = "ANY"

        # Check if line is UI chrome / header
        is_ignore = False
        for phrase in IGNORE_PHRASES:
            if phrase in l_lower:
                is_ignore = True
                break

        if is_ignore:
            continue

        # Strip square brackets e.g. [ Latest ] -> Latest, [ Daily ] -> Daily
        cleaned = re.sub(r"\[\s*([^\]]+?)\s*\]", r" \1 ", line)
        # Strip parentheses only around timeframe markers e.g. ( Latest ), ( -1 day )
        cleaned = re.sub(r"\(\s*(latest|daily|weekly|monthly|intraday|\-?\d+\s*day[s]?\s*ago)\s*\)", r" \1 ", cleaned, flags=re.IGNORECASE)
        
        # OCR typo corrections
        cleaned = re.sub(r"\bRS[!1|lI]\b", "RSI", cleaned, flags=re.I)
        cleaned = re.sub(r"\bRS[!1|lI]\s*(\d+)", r"RSI \1", cleaned, flags=re.I)
        cleaned = re.sub(r"(\d+)(greater|less|crossed|above|below)", r"\1 \2", cleaned, flags=re.I)
        cleaned = re.sub(r"\bEM[4A]\b", "EMA", cleaned, flags=re.I)
        cleaned = re.sub(r"\bSM[4A]\b", "SMA", cleaned, flags=re.I)
        cleaned = re.sub(r"\bcl[o0]se\b", "Close", cleaned, flags=re.I)
        cleaned = re.sub(r"\bv[o0]lume\b", "Volume", cleaned, flags=re.I)
        cleaned = re.sub(r"\b[0o]pen\b", "Open", cleaned, flags=re.I)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # If line contains an operator or indicator keyword, keep it
        has_kw = any(k in cleaned.lower() for k in [
            "close", "open", "high", "low", "volume", "rsi", "ema", "sma",
            "supertrend", "macd", "greater", "less", "crossed", ">", "<", "="
        ])
        if has_kw and len(cleaned) >= 3:
            cleaned_lines.append(cleaned)

    return logic, cleaned_lines


def parse_chartink_screenshot(image_input) -> Dict[str, Any]:
    """
    Complete pipeline: takes screenshot input, extracts text via OCR,
    cleans filter rows, and parses them into ScreenerClause list.
    """
    from screener_engine import parse_chartink_query, ScreenerClause

    raw_text, err = extract_text_from_image(image_input)
    if err:
        return {
            "success": False,
            "error": err,
            "raw_text": "",
            "logic": "ALL",
            "clauses": [],
            "cleaned_lines": []
        }

    logic, cleaned_lines = clean_chartink_ocr_text(raw_text)

    # Rejoin cleaned lines and parse through the parser
    joined_query = "\n".join(cleaned_lines)
    clauses = parse_chartink_query(joined_query)

    return {
        "success": True,
        "error": None,
        "raw_text": raw_text,
        "logic": logic,
        "clauses": clauses,
        "cleaned_lines": cleaned_lines
    }
