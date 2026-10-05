import os
import io
import re

import pdfplumber
import pymupdf
import pytesseract

from PIL import Image


# Tesseract location
# 1. TESSERACT_CMD environment variable, if set
# 2. Default Windows install folder, if it exists
# 3. Otherwise "tesseract" on PATH (Linux / Docker)
WINDOWS_TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

TESSERACT_CMD = os.environ.get("TESSERACT_CMD")

if not TESSERACT_CMD and os.path.exists(WINDOWS_TESSERACT_PATH):
    TESSERACT_CMD = WINDOWS_TESSERACT_PATH

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# Minimum amount of readable text before OCR is required
MIN_TEXT_LETTERS = 50


def has_enough_text(text):
    """
    Check whether the extracted page contains enough
    readable English letters to avoid OCR.
    """

    if not text:
        return False

    letters = re.findall(
        r"[A-Za-z]",
        text
    )

    return len(letters) >= MIN_TEXT_LETTERS


def ocr_page(pdf_document, page_index):
    """
    Convert one PDF page to an image and run OCR.
    """

    page = pdf_document.load_page(
        page_index
    )

    matrix = pymupdf.Matrix(
        2.5,
        2.5
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    image_bytes = pixmap.tobytes(
        "png"
    )

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    text = pytesseract.image_to_string(
        image,
        lang="eng",
        config="--psm 6"
    )

    return text


def read_pdf_pages(pdf_path):
    """
    Read the PDF page by page.

    Returns:

    [
        {
            "page": 1,
            "text": "..."
        },
        {
            "page": 2,
            "text": "..."
        }
    ]

    This preserves the page number so that ChromaDB
    can store page metadata for every chunk.
    """

    if not os.path.exists(pdf_path):

        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    pages = []

    print(
        f"Reading PDF: {pdf_path}"
    )

    # pdfplumber handles normal text extraction
    pdf = pdfplumber.open(
        pdf_path
    )

    # PyMuPDF is used when OCR is required
    pdf_document = pymupdf.open(
        pdf_path
    )

    try:

        total_pages = len(
            pdf.pages
        )

        print(
            "Total pages:",
            total_pages
        )

        for page_index, page in enumerate(
            pdf.pages
        ):

            page_number = (
                page_index + 1
            )

            # Try normal extraction first
            text = page.extract_text() or ""

            # If page contains almost no usable text,
            # perform OCR
            if not has_enough_text(text):

                print(
                    f"OCR page "
                    f"{page_number}/{total_pages}"
                )

                text = ocr_page(
                    pdf_document,
                    page_index
                )

            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    finally:

        pdf.close()
        pdf_document.close()

    return pages


def extract_pdf_text(pdf_path):
    """
    Read all pages and combine them into one string.

    This keeps compatibility with your existing code
    that expects read_pdf() to return a single string.
    """

    pages = read_pdf_pages(
        pdf_path
    )

    output = []

    for page in pages:

        output.append(
            f"""
=== PAGE {page["page"]} ===

{page["text"]}
"""
        )

    return "\n".join(
        output
    )


def read_pdf(pdf_path):
    """
    Existing interface used by the rest of the project.
    """

    return extract_pdf_text(
        pdf_path
    )


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )

    PDF_PATH = os.path.join(
        BASE_DIR,
        "data",
        "dummy_it_policy(1).pdf"
    )

    pages = read_pdf_pages(
        PDF_PATH
    )

    print("\n==============================")
    print("PDF READER TEST")
    print("==============================")

    print(
        "Pages:",
        len(pages)
    )

    if pages:

        print(
            "\nFirst page:"
        )

        print(
            pages[0]["text"][:500]
        )