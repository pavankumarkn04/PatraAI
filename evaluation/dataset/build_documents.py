"""Recreate the two derived files of the evaluation dataset from their originals.

Usage:
    py -3.11 evaluation/dataset/build_documents.py <kerala_gazette.pdf> <tn_scholarship_letter.pdf>

* kerala-gazette-right-to-service-2013-pages-1-3.pdf: pages 1-3 of the 22-page
  scanned Kerala Gazette (https://kite.kerala.gov.in/KITE/SEVANAAVAKASHANIYAMAM-DGE.pdf),
  excerpted so that OCR on a CPU stays practical.
* tn-bc-welfare-scholarship-letter-page-1.jpg: page 1 of
  tn-bc-welfare-scholarship-letter-2025.pdf rendered as a JPG, to test image input.

All other documents are stored exactly as downloaded (see README.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz

DOCUMENTS = Path(__file__).resolve().parent / "documents"


def main(gazette: Path, letter: Path) -> None:
    with fitz.open(gazette) as source:
        excerpt = fitz.open()
        excerpt.insert_pdf(source, from_page=0, to_page=2)
        excerpt.save(DOCUMENTS / "kerala-gazette-right-to-service-2013-pages-1-3.pdf", deflate=True)
        excerpt.close()
    with fitz.open(letter) as source:
        source[0].get_pixmap(matrix=fitz.Matrix(2, 2)).save(DOCUMENTS / "tn-bc-welfare-scholarship-letter-page-1.jpg", jpg_quality=85)
    print("derived documents written to", DOCUMENTS)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
