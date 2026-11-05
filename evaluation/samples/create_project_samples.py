"""Build local-only synthetic PDFs for reproducible integration checks."""

from __future__ import annotations

import io
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent


def make_long_pdf() -> None:
    pdf = fitz.open()
    for page_number in range(1, 102):
        page = pdf.new_page(width=595, height=842)
        if page_number == 1:
            content = (
                "STATE EDUCATION DEPARTMENT\nSTUDENT SUPPORT NOTICE\n"
                "Date: 15/11/2026\nReference Number: LONG-101-2026\n"
                "Eligible students must submit an application with proof of residence."
            )
        elif page_number == 101:
            content = "FINAL PAGE\nReview code: FINAL-101-PAGE-8472\nSubmit the signed original before the deadline."
        else:
            content = (
                f"Page {page_number} - Application guidance and administrative review.\n"
                "Applicants must retain a copy of the form and attach the required proof of residence.\n"
                "The department will review complete applications in the order received."
            )
        page.insert_textbox(fitz.Rect(45, 70, 550, 760), content, fontname="helv", fontsize=13, lineheight=1.45)
    pdf.save(ROOT / "long-searchable-101-pages.pdf", deflate=True)
    pdf.close()


def make_scanned_language_pdf(filename: str, pages: list[list[str]]) -> None:
    font_path = Path("C:/Windows/Fonts/Nirmala.ttc")
    if not font_path.exists():
        raise FileNotFoundError("Nirmala.ttc is required to render the local Kannada and Hindi scan fixtures.")
    font = ImageFont.truetype(str(font_path), 38, index=0)
    pdf = fitz.open()
    for lines in pages:
        image = Image.new("RGB", (1000, 1400), "white")
        draw = ImageDraw.Draw(image)
        y = 120
        for source_line in lines:
            words = source_line.split()
            wrapped: list[str] = []
            current = ""
            for word in words:
                candidate = f"{current} {word}".strip()
                if current and draw.textlength(candidate, font=font) > 830:
                    wrapped.append(current)
                    current = word
                else:
                    current = candidate
            if current:
                wrapped.append(current)
            for line in wrapped:
                draw.text((80, y), line, fill="black", font=font)
                y += 90
            y += 24
        data = io.BytesIO()
        image.save(data, format="JPEG", quality=82, optimize=True)
        page = pdf.new_page(width=500, height=700)
        page.insert_image(page.rect, stream=data.getvalue())
    pdf.save(ROOT / filename, deflate=True)
    pdf.close()


def main() -> None:
    make_long_pdf()
    make_scanned_language_pdf("kannada-scanned-2-pages.pdf", [
        ["ಸರ್ಕಾರಿ ಅಧಿಸೂಚನೆ", "ದಿನಾಂಕ: 15-11-2026", "ವಿಭಾಗ: ಶಿಕ್ಷಣ ಇಲಾಖೆ", "ಅರ್ಹತೆ: ಅರ್ಜಿದಾರರು ಕರ್ನಾಟಕದ ನಿವಾಸಿಯಾಗಿರಬೇಕು."],
        ["ಅರ್ಜಿ ಸಲ್ಲಿಸಲು ಕೊನೆಯ ದಿನಾಂಕ: 30-11-2026", "ಅಗತ್ಯ ದಾಖಲೆಗಳು: ಗುರುತಿನ ಚೀಟಿ, ಆದಾಯ ಪ್ರಮಾಣಪತ್ರ", "ಅರ್ಜಿಯನ್ನು ಶಿಕ್ಷಣ ಇಲಾಖೆಗೆ ಸಲ್ಲಿಸಿ."],
    ])
    make_scanned_language_pdf("hindi-scanned-2-pages.pdf", [
        ["सरकारी सूचना", "दिनांक: 15-11-2026", "विभाग: शिक्षा विभाग", "पात्रता: आवेदक कर्नाटक का निवासी होना चाहिए।"],
        ["आवेदन की अंतिम तिथि: 30-11-2026", "आवश्यक दस्तावेज़: पहचान पत्र, आय प्रमाण पत्र", "आवेदन शिक्षा विभाग में जमा करें।"],
    ])
    for path in sorted(ROOT.glob("*-pages.pdf")):
        print(path.name, path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
