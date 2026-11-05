"""Footer sections of the PatraAI website: Privacy Policy and Terms & Conditions.

Academic model evaluation is a separate developer workflow
(evaluation/run_evaluation.py) and is not part of the website.
"""

from __future__ import annotations

import streamlit as st

from document_processing import analysis_service_description
from translation import translation_service_description


# ---------------------------------------------------------------------------
# Privacy Policy and Terms & Conditions (footer sections)
# ---------------------------------------------------------------------------

def render_privacy() -> None:
    analysis = analysis_service_description()
    translation = translation_service_description()
    if analysis["local"] and translation["local"]:
        where = "With the current setup, document analysis, OCR and translation all run <b>on the computer that runs PatraAI</b>. Your documents are not sent to an outside AI service."
    else:
        parts = [name for name, local in (("document analysis and scanned-page reading", analysis["local"]), ("translation", translation["local"])) if not local]
        where = (f"With the current setup, {' and '.join(parts)} use an <b>external service</b> chosen by whoever set up PatraAI. "
                 "Document text, and images of scanned pages, are sent to that service, and its own data-handling terms apply.")
    st.markdown(f"""
<div class="policy">
<h3>How PatraAI handles your documents</h3>
<p>{where} Text is read directly from PDF pages and Word files; only scanned pages, images, or pages whose text cannot be read are passed to an OCR or vision model. Results in other languages are produced by translating PatraAI's analysis (not the original file) with the translation service described above.</p>
<h3>What is stored, and for how long</h3>
<ul>
<li><b>Uploaded documents are not saved to disk.</b> The file is held in memory for your browser session.</li>
<li><b>Results are kept in memory temporarily</b> so follow-up questions are fast: extracted text, OCR text, summaries, answers and translations for up to the 8 most recently analyzed documents. Older entries are removed as new documents are analyzed.</li>
<li><b>Closing the browser tab does not erase this memory immediately.</b> It is cleared when PatraAI is stopped or restarted.</li>
<li><b>Model files</b> (for example IndicTrans2) are downloaded once and kept on this computer. They contain no document data.</li>
<li><b>Downloads</b> you make are saved wherever your browser saves files.</li>
</ul>
<h3>Other network requests</h3>
<p>The page loads its fonts from Google Fonts; this request does not include your documents. Streamlit usage statistics are turned off.</p>
<h3>What you should avoid uploading</h3>
<ul>
<li>Documents you do not have permission to process, or other people's personal documents without their consent.</li>
<li>Unnecessary sensitive data: full Aadhaar or PAN numbers, bank or card details, passwords, OTPs or medical records. Mask them first where you can.</li>
</ul>
<p>On a shared computer, anyone using the same running PatraAI can upload the same file and see its cached results. Stop PatraAI when you finish.</p>
</div>""", unsafe_allow_html=True)


def render_terms() -> None:
    st.markdown("""
<div class="policy">
<h3>1. Purpose</h3>
<p>PatraAI is an academic tool that helps you read Indian administrative documents such as notices, circulars, forms and brochures. It extracts key details, summarizes, answers questions about a document and translates the results. It is a reading aid, not an authority on what a document means.</p>
<h3>2. Acceptable use</h3>
<p>Use PatraAI lawfully and only with documents you are allowed to process. Do not use it to create misleading copies of official documents, or to attack or overload the computer it runs on.</p>
<h3>3. Your responsibility</h3>
<p>You are responsible for the documents you upload and for having permission to process them, including any personal information about other people they contain.</p>
<h3>4. AI-generated content may be wrong</h3>
<p>Summaries, extracted fields, answers and translations are produced by AI models and can contain mistakes, omissions or misread text, even when page references are shown. <b>Always check names, dates, deadlines, amounts, reference numbers, eligibility conditions and official instructions against the original document before you act on them.</b></p>
<h3>5. Not official or professional advice</h3>
<p>PatraAI does not replace official government notifications, the issuing authority, or legal, financial or other professional advice. If PatraAI and the original document differ, the original is correct.</p>
<h3>6. Limitations</h3>
<ul>
<li>Scanned, handwritten, low-resolution or stamped pages may be read incorrectly or only partly.</li>
<li>Recognition and translation quality varies between Indian languages and scripts, and machine translation can change meaning.</li>
<li>For long documents, the summary is built from each section's most informative passages, so it cannot mention every detail. Ask a question to look up a specific point.</li>
<li>Tables and multi-column layouts may lose their structure when converted to text.</li>
</ul>
<h3>7. Availability</h3>
<p>PatraAI is provided as-is for academic use, with no guarantee that it is available, error-free or fast. Speed depends on the computer and models; long or scanned documents can take several minutes. Features stop working if the model service they depend on is not running.</p>
</div>""", unsafe_allow_html=True)
