# PatraAI — CS3235 Project 3 evaluation report

*Indian Document Understanding Tool. Generated 2026-11-05 16:17:34 UTC from an actual evaluation run with the qwen3.5:4b model (ollama).*

## 1. Results

| Metric | Result |
| --- | --- |
| Test cases evaluated | 45 of 45 |
| **Field-extraction accuracy** | **75.6%** (34 of 45 fields) |
| **QA accuracy** | **77.8%** (35 of 45 answers) |
| Answer page-reference accuracy | 77.8% |
| Test cases with every field and the answer correct | 27 |
| Processing errors | 0 |

## 2. Dataset

45 test cases over 18 documents, with 45 field checks and one question per case. Expected values were verified against the documents and were not produced by the model under evaluation. Sources and verification are described in `evaluation/dataset/README.md`.

| Coverage | Test cases |
| --- | --- |
| Languages | English 25, Hindi 7, Kannada 6, Tamil 3, Malayalam 2, English + Punjabi 2 |
| Document types | Government circular 6, Vacancy circular 4, Scholarship notification 4, Admission brochure 3, Press release / notice 3, Official circular / letter 3, Government scheme order 3, Public policy document 3, Gazette notification 2, Notification with schedule table 2, Government order 2, Official proceedings / letter 2, Government certificate (prescribed format) 2, Application form + certificate format 2, Application form 2, Educational document 2 |
| Formats | Searchable PDF 18, Scanned PDF 7, Mixed PDF (searchable + scanned pages) 6, Scanned PDF with OCR text layer 3, Legacy-font PDF (needs OCR) 3, Long PDF (83 pages, mostly searchable) 3, Mixed PDF (searchable + scanned page) 2, Word (DOCX) 2, Image (JPG) 1 |
| Sources | real 42, derived 3 |

| Document | Language | Type | Format | Source | Cases |
| --- | --- | --- | --- | --- | --- |
| mh-cet-admission-brochure-2026-27.pdf | English | Admission brochure | Searchable PDF | real | 3 |
| cbse-three-language-press-release-2026.pdf | English | Press release / notice | Searchable PDF | real | 3 |
| chd-hindi-vacancy-circular-2026.pdf | Hindi | Vacancy circular | Mixed PDF (searchable + scanned pages) | real | 4 |
| law-ministry-hindi-proof-reader-circular-2025.pdf | Hindi | Official circular / letter | Scanned PDF with OCR text layer | real | 3 |
| dsert-kannada-fln-circular-2026.pdf | Kannada | Government circular | Scanned PDF | real | 3 |
| dsert-kannada-bagless-day-circular-2026.pdf | Kannada | Government circular | Legacy-font PDF (needs OCR) | real | 3 |
| tn-bc-welfare-scholarship-letter-2025.pdf | English | Scholarship notification | Searchable PDF | real | 3 |
| tn-bc-welfare-scholarship-letter-page-1.jpg | English | Scholarship notification | Image (JPG) | derived | 1 |
| ap-thalliki-vandanam-scheme-go-2025.pdf | English | Government scheme order | Searchable PDF | real | 3 |
| tn-state-education-policy-2025-tamil.pdf | Tamil | Public policy document | Long PDF (83 pages, mostly searchable) | real | 3 |
| kerala-gazette-right-to-service-2013-pages-1-3.pdf | Malayalam | Gazette notification | Scanned PDF | derived | 2 |
| telangana-gcpe-notification-schedule-2026.pdf | English | Notification with schedule table | Scanned PDF | real | 2 |
| telangana-go-bc-welfare-sanction-2026.pdf | English | Government order | Searchable PDF | real | 2 |
| telangana-scert-proceedings-2026.pdf | English | Official proceedings / letter | Searchable PDF | real | 2 |
| central-obc-ncl-certificate-format.pdf | English | Government certificate (prescribed format) | Mixed PDF (searchable + scanned page) | real | 2 |
| odisha-ews-income-asset-certificate-form.pdf | English | Application form + certificate format | Searchable PDF | real | 2 |
| punjab-income-certificate-application.pdf | English + Punjabi | Application form | Mixed PDF (searchable + scanned pages) | real | 2 |
| cs3235-mini-project-sheet.docx | English | Educational document | Word (DOCX) | real | 2 |

## 3. Results by document language

| Group | Cases | Field-extraction accuracy | QA accuracy |
| --- | --- | --- | --- |
| English | 25 | 76.0% | 88.0% |
| English + Punjabi | 2 | 100.0% | 100.0% |
| Hindi | 7 | 71.4% | 71.4% |
| Kannada | 6 | 100.0% | 83.3% |
| Malayalam | 2 | 50.0% | 50.0% |
| Tamil | 3 | 33.3% | 0.0% |

## 4. Results by document type

| Group | Cases | Field-extraction accuracy | QA accuracy |
| --- | --- | --- | --- |
| Admission brochure | 3 | 33.3% | 66.7% |
| Application form | 2 | 100.0% | 100.0% |
| Application form + certificate format | 2 | 50.0% | 100.0% |
| Educational document | 2 | 50.0% | 100.0% |
| Gazette notification | 2 | 50.0% | 50.0% |
| Government certificate (prescribed format) | 2 | 100.0% | 100.0% |
| Government circular | 6 | 100.0% | 83.3% |
| Government order | 2 | 100.0% | 100.0% |
| Government scheme order | 3 | 66.7% | 66.7% |
| Notification with schedule table | 2 | 100.0% | 100.0% |
| Official circular / letter | 3 | 100.0% | 100.0% |
| Official proceedings / letter | 2 | 100.0% | 100.0% |
| Press release / notice | 3 | 100.0% | 100.0% |
| Public policy document | 3 | 33.3% | 0.0% |
| Scholarship notification | 4 | 75.0% | 75.0% |
| Vacancy circular | 4 | 50.0% | 50.0% |

## 5. Results by document format

| Group | Cases | Field-extraction accuracy | QA accuracy |
| --- | --- | --- | --- |
| Image (JPG) | 1 | 100.0% | 100.0% |
| Legacy-font PDF (needs OCR) | 3 | 100.0% | 66.7% |
| Long PDF (83 pages, mostly searchable) | 3 | 33.3% | 0.0% |
| Mixed PDF (searchable + scanned page) | 2 | 100.0% | 100.0% |
| Mixed PDF (searchable + scanned pages) | 6 | 66.7% | 66.7% |
| Scanned PDF | 7 | 85.7% | 85.7% |
| Scanned PDF with OCR text layer | 3 | 100.0% | 100.0% |
| Searchable PDF | 18 | 72.2% | 83.3% |
| Word (DOCX) | 2 | 50.0% | 100.0% |

## 6. Results by document source

| Group | Cases | Field-extraction accuracy | QA accuracy |
| --- | --- | --- | --- |
| derived | 3 | 66.7% | 66.7% |
| real | 42 | 76.2% | 78.6% |

## 7. Correct and incorrect predictions

| Case | Language | Type | Format | Fields correct | Answer | Answer page |
| --- | --- | --- | --- | --- | --- | --- |
| brochure-01 | English | Admission brochure | Searchable PDF | 1/1 | incorrect | incorrect |
| brochure-02 | English | Admission brochure | Searchable PDF | 0/1 | correct | correct |
| brochure-03 | English | Admission brochure | Searchable PDF | 0/1 | correct | correct |
| cbse-01 | English | Press release / notice | Searchable PDF | 1/1 | correct | correct |
| cbse-02 | English | Press release / notice | Searchable PDF | 1/1 | correct | correct |
| cbse-03 | English | Press release / notice | Searchable PDF | 1/1 | correct | incorrect |
| chd-01 | Hindi | Vacancy circular | Mixed PDF (searchable + scanned pages) | 1/1 | correct | correct |
| chd-02 | Hindi | Vacancy circular | Mixed PDF (searchable + scanned pages) | 1/1 | incorrect | incorrect |
| chd-03 | Hindi | Vacancy circular | Mixed PDF (searchable + scanned pages) | 0/1 | correct | correct |
| chd-04 | Hindi | Vacancy circular | Mixed PDF (searchable + scanned pages) | 0/1 | incorrect | incorrect |
| law-01 | Hindi | Official circular / letter | Scanned PDF with OCR text layer | 1/1 | correct | correct |
| law-02 | Hindi | Official circular / letter | Scanned PDF with OCR text layer | 1/1 | correct | correct |
| law-03 | Hindi | Official circular / letter | Scanned PDF with OCR text layer | 1/1 | correct | correct |
| fln-01 | Kannada | Government circular | Scanned PDF | 1/1 | correct | correct |
| fln-02 | Kannada | Government circular | Scanned PDF | 1/1 | correct | correct |
| fln-03 | Kannada | Government circular | Scanned PDF | 1/1 | correct | correct |
| bagless-01 | Kannada | Government circular | Legacy-font PDF (needs OCR) | 1/1 | incorrect | correct |
| bagless-02 | Kannada | Government circular | Legacy-font PDF (needs OCR) | 1/1 | correct | correct |
| bagless-03 | Kannada | Government circular | Legacy-font PDF (needs OCR) | 1/1 | correct | correct |
| tnscholar-01 | English | Scholarship notification | Searchable PDF | 0/1 | correct | correct |
| tnscholar-02 | English | Scholarship notification | Searchable PDF | 1/1 | incorrect | incorrect |
| tnscholar-03 | English | Scholarship notification | Searchable PDF | 1/1 | correct | incorrect |
| tnscholar-img-01 | English | Scholarship notification | Image (JPG) | 1/1 | correct | correct |
| apscheme-01 | English | Government scheme order | Searchable PDF | 0/1 | correct | correct |
| apscheme-02 | English | Government scheme order | Searchable PDF | 1/1 | correct | correct |
| apscheme-03 | English | Government scheme order | Searchable PDF | 1/1 | incorrect | correct |
| tapolicy-01 | Tamil | Public policy document | Long PDF (83 pages, mostly searchable) | 0/1 | incorrect | incorrect |
| tapolicy-02 | Tamil | Public policy document | Long PDF (83 pages, mostly searchable) | 1/1 | incorrect | incorrect |
| tapolicy-03 | Tamil | Public policy document | Long PDF (83 pages, mostly searchable) | 0/1 | incorrect | incorrect |
| klgazette-01 | Malayalam | Gazette notification | Scanned PDF | 0/1 | correct | correct |
| klgazette-02 | Malayalam | Gazette notification | Scanned PDF | 1/1 | incorrect | correct |
| tgschedule-01 | English | Notification with schedule table | Scanned PDF | 1/1 | correct | correct |
| tgschedule-02 | English | Notification with schedule table | Scanned PDF | 1/1 | correct | correct |
| tgorder-01 | English | Government order | Searchable PDF | 1/1 | correct | correct |
| tgorder-02 | English | Government order | Searchable PDF | 1/1 | correct | correct |
| tgproceedings-01 | English | Official proceedings / letter | Searchable PDF | 1/1 | correct | correct |
| tgproceedings-02 | English | Official proceedings / letter | Searchable PDF | 1/1 | correct | correct |
| obc-01 | English | Government certificate (prescribed format) | Mixed PDF (searchable + scanned page) | 1/1 | correct | correct |
| obc-02 | English | Government certificate (prescribed format) | Mixed PDF (searchable + scanned page) | 1/1 | correct | correct |
| odisha-01 | English | Application form + certificate format | Searchable PDF | 1/1 | correct | correct |
| odisha-02 | English | Application form + certificate format | Searchable PDF | 0/1 | correct | incorrect |
| punjab-01 | English + Punjabi | Application form | Mixed PDF (searchable + scanned pages) | 1/1 | correct | correct |
| punjab-02 | English + Punjabi | Application form | Mixed PDF (searchable + scanned pages) | 1/1 | correct | correct |
| docx-01 | English | Educational document | Word (DOCX) | 1/1 | correct | correct |
| docx-02 | English | Educational document | Word (DOCX) | 0/1 | correct | correct |

## 8. Failure categories

| Category | Incorrect checks |
| --- | --- |
| Wrong or missing page reference | 10 |
| Incorrect answer | 8 |
| Missing extracted field | 6 |
| Incorrect extracted value | 5 |
| Answer not found in document (model abstained) | 2 |

## 9. Examples: expected versus predicted

| Case | Language | Check | Expected (verified) | Predicted | Category |
| --- | --- | --- | --- | --- | --- |
| brochure-01 | English | answer | State Common Entrance Test Cell | The document does not explicitly state who **issued** the Information Brochure. However, it states that: - The brochure is published on the website of the **Com | Incorrect answer |
| brochure-02 | English | amount | ₹ 1300/- / Rs. 1300 |  | Missing extracted field |
| brochure-03 | English | eligibility_criteria | 8 Lakh | Citizenship in India for all candidates.; Passed Bachelor Degree or equivalent with minimum fifty per cent marks (forty five per cent for reserved categories/EW | Incorrect extracted value |
| chd-02 | Hindi | answer | 3 / 03 / three / तीन | The number of **Assistant Director** positions is not mentioned in the document. | Incorrect answer |
| chd-03 | Hindi | reference_number | 2-5/2020 | CBC2152/11/0002/2526,EN32/33 | Incorrect extracted value |
| chd-04 | Hindi | required_documents | APAR / Annual Performance Appraisal / वार्षिक कार्य-निष्पादन मूल्यांकन रिपोर्ट / integrity certificate / सत्यनिष्ठा प्रमाण-पत्र |  | Missing extracted field |
| chd-04 | Hindi | answer | Sangita James / संगीता जेम्स | The vacancy circular was signed by **Sangita Jha**, who is identified in the document as an Administrative Officer (Administrative) of the Central Hindi Directo | Incorrect answer |
| bagless-01 | Kannada | answer | 09-06-2026 | The document states that the number (ಸಂಖ್ಯೆ) for this circular is E- 8146 DSERT/EVG/ ಸ೦ಭ್ರಮ ಶನಿವಾರ/2026-27, but it does not explicitly state a specific date of  | Incorrect answer |
| tnscholar-01 | English | reference_number | DBC/5/2025-D2-2 / DBC/ 5 /2025-D2-2 | Lr.No DBC/ 5 /2025-D2-2, G.O.(Ms).No.72 BC MBC & MW Department Dt:12.11.2019 | Incorrect extracted value |
| tnscholar-02 | English | answer | Rs.2.00 Lakh / 2.00 Lakh / 2 lakh | The document states: RS 30. [Page 12] | Incorrect answer |
| apscheme-01 | English | reference_number | G.O.MS.No. 26 / G.O.Ms.No.26 | G.O.Rt.No. 326, G.O.Ms.No.9, Comp. No. 2858012 | Incorrect extracted value |
| apscheme-03 | English | answer | Classes I to XII / Class I to XII / I to XII | The "THALLIKI VANDANAM" scheme covers students from **Class I to Class XII**, including **Intermediate Education**. - The assistance is provided for sending chi | Incorrect answer |
| tapolicy-01 | Tamil | document_title | தமிழ் நாடு மாநில கல்விக் கொள்கை- 2025 / Tamil Nadu State Education Policy 2025 / State Education Policy 2025 |  | Missing extracted field |
| tapolicy-01 | Tamil | answer | 58,800 / 58800 | The document does not contain information about the total number of schools in the Tamil Nadu School Education Department as of March 2024. | Answer not found in document (model abstained) |
| tapolicy-02 | Tamil | answer | 1.16 crore / 1.16 கோடி | The number of students as of March 2024 was not found in the document. | Incorrect answer |
| tapolicy-03 | Tamil | amount | 44,042 கோடி / Rs. 44,042 crore / 44,042 crore |  | Missing extracted field |
| tapolicy-03 | Tamil | answer | 13.7% | The document does not contain information about the share of total state expenditure allocated to school education. | Answer not found in document (model abstained) |
| klgazette-01 | Malayalam | reference_number | G.O.(P) No. 145/2013/G.Edn / 145/2013 |  | Missing extracted field |
| klgazette-02 | Malayalam | answer | Right to Service Act / സേവനാവകാശ | This notification is issued under **Section 2(0)** of the Kerala State Service Eligibility Rules (2012-921) [Page 1]. | Incorrect answer |
| odisha-02 | English | required_documents | Voter ID / Aadhaar / IT returns / RoR |  | Missing extracted field |
| docx-02 | English | important_instructions | 30–50 representative test cases / 30-50 representative test cases | Confirm specific checkpoint availability, licensing, language coverage, GPU requirements before assigning a model.; Dataset benchmarks are not substitutes for t | Incorrect extracted value |

## 10. Observations

- By document format, QA accuracy was highest for Mixed PDF (searchable + scanned page) (100.0%, 2 cases) and lowest for Long PDF (83 pages, mostly searchable) (0.0%, 3 cases).
- By language, QA accuracy was highest for English + Punjabi (100.0%, 2 cases) and lowest for Tamil (0.0%, 3 cases).
- By document type, QA accuracy was highest for Application form (100.0%, 2 cases) and lowest for Public policy document (0.0%, 3 cases).
- The most common failure category was "Wrong or missing page reference": 10 of 31 incorrect checks.
- Answers came from: retrieved excerpts + model (28), whole document + model (15), source-text match (2).

## 11. Limitations

- Patram was not evaluated: it needs a GPU inference endpoint that was not available. These results measure the local pipeline with the model named above.
- Telugu document understanding is not measured: no public Unicode Telugu government document was found, so the dataset has no Telugu document. Tamil (3 cases) and Malayalam (2 cases) use real documents, but their per-language figures rest on very few cases.
- Scoring is automatic and lexical: a correct answer phrased in a way the verified alternatives do not cover is counted as incorrect. Dates, amounts and numbers are compared exactly by design.
- Summaries and translations are not scored. The project metrics are field-extraction accuracy and QA accuracy.
- Each case was run once on a CPU-only laptop with temperature 0.1, so results can vary slightly between runs.

## 12. Method

Each test case's document runs through the same pipeline as the PatraAI website: text extraction, OCR of scanned or unreadable pages, field extraction, summary and retrieval-based question answering. Predicted fields and answers are compared with the verified values using the rules in `evaluation/scoring.py`: differences in case, spacing, punctuation, and in how a date or rupee amount is written are accepted, while a different number, date, amount or reference is not. An abstention counts as incorrect. Answer page references are scored separately.

Reproduce with `py -3.11 evaluation/run_evaluation.py`. Every check is listed in `evaluation-results.csv`, and `evaluation-results.json` contains all predictions.
