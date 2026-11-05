# PatraAI evaluation benchmark

This folder holds the curated benchmark for the CS3235 Project 3 evaluation (field-extraction accuracy, question-answering accuracy and page-reference accuracy). It contains **45 test cases over 18 independent documents** (`cases.jsonl`) and the documents themselves (`documents/`). The benchmark measures general performance; the PatraAI website accepts any supported document and never depends on this dataset.

Each test case has:

- a unique ID
- an independent source document and its source reference
- the document category, language and format
- manually verified **expected fields**, each with its source page
- a **question** and its manually verified **expected answer**
- the **page** (or, for Word files, the **section**) that contains the answer
- a note on how the values were verified

Every expected value was read from the cited page of the original document: its text layer, or the rendered page image for scanned pages. **None of them came from the model being evaluated.** For documents in Indian languages, an expected value may list verified alternatives with `{"any_of": [...]}`, for example an office name in its original script and its official English name.

## Coverage

| | Test cases |
| --- | --- |
| Languages | English 25, Hindi 7, Kannada 6, Tamil 3, Malayalam 2, English + Punjabi 2 |
| Formats | Searchable PDF 18, Scanned PDF 7, Mixed searchable + scanned 8, Scanned with OCR layer 3, Legacy-font PDF that needs OCR 3, Long 83-page PDF 3, Image (JPG) 1, Word (DOCX) 2 |
| Categories | Admission brochure, press release, vacancy circular, government circulars, official letter, scholarship notification, government scheme order, public policy document, gazette notification, notification with schedule table, government order, official proceedings, government certificate format, application forms, educational document |
| Sources | 42 cases on original public documents, 3 on clearly labelled derivatives of them (an excerpt and a JPG). No constructed or fabricated documents |
| Checks | 45 field checks, 45 answer checks, 45 page-reference checks |

**Telugu is not covered.** No public government document with Unicode Telugu text could be found from official sources during dataset creation; the Telangana and Andhra Pradesh documents found were in English. Telugu output (translation) is tested separately, but Telugu *document understanding* is not measured by this benchmark.

## Documents

| File | Language | Category | Format | Source |
| --- | --- | --- | --- | --- |
| `mh-cet-admission-brochure-2026-27.pdf` | English | Admission brochure (127 pages) | Searchable | State CET Cell, Government of Maharashtra |
| `cbse-three-language-press-release-2026.pdf` | English | Press release | Searchable | CBSE, 29-06-2026 |
| `chd-hindi-vacancy-circular-2026.pdf` | Hindi | Vacancy circular | 3 searchable + 7 scanned pages | http://chd.education.gov.in/sites/default/files/assistantdirectorlhindiadvt.pdf |
| `law-ministry-hindi-proof-reader-circular-2025.pdf` | Hindi | Official circular | Scanned with an OCR layer | https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2025/11/20251126290737471.pdf |
| `dsert-kannada-fln-circular-2026.pdf` | Kannada | Government circular | Scanned | https://dsert.karnataka.gov.in/uploads/FLN-circular_1789985175.pdf |
| `dsert-kannada-bagless-day-circular-2026.pdf` | Kannada | Government circular | Legacy (non-Unicode) font | https://dsert.karnataka.gov.in/uploads/sambramashanivara-circular26-27_1781677173.pdf |
| `tn-bc-welfare-scholarship-letter-2025.pdf` | English | Scholarship notification | Searchable | Directorate of Backward Classes Welfare, Tamil Nadu (copy at https://www.nitk.ac.in/document/attachments/8806/Govt_of_Tamilnadu_welfare_scholarship.pdf) |
| `tn-bc-welfare-scholarship-letter-page-1.jpg` | English | Scholarship notification | Image | Derived: page 1 of the letter above rendered as a JPG |
| `ap-thalliki-vandanam-scheme-go-2025.pdf` | English | Government scheme order | Searchable | G.O.MS.No.26, Government of Andhra Pradesh (copy at https://www.aputf.org/wp-content/uploads/2025/06/G.O.MS_.No_.-26-Dt.12.06.2025.pdf) |
| `tn-state-education-policy-2025-tamil.pdf` | Tamil | Public policy document (83 pages) | Mostly searchable | Tamil Nadu State Education Policy 2025, via https://tamildigitallibrary.in/ |
| `kerala-gazette-right-to-service-2013-pages-1-3.pdf` | Malayalam | Gazette notification | Scanned | Derived: pages 1–3 of https://kite.kerala.gov.in/KITE/SEVANAAVAKASHANIYAMAM-DGE.pdf (22 pages; excerpted to keep CPU OCR time reasonable) |
| `telangana-gcpe-notification-schedule-2026.pdf` | English | Notification with schedule table | Scanned | https://schooledu.telangana.gov.in/ISMS/ |
| `telangana-go-bc-welfare-sanction-2026.pdf` | English | Government order | Searchable | G.O.Rt.No.39, Government of Telangana (archive.org mirror of goir.telangana.gov.in) |
| `telangana-scert-proceedings-2026.pdf` | English | Official proceedings | Searchable | https://schooledu.telangana.gov.in/ISMS/ |
| `central-obc-ncl-certificate-format.pdf` | English | Government certificate (prescribed format) | Searchable + scanned page | https://cdnbbsr.s3waas.gov.in/s313111c20aee51aeb480ecbd988cd8cc9/uploads/2024/06/2024060566.pdf |
| `odisha-ews-income-asset-certificate-form.pdf` | English | Application form and certificate format | Searchable | https://cdnbbsr.s3waas.gov.in/s3kv051f1ec528491d7b97c00f6443761f/uploads/2025/03/2025032624.pdf |
| `punjab-income-certificate-application.pdf` | English + Punjabi | Application form | Searchable + scanned pages | https://punjab.gov.in/wp-content/uploads/2020/05/Issuance-of-Income-Certificate.pdf |
| `cs3235-mini-project-sheet.docx` | English | Educational document | Word | CS3235 mini-project sheet (course document) |

All documents are publicly published government or institutional documents, or the course document, used unmodified for academic evaluation, apart from the two labelled derivatives. Blank forms and formats contain no personal data. The government orders name only the signing officials.

## Running the evaluation

Use the dashboard (**Evaluation Dashboard.bat**), or the command line:

```powershell
py -3.11 evaluation/run_evaluation.py            # run pending cases; results are saved after each case
py -3.11 evaluation/run_evaluation.py --failed   # re-run only the failed cases
py -3.11 evaluation/run_evaluation.py --rerun    # run every case again
py -3.11 evaluation/run_evaluation.py --report   # rebuild the report from saved results
```

Results are written to `evaluation/results/`:

- `CS3235-evaluation-report.md`
- `evaluation-results.csv`
- `evaluation-results.json`

To add verified cases, import a JSONL file in the dashboard. The required keys are listed there. Imported cases are stored in `cases_imported.jsonl`. Scoring rules are documented in `evaluation/scoring.py` and on the dashboard.
