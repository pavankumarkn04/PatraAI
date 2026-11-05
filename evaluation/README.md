# Evaluation

The CS3235 evaluation (field-extraction accuracy and QA accuracy) is a developer-only workflow. Run it with `py -3.11 evaluation/run_evaluation.py`. It uses the curated dataset in [dataset/](dataset/README.md) and writes `CS3235-evaluation-report.md`, `evaluation-results.csv` and `evaluation-results.json` to `results/`. Scoring rules are in `scoring.py`, and the runner is in `benchmark.py`. The evaluation is not part of the website.

## Local integration checks

```powershell
py -3.11 evaluation/samples/create_project_samples.py
py -3.11 evaluation/run_cases.py
py -3.11 evaluation/run_pdf_samples.py
py -3.11 evaluation/run_pdf_samples.py long-searchable-101-pages.pdf
py -3.11 evaluation/run_translation_samples.py
py -3.11 -m pip install -r requirements-dev.txt
py -3.11 -m pytest -q
```

The generated smoke inputs include an 11-page searchable PDF, a 101-page searchable PDF, a 3-page image-only English PDF, and 2-page image-only Kannada and Hindi PDFs. Pass exact filenames to `run_pdf_samples.py` to run selected cases. All are synthetic, local-only integration fixtures, not representative academic accuracy data. `run_cases.py` produces actual model predictions for 30 synthetic notice cases. Translation samples check target-script presence and preservation of a reference number; those checks do not replace fluent human review.

The scorer in `evaluate.py` reports normalized exact field match, strict QA exact match, a short-answer hit rate and per-case mismatches. Answer-hit is a lexical check, not a human semantic judgment. Generated live reports are excluded from version control.
