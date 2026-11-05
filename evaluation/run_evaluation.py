"""CS3235 Project 3 academic evaluation (developer workflow, not part of the website).

Runs the curated test dataset (evaluation/dataset/cases.jsonl) through the real
PatraAI pipeline, scores field extraction and question answering against the
manually verified values, and writes genuine results to evaluation/results/:

    CS3235-evaluation-report.md   academic report for submission
    evaluation-results.csv        every field, answer and page check
    evaluation-results.json       metrics, predictions and checks

Usage (from the project folder, with Ollama running):

    py -3.11 evaluation/run_evaluation.py            run pending cases, then report
    py -3.11 evaluation/run_evaluation.py --rerun    evaluate every case again
    py -3.11 evaluation/run_evaluation.py --report   rebuild the report from saved results
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.benchmark import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
