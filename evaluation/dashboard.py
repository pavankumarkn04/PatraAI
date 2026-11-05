"""PatraAI evaluation dashboard (developer/academic use; not part of the website).

Run with "Evaluation Dashboard.bat" or:
    py -3.11 -m streamlit run evaluation/dashboard.py --server.port 8502
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from evaluation import benchmark, scoring  # noqa: E402
from ui_theme import apply_theme, stat_cards  # noqa: E402

FAILURE_CATEGORIES = (
    "Incorrect extracted value", "Missing extracted field", "Incorrect answer", "Answer not found in document (model abstained)",
    "Unsupported or hallucinated information", "Wrong or missing page reference", "OCR or document-reading failure", "Processing error",
)

st.set_page_config(page_title="PatraAI | Evaluation dashboard", page_icon="📊", layout="wide")
apply_theme()
st.markdown('<div class="hero" style="padding:20px 0 10px"><div class="eyebrow">CS3235 · DEVELOPER EVALUATION</div><h1>PatraAI evaluation dashboard</h1>'
            '<p>Field-extraction accuracy, question-answering accuracy and page-reference accuracy of the PatraAI pipeline on a curated set of '
            'manually verified test cases. This dashboard is separate from the PatraAI website.</p></div>', unsafe_allow_html=True)


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def ratio(correct: int, total: int) -> str:
    return f"{correct} of {total}" if total else "nothing evaluated yet"


def run_with_progress(label: str, **kwargs: object) -> None:
    with st.status(label, expanded=True) as status:
        log = st.empty()
        lines: list[str] = []

        def progress(message: str) -> None:
            lines.append(message)
            log.code("\n".join(lines[-14:]))

        try:
            benchmark.run(progress=progress, **kwargs)
            status.update(label="Evaluation finished", state="complete")
        except Exception as exc:  # keep the dashboard usable; completed cases are saved
            status.update(label=f"Evaluation stopped: {exc}", state="error")
            return
    st.rerun()


cases = benchmark.load_cases()
report = benchmark.load_results()
metrics = report.get("metrics") or {}
rows = report.get("cases") or []
current_ids = {case["id"] for case in cases}
evaluated = [row for row in rows if row["case"]["id"] in current_ids]
pending = len(cases) - len(evaluated)
failed = [row for row in evaluated if row["error"] or not (row["fields_correct"] and row["qa_correct"])]
checks = [check for row in evaluated for check in row["checks"]]
page_checks = [check for check in checks if check["kind"] == "page"]

# ---------------------------------------------------------------- headline
if not evaluated:
    st.info("No evaluation results yet. Run the evaluation below; accuracy figures appear only after test cases have actually been evaluated.")
stat_cards([
    ("Total test cases", str(len(cases)), f"{len({case['document'] for case in cases})} independent documents", ""),
    ("Field-extraction accuracy", percent(metrics.get("field_accuracy")) if evaluated else "—", ratio(metrics.get("field_correct", 0), metrics.get("field_checks", 0)), ""),
    ("Question-answering accuracy", percent(metrics.get("qa_accuracy")) if evaluated else "—", ratio(metrics.get("qa_correct", 0), metrics.get("qa_checks", 0)), ""),
    ("Page-reference accuracy", percent(metrics.get("page_reference_accuracy")) if evaluated else "—", ratio(sum(check["correct"] for check in page_checks), len(page_checks)), ""),
])
models = sorted({model for row in evaluated for model in row.get("models", [])})
st.markdown(
    f'<div class="results-sub">Evaluated <b>{len(evaluated)}</b> · Pending <b>{pending}</b> · Failed <b>{len(failed)}</b> · '
    f'Processing errors <b>{sum(1 for row in evaluated if row["error"])}</b> · Last evaluation <b>{html.escape(str(report.get("generated_at", "never")).replace("T", " "))}</b> · '
    f'Model <b>{html.escape(", ".join(models) or (report.get("pipeline") or {}).get("model", "—"))}</b></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- run controls
st.markdown("#### Run the evaluation")
first, second, third = st.columns(3)
if first.button(f"Run pending cases ({pending})", type="primary", disabled=not pending, width="stretch"):
    run_with_progress(f"Evaluating {pending} pending case(s)...")
if second.button(f"Re-run failed cases ({len(failed)})", disabled=not failed, width="stretch"):
    run_with_progress(f"Re-running {len(failed)} failed case(s)...", only=[row["case"]["id"] for row in failed])
if third.button("Re-run the complete dataset", width="stretch"):
    run_with_progress(f"Evaluating all {len(cases)} case(s)...", rerun=True)
st.caption("Each case runs through the real PatraAI pipeline. Results are saved after every case and kept after restarts; a case is only re-run when it, the model configuration, or your request changes. Running the full dataset on a CPU takes one to two hours.")

if evaluated:
    # ------------------------------------------------------------ breakdowns
    st.markdown("#### Results by group")
    tabs = st.tabs(["By language", "By document category", "By document format", "By model"])
    for tab, (key, label) in zip(tabs, (("by_language", "Language"), ("by_document_type", "Category"), ("by_format", "Format"), ("by_model", "Model"))):
        groups = metrics.get(key) or {}
        frame = pd.DataFrame([{label: name, "Test cases": values["cases"],
                               "Field-extraction accuracy (%)": None if values["field_accuracy"] is None else round(values["field_accuracy"] * 100, 1),
                               "QA accuracy (%)": None if values["qa_accuracy"] is None else round(values["qa_accuracy"] * 100, 1)} for name, values in groups.items()])
        with tab:
            if frame.empty:
                st.caption("No data.")
                continue
            table, chart = st.columns([3, 2])
            table.dataframe(frame, hide_index=True, width="stretch")
            chart.bar_chart(frame.set_index(label)[["Field-extraction accuracy (%)", "QA accuracy (%)"]], height=260, stack=False)
            if key == "by_model" and len(groups) == 1:
                st.caption("Only one model configuration has been evaluated.")

    # ------------------------------------------------------------ failure analysis
    st.markdown("#### Failure analysis")
    counts = metrics.get("failure_categories") or {}
    failure_frame = pd.DataFrame([{"Failure category": name, "Incorrect checks": counts.get(name, 0)} for name in FAILURE_CATEGORIES]
                                 + [{"Failure category": name, "Incorrect checks": count} for name, count in counts.items() if name not in FAILURE_CATEGORIES])
    left, right = st.columns([3, 2])
    left.dataframe(failure_frame, hide_index=True, width="stretch")
    right.bar_chart(failure_frame.set_index("Failure category"), height=280, horizontal=True)
    st.caption("Translation error: translations are not part of this benchmark (the project metrics are field-extraction and QA accuracy), so no translation errors are counted.")

    table = pd.DataFrame(report.get("checks") or [])
    if not table.empty:
        table = table[table["case_id"].isin(current_ids)]
        only_incorrect = st.toggle("Show only incorrect predictions", value=True)
        shown = table[table["result"] == "incorrect"] if only_incorrect else table
        st.dataframe(shown[["case_id", "language", "document_type", "check", "expected", "predicted", "result", "error_category", "source_pages"]],
                     hide_index=True, width="stretch",
                     column_config={"case_id": "Test case", "language": "Language", "document_type": "Category", "check": "Check",
                                    "expected": "Expected (verified)", "predicted": "Predicted", "result": "Result",
                                    "error_category": "Failure category", "source_pages": "Correct page(s)"})

    # ------------------------------------------------------------ inspect a failure
    if failed:
        st.markdown("#### Inspect a failed case")
        selected = st.selectbox("Failed test case", [row["case"]["id"] for row in failed])
        row = next(item for item in failed if item["case"]["id"] == selected)
        case = row["case"]
        info, preview = st.columns([3, 2])
        with info:
            st.markdown(f"**{html.escape(case['document'])}** · {html.escape(case['document_type'])} · {html.escape(case['language'])} · {html.escape(case['format'])}")
            st.caption("Source: " + case.get("source_reference", ""))
            if row["error"]:
                st.error(row["error"])
            st.markdown(f"**Question:** {case['question']}  \n**Verified answer:** {scoring.display(case['expected_answer'])} (page {', '.join(map(str, case['answer_pages']))})  \n"
                        f"**PatraAI answer:** {row.get('predicted_answer') or '—'}  \n**PatraAI page(s):** {', '.join(map(str, row.get('predicted_pages') or [])) or '—'}")
            st.dataframe(pd.DataFrame([{"Check": check["check"], "Expected": scoring.display(check["expected"]), "Predicted": scoring.display(check["predicted"]),
                                        "Result": "correct" if check["correct"] else "incorrect", "Failure category": check["error_category"],
                                        "Correct page(s)": ", ".join(map(str, check["pages"]))} for check in row["checks"]]), hide_index=True, width="stretch")
        with preview:
            path = benchmark.DATASET / "documents" / case["document"]
            page = (case["answer_pages"] or [1])[0]
            try:
                if path.suffix == ".pdf":
                    from document_processing import render_pdf_page

                    st.image(render_pdf_page(path.read_bytes(), page, scale=1.0), caption=f"Original document, page {page}", width="stretch")
                elif path.suffix in {".jpg", ".jpeg", ".png", ".webp"}:
                    st.image(str(path), caption="Original image", width="stretch")
                else:
                    from document_processing import extract_docx

                    section = extract_docx(path.read_bytes())["page_texts"][page - 1]
                    st.markdown(f'<div class="summary-text"><h4>Section {page}</h4><p>{html.escape(section[:1500])}</p></div>', unsafe_allow_html=True)
            except Exception as exc:
                st.caption(f"Preview unavailable: {exc}")

    # ------------------------------------------------------------ export
    st.markdown("#### Export")
    export = st.columns(3)
    for column, (label, name, mime) in zip(export, (("Report (Markdown)", benchmark.REPORT_FILE, "text/markdown"),
                                                    ("Results (CSV)", benchmark.RESULTS_CSV, "text/csv"),
                                                    ("Results (JSON)", benchmark.RESULTS_JSON, "application/json"))):
        path = benchmark.RESULTS / name
        if path.is_file():
            column.download_button(label, data=path.read_bytes(), file_name=name, mime=mime, on_click="ignore", width="stretch")

# ---------------------------------------------------------------- dataset and scoring
st.markdown("#### Dataset")
st.dataframe(pd.DataFrame([{"Test case": case["id"], "Document": case["document"], "Category": case["document_type"], "Language": case["language"],
                            "Format": case["format"], "Source": case["source"], "Question": case["question"]} for case in cases]),
             hide_index=True, width="stretch", height=260)
with st.expander("How accuracy is calculated"):
    st.markdown(
        "- **Field-extraction accuracy** = correct fields ÷ verified fields. A field is correct when its normalized value equals the verified value "
        "(case, spacing and punctuation ignored); when both contain the same dates or rupee amounts in different formats; when the prediction contains "
        "the verified value as whole words with exactly the same numbers; or, for list fields, when every verified item appears. A missing field is incorrect.\n"
        "- **QA accuracy** = correct answers ÷ verified answers. Short answers use the same normalized matching (the verified answer, or all its dates and "
        "amounts, must appear). Longer verified answers (five or more content words) also accept a paraphrase whose content-word overlap has F1 ≥ 0.7 and "
        "recall ≥ 0.75, with every number identical. Abstentions are incorrect.\n"
        "- **Page-reference accuracy** = answers whose cited page (or Word section) is one of the verified pages ÷ answers.\n"
        "- Scores come only from comparing predictions with manually verified values; model confidence is never used.")
with st.expander("Import additional verified test cases (JSONL)"):
    st.write("One JSON object per line with: id, document (a file in evaluation/dataset/documents), document_type, language, format, source, "
             "source_reference, question, expected_answer, answer_pages, expected_fields, field_pages, verification. Upload any new documents too.")
    jsonl = st.file_uploader("Cases (.jsonl)", type=["jsonl"])
    uploads = st.file_uploader("New documents", type=["pdf", "docx", "jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    confirmed = st.checkbox("I verified every expected value against the original document.")
    if st.button("Import", disabled=not (jsonl and confirmed)):
        for upload in uploads or []:
            (benchmark.DATASET / "documents" / Path(upload.name).name).write_bytes(upload.getvalue())
        added, problems = benchmark.import_cases(jsonl.getvalue())
        if added:
            st.success(f"Imported {added} case(s). They are pending until you run the evaluation.")
        for line, problem in problems:
            st.error(f"Line {line}: {problem}")
