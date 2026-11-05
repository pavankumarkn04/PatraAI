"""Shared PatraAI styling and small HTML helpers for every page."""

from __future__ import annotations

import html
import time

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { color-scheme:light; --ink:#20251f; --muted:#626960; --green:#28694f; --paper:#f6f5f0; --line:#e1e2da; }
.stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',system-ui,sans-serif; }
.block-container { max-width:1480px !important; padding:1.5rem clamp(1rem,4vw,3.5rem) 3rem !important; }
#MainMenu, footer, header { visibility:hidden; }
[data-testid="stToolbar"] { display:none; }
.mast { display:flex; align-items:center; justify-content:space-between; padding:2px 0 6px; }
.brand { display:flex; gap:11px; align-items:center; font:700 14px 'Manrope',system-ui,sans-serif; letter-spacing:-.3px; }
.brandmark { width:29px;height:29px;border-radius:9px;background:#28694f;color:#fff;display:grid;place-items:center;font-size:17px; }
.topmeta { font:11px 'DM Mono',ui-monospace,monospace;color:var(--muted);letter-spacing:.04em; }
.mast-rule { border-bottom:1px solid var(--line); margin:4px 0 0; }
.hero { padding:32px 0 26px; }
.eyebrow { font:11px 'DM Mono',ui-monospace,monospace;text-transform:uppercase;letter-spacing:.14em;color:var(--green);margin-bottom:12px; }
.hero h1 { font:700 clamp(30px,3.6vw,48px)/1.08 'Manrope',system-ui,sans-serif;letter-spacing:-1.8px;margin:0;color:var(--ink); }
.hero p { font-size:15px;color:var(--muted);margin:11px 0 0;max-width:640px;line-height:1.6; }
.layout-label { font:10px 'DM Mono',ui-monospace,monospace;letter-spacing:.12em;text-transform:uppercase;color:#8a8d85;margin:0 0 12px; }
div[data-testid="stVerticalBlockBorderWrapper"] { background:#fff;border-color:var(--line);border-radius:14px; }
.panel-title { font:700 14px 'Manrope',system-ui,sans-serif;margin:0 0 3px; }
.panel-sub { font-size:12px;color:var(--muted);line-height:1.5;margin:0 0 15px; }
.paper-preview { background:#f3f1e9;border:1px solid #e9e5d8;border-radius:10px;min-height:250px;padding:14px;display:flex;align-items:center;justify-content:center;overflow:hidden; }
.stImage { background:#f3f1e9;border:1px solid #e9e5d8;border-radius:10px;padding:12px; }
.empty-preview { text-align:center;color:#8c9189;font-size:12px;line-height:1.6; }
.empty-icon { font-size:29px;margin-bottom:5px; }
.result-head { display:flex;align-items:center;justify-content:space-between;margin-bottom:13px; }
.tag { font:10px 'DM Mono',ui-monospace,monospace;background:#edf4ef;color:var(--green);padding:5px 8px;border-radius:99px; }
.field { padding:11px 0;border-top:1px solid #f0efe9; }
.field-label { font:10px 'DM Mono',ui-monospace,monospace;color:#888d84;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px; }
.field-value { font-size:13px;color:var(--ink);line-height:1.5; }
.summary { font-size:13px;line-height:1.68;color:#444a43;background:#f7f8f5;border:1px solid #ecefe9;border-radius:9px;padding:13px;margin:9px 0 14px; }
.translation-card { font-size:13px;line-height:1.75;color:#26352b;background:linear-gradient(135deg,#eff6f0,#f7f8f5);border:1px solid #dce9dd;border-radius:11px;padding:15px;margin:9px 0 14px;white-space:pre-wrap; }
.stages { list-style:none;margin:4px 0 12px;padding:0;font-size:13px; }
.stages li { display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-top:1px solid #f0efe9; }
.stages li:first-child { border-top:none; }
.stages .label { color:var(--ink);font-weight:600; }
.stages .value { font:12px 'DM Mono',ui-monospace,monospace;color:var(--muted);text-align:right; }
.stages .done .value { color:var(--green); }
.stages .running .value { color:#8a5a00; }
.stages .failed .value { color:#a3312a; }
.stat-grid { display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:6px 0 16px; }
.stat { background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 16px; }
.stat .k { font:10px 'DM Mono',ui-monospace,monospace;text-transform:uppercase;letter-spacing:.1em;color:#888d84; }
.stat .v { font:700 26px/1.2 'Manrope',system-ui,sans-serif;color:var(--ink);margin-top:6px; }
.stat .s { font-size:12px;color:var(--muted);margin-top:2px; }
.stat.bad .v { color:#a3312a; }
.stat.good .v { color:var(--green); }
.policy { max-width:760px; }
.policy h2 { font:700 22px 'Manrope',system-ui,sans-serif;margin:26px 0 8px; }
.policy h3 { font:700 15px 'Manrope',system-ui,sans-serif;margin:22px 0 6px; }
.policy p, .policy li { font-size:14px;line-height:1.7;color:#3b4139; }
.stButton button, .stDownloadButton button, .stFormSubmitButton button { border-radius:9px;font-family:'DM Sans',system-ui,sans-serif;font-weight:600;min-height:42px; }
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] { background:#28694f !important;border-color:#28694f !important;color:#fff !important; }
.stButton button[kind="primary"]:hover { background:#20573f !important;border-color:#20573f !important; }
.stButton button:disabled, .stButton button[kind="primary"]:disabled { background:#e3e7e2 !important;border-color:#d5dbd4 !important;color:#687168 !important;opacity:1 !important;cursor:not-allowed !important; }
[data-testid="stFileUploader"] section { background:#fbfcfa !important;border:1px dashed #cbd5ca;border-radius:10px; }
[data-testid="stFileUploader"] button { background:#28694f !important;color:#fff !important;border-color:#28694f !important; }
[data-testid="stFileUploaderDropzoneInstructions"], [data-testid="stFileUploaderDropzoneInstructions"] * { color:#555e54 !important; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] [data-baseweb="select"] > div, [data-testid="stSelectbox"] [data-baseweb="select"] > div > div { background:#fff !important;color:#20251f !important;border-color:#d7dbd4 !important; }
[data-testid="stTextInput"] input::placeholder, [data-testid="stTextArea"] textarea::placeholder { color:#737a71 !important;opacity:1; }
[data-testid="stSelectbox"] [data-baseweb="select"] *, [data-testid="stSelectbox"] svg { color:#20251f !important;fill:#20251f !important; }
[data-testid="stFileUploaderFile"] { background:#fff !important;color:#20251f !important;border:1px solid #d7dbd4 !important;border-radius:9px !important; }
[data-testid="stFileUploaderFile"] * { color:#20251f !important; }
[data-testid="stFileUploader"] small, [data-testid="stCaptionContainer"] { color:#626960 !important; }
div[data-testid="stExpander"] { border:1px solid var(--line);border-radius:10px;background:white; }
[data-testid="stPageLink"] a { padding:2px 4px; }
[data-testid="stPageLink"] p { font-size:12px;color:var(--muted); }
[data-testid="stPageLink-NavLink"]:hover p { color:var(--green); }
/* Results section */
.results-head { display:flex;align-items:center;justify-content:space-between;gap:10px;margin:34px 0 4px;padding-top:20px;border-top:2px solid var(--green); }
.results-head h2 { font:800 26px/1.2 'Manrope',system-ui,sans-serif;letter-spacing:-.6px;margin:0;color:var(--ink); }
.results-sub { font-size:13px;color:#4a5047;margin:0 0 6px; }
.section-title { display:flex;align-items:center;gap:9px;font:700 17px 'Manrope',system-ui,sans-serif;color:var(--ink);margin:26px 0 10px; }
.section-title .num { display:grid;place-items:center;width:24px;height:24px;border-radius:50%;background:var(--green);color:#fff;font-size:12px; }
.overview-grid { display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px; }
.overview-item { background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 14px; }
.overview-item .k { font:10px 'DM Mono',ui-monospace,monospace;text-transform:uppercase;letter-spacing:.08em;color:#596057; }
.overview-item .v { font-size:14px;font-weight:600;color:var(--ink);margin-top:5px;line-height:1.4;word-break:break-word; }
.missing { color:#5f655c;font-style:italic;font-weight:400; }
.info-table-wrap { overflow-x:auto; }
.info-table { width:100%;border-collapse:separate;border-spacing:0;background:#fff;border:1px solid var(--line);border-radius:10px;font-size:13.5px; }
.info-table th { text-align:left;width:24%;background:#f4f6f1;color:#2f352d;font-weight:600;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top; }
.info-table td { padding:10px 12px;border-bottom:1px solid var(--line);color:var(--ink);vertical-align:top;line-height:1.5; }
.info-table tr:last-child th, .info-table tr:last-child td { border-bottom:none; }
.info-table td ul { margin:0;padding-left:18px; }
.info-table .pages { font:11px 'DM Mono',ui-monospace,monospace;color:#3f6b55;white-space:nowrap;width:1%; }
.summary-text { font-size:14.5px;line-height:1.7;color:#2b312a;background:#fff;border:1px solid var(--line);border-radius:10px;padding:16px 18px; }
.summary-text p { margin:0 0 10px; }
.summary-text ul { margin:0;padding-left:20px; }
.summary-text li { margin:4px 0; }
.answer-card { background:#fff;border:1px solid #cfe0d4;border-left:4px solid var(--green);border-radius:10px;padding:14px 16px;margin:10px 0; }
.answer-card.not-found { border-left-color:#a86a12;background:#fffaf1;border-color:#ecd9b8; }
.answer-card .q { font-weight:700;color:#2f352d;margin-bottom:6px;font-size:13.5px; }
.answer-card .a { color:var(--ink);font-size:14.5px;line-height:1.6;white-space:pre-wrap; }
.answer-card .meta { font:11px 'DM Mono',ui-monospace,monospace;color:#3f6b55;margin-top:8px; }
.answer-card.not-found .meta { color:#8a560e; }
.translation-head { font:600 13px 'DM Sans',system-ui,sans-serif;color:#1f4f3b;margin:10px 0 4px; }
.results-nav { display:flex;flex-wrap:wrap;gap:6px 8px;margin:10px 0 14px; }
.results-nav a { font:600 12.5px 'DM Sans',system-ui,sans-serif;color:#1f4f3b !important;text-decoration:none;background:#edf4ef;border:1px solid #d4e5d9;border-radius:99px;padding:5px 12px; }
.results-nav a:hover { background:#28694f;color:#fff !important; }
.section-title { scroll-margin-top:16px; }
.info-table tr.head th, .info-table tr.head td { background:#e9efe6;color:#1f4f3b;font:700 11px 'DM Mono',ui-monospace,monospace;text-transform:uppercase;letter-spacing:.06em;padding:8px 12px; }
.info-table th { font-weight:700;color:#1f2a22; }
.info-table td ul { margin:0;padding-left:18px; }
.info-table td li { margin:2px 0; }
.info-table .original { font-size:12px;color:#4f564d;margin-top:3px; }
.info-table.compact { margin-top:8px;font-size:13px; }
.info-table.compact th { width:34%; }
.summary-text h4 { font:700 13.5px 'Manrope',system-ui,sans-serif;color:#1f4f3b;margin:14px 0 4px; }
.summary-text h4:first-child { margin-top:0; }
.summary-text ul, .summary-text ol { margin:0 0 4px;padding-left:20px; }
.summary-text li { margin:3px 0; }
.translation-card h4 { font:700 13px 'Manrope',system-ui,sans-serif;color:#1f4f3b;margin:10px 0 4px; }
.translation-card h4:first-child { margin-top:0; }
.translation-card p { margin:0 0 6px; }
.translation-card ul { margin:0 0 6px;padding-left:20px; }
.translation-card { white-space:normal !important; }
.translation-card .lang { font:700 11px 'DM Mono',ui-monospace,monospace;text-transform:uppercase;letter-spacing:.08em;color:#28694f;margin-bottom:8px; }
.answer-card .a p { margin:0 0 6px; }
.answer-card .a ul { margin:2px 0 6px;padding-left:20px; }
.answer-card .a li { margin:2px 0; }
.answer-card mark { background:#fdf0c8;color:#20251f;padding:0 3px;border-radius:3px; }
.answer-card .a { white-space:normal !important; }
.answer-card .q { display:flex;gap:8px;align-items:baseline;padding-bottom:7px;border-bottom:1px dashed #dce9dd; }
.answer-card .q-label { flex:none;font:700 10.5px 'DM Mono',ui-monospace,monospace;color:#fff;background:var(--green);border-radius:5px;padding:1px 6px; }
.answer-card .a { margin-top:8px; }
.qa-hint { font-size:12.5px;color:#4a5047;margin:0 0 8px; }
/* Indian scripts: system fonts that cover them, and more line spacing for their marks. */
.answer-card .a, .summary-text, .translation-card, .info-table td, .overview-item .v { font-family:'DM Sans','Nirmala UI','Noto Sans Kannada','Noto Sans Devanagari','Noto Sans Tamil','Noto Sans Telugu','Noto Sans Malayalam',system-ui,sans-serif; }
.answer-card .a, .translation-card { line-height:1.8; }
@media(max-width:700px){ .info-table th { width:38%; } .results-head h2 { font-size:22px; } .overview-grid { grid-template-columns:1fr 1fr; } }
/* Footer, Privacy Policy and Terms & Conditions */
.site-footer { display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px 24px;border-top:1px solid #cfd3c7;margin-top:36px;padding:20px 0 14px; }
.footer-brand { display:flex;align-items:center;gap:9px;font:700 15px 'Manrope',system-ui,sans-serif;color:#20251f; }
.footer-brand .brandmark { width:24px;height:24px;font-size:13px;border-radius:7px; }
.footer-brand .sub { font-weight:500;color:#3b4139; }
.footer-disclaimer { font-size:12.5px;color:#4a5047;line-height:1.5; }
div[data-testid="stExpander"] { border:1px solid #d4d8cc !important;border-radius:10px;background:#fff; }
div[data-testid="stExpander"] summary { padding:12px 16px; }
div[data-testid="stExpander"] summary p { font-weight:600;color:#20251f !important;font-size:14px; }
div[data-testid="stExpander"] summary svg, div[data-testid="stExpander"] summary [data-testid="stIconMaterial"] { color:#28694f !important;fill:#28694f !important; }
div[data-testid="stExpander"] summary:hover p { color:#28694f !important; }
.policy { border-left:3px solid #28694f;padding:2px 4px 6px 18px;margin:4px 0; }
.policy h3 { font:700 15px 'Manrope',system-ui,sans-serif;color:#1f4f3b;margin:18px 0 6px; }
.policy h3:first-child { margin-top:4px; }
.policy p, .policy li { font-size:14px;line-height:1.7;color:#2b312a; }
.policy code { background:#f1f3ed;color:#2b312a;border-radius:4px;padding:1px 5px; }
.stTabs [data-baseweb="tab-list"] { gap:20px; }
.stTabs [data-baseweb="tab"] { font-size:13px; }
[data-testid="stAlert"] { border-radius:10px; }
[data-testid="stAlert"] p { font-size:13px; line-height:1.55; }
[data-testid="stImage"] img { max-height:440px;object-fit:contain; }
@media(max-width:850px){.block-container{padding:1.25rem 1rem 2rem !important}.hero{padding:25px 0 20px}.hero h1{letter-spacing:-1.2px}.topmeta{display:none}}
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def masthead() -> None:
    st.markdown('<div class="mast"><div class="brand"><span class="brandmark">P</span> PatraAI <span style="font-weight:400;color:#687068">| Indian document intelligence</span></div><div class="topmeta">READ | EXTRACT | UNDERSTAND</div></div>', unsafe_allow_html=True)


def stat_cards(cards: list[tuple[str, str, str, str]]) -> None:
    """Render (label, value, subtitle, tone) cards; tone is '', 'good' or 'bad'."""
    body = "".join(
        f'<div class="stat {html.escape(tone)}"><div class="k">{html.escape(label)}</div><div class="v">{html.escape(value)}</div><div class="s">{html.escape(sub)}</div></div>'
        for label, value, sub, tone in cards
    )
    st.markdown(f'<div class="stat-grid">{body}</div>', unsafe_allow_html=True)


class StageBoard:
    """A live list of the processing stages that are actually running.

    Stages are added the first time they report progress, so stages that are
    skipped or served from cache never appear.
    """

    LABELS = {
        "ocr": "Scanned-page OCR", "analysis": "Information extraction + summary", "fields": "Field extraction", "summary": "Summary generation",
        "qa": "Question answering", "translation": "Translation",
    }

    def __init__(self, placeholder: st.delta_generator.DeltaGenerator) -> None:
        self.placeholder = placeholder
        self.rows: dict[str, tuple[str, str]] = {}
        self.started: dict[str, float] = {}
        self.finished: dict[str, float] = {}

    def update(self, stage: str, done: int, total: int) -> None:
        if stage == "ocr":
            text = f"{done}/{total} pages"
        elif stage == "summary" and total > 1:
            sections = total - 1
            text = f"section {done + 1} of {sections}" if done < sections else ("combining sections" if done == sections else "done")
        elif stage == "translation":
            text = f"part {done} of {total}" if done < total else "done"
        else:
            text = "done" if done >= total else "running"
        state = "done" if done >= total else "running"
        if stage == "ocr" and done >= total:
            text += " · done"
        self.started.setdefault(stage, time.perf_counter())
        if state == "done":
            self.finished.setdefault(stage, time.perf_counter())
        self.rows[stage] = (state, text)
        self.render()

    def elapsed(self, stage: str) -> float | None:
        if stage not in self.started:
            return None
        return self.finished.get(stage, time.perf_counter()) - self.started[stage]

    def fail(self, stage: str, message: str = "failed") -> None:
        self.finished.setdefault(stage, time.perf_counter())
        self.rows[stage] = ("failed", message)
        self.render()

    def set(self, stage: str, state: str, text: str) -> None:
        """Override a stage after validating its output (state: done, running or failed)."""
        self.rows[stage] = (state, text)
        self.render()

    def render(self) -> None:
        def measured(stage: str, state: str, text: str) -> str:
            seconds = self.elapsed(stage)
            # Measured wall-clock time of the stage, shown once it has finished.
            return f"{text} · {seconds:.1f} s" if state != "running" and seconds is not None else text

        items = "".join(
            f'<li class="{state}"><span class="label">{html.escape(self.LABELS.get(stage, stage))}</span><span class="value">{html.escape(measured(stage, state, text))}</span></li>'
            for stage, (state, text) in self.rows.items()
        )
        self.placeholder.markdown(f'<ul class="stages">{items}</ul>', unsafe_allow_html=True)
