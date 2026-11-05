"""Developer tool: translate PatraAI's interface labels once with IndicTrans2.

    py -3.11 evaluation/build_label_translations.py

Writes indictrans2_labels.json next to translation.py. The website loads it
into its translation cache, so the field names and summary headings are not
translated again for every document. Run it again after changing the labels in
localization.py or the IndicTrans2 model. The labels are application text; no
document content is stored.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.common import load_project_env  # noqa: E402

load_project_env()

import translation  # noqa: E402
from localization import english_labels  # noqa: E402


def main() -> int:
    labels = list(dict.fromkeys(english_labels().values()))
    translation.load_indictrans2()
    languages = {}
    for language, (code, _name) in translation.TRANSLATION_LANGUAGES.items():
        if language == "English":
            continue
        started = time.perf_counter()
        # The same call translate_segments makes: one line per label.
        lines = translation._translate_indictrans2("\n".join(labels), code, None).split("\n")
        if len(lines) != len(labels):
            raise SystemExit(f"{language}: {len(lines)} lines for {len(labels)} labels")
        languages[language] = {label: line.strip() for label, line in zip(labels, lines)}
        print(f"{language}: {len(labels)} labels in {time.perf_counter() - started:.1f}s", flush=True)
    translation.LABEL_TRANSLATIONS_FILE.write_text(
        json.dumps({"model": translation.INDICTRANS2_MODEL, "languages": languages}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Wrote", translation.LABEL_TRANSLATIONS_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
