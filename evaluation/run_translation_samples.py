"""Run a factual translation smoke check for each exposed target language."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.common import load_project_env

load_project_env()

from translation import TRANSLATION_LANGUAGES, translate_text


SCRIPT_RANGES = {
    "Kannada": (0x0C80, 0x0CFF),
    "Hindi": (0x0900, 0x097F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Malayalam": (0x0D00, 0x0D7F),
}
SOURCE = "The application deadline is 15 November 2026. Reference number PAI-2026-0042."


def main() -> int:
    results = []
    for language in TRANSLATION_LANGUAGES:
        if language == "English":
            output = translate_text(SOURCE, language)
            has_script = True
        else:
            output = translate_text(SOURCE, language)
            low, high = SCRIPT_RANGES[language]
            has_script = any(low <= ord(character) <= high for character in output)
        results.append({
            "language": language,
            "backend": "configured IndicTrans2 service" if os.getenv("INDIC_TRANS2_URL", "").strip() else f"local Ollama ({os.getenv('OLLAMA_TRANSLATION_MODEL', 'gemma4:e4b')})",
            "target_script_detected": has_script,
            "reference_number_preserved": "PAI-2026-0042" in output,
            "source": SOURCE,
            "translation": output,
        })
        print(f"{language}: script={has_script}, reference_preserved={'PAI-2026-0042' in output}", flush=True)

    path = Path(__file__).resolve().parent / "translation-live.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Detailed translations: {path}", flush=True)
    return 0 if all(row["target_script_detected"] and row["reference_number_preserved"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
