"""Regression test for detecting truncated Streamlit package installs."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import run_app


class LauncherTests(unittest.TestCase):
    def test_missing_streamlit_generated_module_marks_distribution_for_repair(self) -> None:
        def import_side_effect(module: str) -> object:
            if module == "streamlit.proto.RootContainer_pb2":
                raise ModuleNotFoundError("No module named 'streamlit.proto'", name="streamlit.proto")
            return object()

        with patch.object(run_app.importlib, "import_module", side_effect=import_side_effect):
            missing, broken = run_app.package_issues()

        self.assertEqual(missing, [])
        self.assertEqual(broken, ["streamlit"])


if __name__ == "__main__":
    unittest.main()
