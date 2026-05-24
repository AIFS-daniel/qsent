# Written pre-implementation. Tests are expected to fail with ImportError
# until code-builder creates scripts/generate_features.py.

import sys
import os

# scripts/ is not a package and not on the default pytest path, so we insert
# the project root so that `from scripts.generate_features import ...` resolves.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.generate_features import humanize_name  # noqa: E402


def test_given_snake_case_test_name_when_humanize_name_called_then_returns_capitalized_sentence():
    assert humanize_name("test_analyze_valid_ticker") == "Analyze valid ticker"
