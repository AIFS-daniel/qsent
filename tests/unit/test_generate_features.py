# Written pre-implementation. Tests are expected to fail with ImportError
# until code-builder creates scripts/generate_features.py.

import sys
import os

# scripts/ is not a package and not on the default pytest path, so we insert
# the project root so that `from scripts.generate_features import ...` resolves.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.generate_features import humanize_name, extract_test_names  # noqa: E402


def test_given_snake_case_test_name_when_humanize_name_called_then_returns_capitalized_sentence():
    assert humanize_name("test_analyze_valid_ticker") == "Analyze valid ticker"


def test_given_file_with_mixed_functions_when_extract_test_names_called_then_returns_only_test_prefixed_names(tmp_path):
    # Includes a top-level test function, a top-level helper, and a method inside
    # a Test* class to verify all three cases are handled correctly.
    test_file = tmp_path / "sample_test.py"
    test_file.write_text(
        "def test_alpha():\n"
        "    pass\n"
        "\n"
        "def helper_setup():\n"
        "    pass\n"
        "\n"
        "class TestBeta:\n"
        "    def test_beta_method(self):\n"
        "        pass\n"
        "\n"
        "    def not_a_test(self):\n"
        "        pass\n"
    )

    names = extract_test_names(test_file)

    assert sorted(names) == ["test_alpha", "test_beta_method"]
