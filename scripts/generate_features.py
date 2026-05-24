import ast
from pathlib import Path


def extract_test_names(file_path: Path) -> list[str]:
    tree = ast.parse(file_path.read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            names.append(node.name)
    return names


_FEATURE_AREA_MAP = {
    "test_analyze_endpoint": "Sentiment Analysis",
    "test_auth_endpoints": "Authentication",
    "test_news_comparison_endpoint": "News Comparison",
    "test_workflow": "Analysis Pipeline",
    "test_app_flow": "App Experience",
    "test_login_page": "Login",
}


def feature_area_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    if stem in _FEATURE_AREA_MAP:
        return _FEATURE_AREA_MAP[stem]
    without_prefix = stem.removeprefix("test_")
    return without_prefix.replace("_", " ").title()


def humanize_name(test_name: str) -> str:
    without_prefix = test_name.removeprefix("test_")
    sentence = without_prefix.replace("_", " ")
    return sentence.capitalize()
