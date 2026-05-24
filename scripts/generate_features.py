import ast
from collections import OrderedDict
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


def collect_features(tests_dir: Path) -> OrderedDict[str, list[str]]:
    result: OrderedDict[str, list[str]] = OrderedDict()
    subdirs = sorted(
        [tests_dir / "integration", tests_dir / "e2e"],
        key=lambda p: p.name,
    )
    files = sorted(
        f for subdir in subdirs for f in subdir.glob("test_*.py") if subdir.exists()
    )
    for file_path in files:
        names = extract_test_names(file_path)
        if not names:
            continue
        key = feature_area_from_filename(file_path.stem)
        result[key] = [humanize_name(n) for n in names]
    return result
