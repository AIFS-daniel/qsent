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


def render_features_markdown(features: OrderedDict[str, list[str]]) -> str:
    lines = [
        "# QSent Features",
        "",
        "> Auto-generated from the integration and e2e test suite. Each section reflects a product feature area; each item is a verified behavior.",
    ]
    for area, behaviors in features.items():
        lines.append("")
        lines.append(f"## {area}")
        lines.append("")
        for behavior in behaviors:
            lines.append(f"- {behavior}")
    lines.append("")
    return "\n".join(lines)


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


def check_docs(docs_path: Path, generated: str) -> bool:
    if not docs_path.exists():
        return False
    return docs_path.read_text() == generated


if __name__ == "__main__":
    import sys

    repo_root = Path(__file__).parent.parent
    features = collect_features(repo_root / "tests")
    generated = render_features_markdown(features)
    docs_path = repo_root / "docs" / "FEATURES.md"

    if "--check" in sys.argv:
        if check_docs(docs_path, generated):
            print("docs/FEATURES.md is up to date.")
            sys.exit(0)
        else:
            print("docs/FEATURES.md is missing or out of date. Run: python scripts/generate_features.py")
            sys.exit(1)
    else:
        docs_path.write_text(generated)
        print("docs/FEATURES.md written.")
        sys.exit(0)
