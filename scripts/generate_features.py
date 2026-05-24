import ast
from pathlib import Path


def extract_test_names(file_path: Path) -> list[str]:
    tree = ast.parse(file_path.read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            names.append(node.name)
    return names


def humanize_name(test_name: str) -> str:
    without_prefix = test_name.removeprefix("test_")
    sentence = without_prefix.replace("_", " ")
    return sentence.capitalize()
