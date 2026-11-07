"""Every folder has a README.md that lists every file and subfolder in it (plan section 5)."""

import fnmatch
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".hf_cache",
             ".import_linter_cache"}
# generated; described in results/README.md
SKIP_PATHS = {ROOT / "results" / "runs", ROOT / "results" / "service_logs"}


def _folders():
    stack = [ROOT]
    while stack:
        folder = stack.pop()
        yield folder
        for child in folder.iterdir():
            if child.is_dir() and child.name not in SKIP_DIRS and child not in SKIP_PATHS \
                    and not child.name.endswith(".egg-info"):
                stack.append(child)


def _listed_names(readme: Path) -> list[str]:
    names = []
    for token in re.findall(r"`([^`\n]+)`", readme.read_text(encoding="utf-8")):
        token = token.rstrip("/")
        names.append(re.sub(r"<[^>]+>", "*", token))  # `<日期>_<主題>.json` -> `*_*.json`
    return names


def _is_listed(name: str, listed: list[str]) -> bool:
    return any(name == n or (("*" in n or "?" in n) and fnmatch.fnmatchcase(name, n)) for n in listed)


def test_every_folder_has_readme():
    missing = [str(f.relative_to(ROOT)) for f in _folders() if not (f / "README.md").is_file()]
    assert not missing, f"folders without README.md: {missing}"


def test_every_entry_is_listed_in_its_readme():
    problems = []
    for folder in _folders():
        readme = folder / "README.md"
        if not readme.is_file():
            continue
        listed = _listed_names(readme)
        for child in folder.iterdir():
            if child.name in SKIP_DIRS or child.name.endswith(".egg-info"):
                continue
            if not _is_listed(child.name, listed):
                problems.append(str(child.relative_to(ROOT)))
    assert not problems, f"not listed in their folder's README.md: {problems}"
