"""Layer rules from plan section 4, enforced by import-linter (contracts in pyproject.toml)."""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_import_linter_contracts_hold():
    exe = shutil.which("lint-imports", path=str(Path(sys.executable).parent))
    assert exe, "import-linter is not installed; run `uv sync`"
    proc = subprocess.run([exe, "--config", str(ROOT / "pyproject.toml")], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stdout + proc.stderr
