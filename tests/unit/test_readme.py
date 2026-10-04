"""The README's Python examples run as written (offline, with the fake provider)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[2] / "README.md"


def test_readme_examples_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), re.S)
    assert len(blocks) >= 2
    monkeypatch.chdir(tmp_path)
    scope: dict[str, object] = {}
    for block in blocks:
        exec(compile(block, str(README), "exec"), scope)  # noqa: S102 - our own documentation
    assert (tmp_path / ".meldr/specs/erp_db/Organization/v0001.yaml").exists()
