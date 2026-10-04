"""The library's own text follows ASD-STE100 (see tests/ste.py for the rules it checks)."""

from __future__ import annotations

import pytest

from tests.ste import check

MINIMUM = 0.80


@pytest.mark.parametrize("kind", ["docs", "docstring", "message", "cli", "comment"])
def test_text_follows_simplified_technical_english(kind: str) -> None:
    report = check()[kind]
    failing = [f"{n} x {rule}" for rule, n in report.violations.most_common(5)]
    assert report.rate >= MINIMUM, (
        f"{kind}: {report.rate:.0%} of {report.sentences} sentences pass ASD-STE100 checks "
        f"(minimum {MINIMUM:.0%}). Most common: {failing}. Run `uv run python -m tests.ste`."
    )
