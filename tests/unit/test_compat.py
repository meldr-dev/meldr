from __future__ import annotations

import hashlib
from datetime import timezone
from pathlib import Path

from meldr._compat import UTC, StrEnum, file_digest


class _Kind(StrEnum):
    VALUE = "value"


def test_compat_exports_python_version_helpers(tmp_path: Path) -> None:
    assert UTC is timezone.utc
    assert str(_Kind.VALUE) == "value"

    path = tmp_path / "data.bin"
    path.write_bytes(b"meldr")
    with path.open("rb") as handle:
        actual = file_digest(handle, "sha256").hexdigest()
    assert actual == hashlib.sha256(b"meldr").hexdigest()
