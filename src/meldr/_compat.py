"""Small compatibility helpers for supported Python versions."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import timezone
from enum import Enum
from typing import Any, BinaryIO


class StrEnum(str, Enum):
    """String enum for Python versions before 3.11."""

    def __str__(self) -> str:
        return str(self.value)


UTC = timezone.utc


def file_digest(
    fileobj: BinaryIO, digest: str | Callable[[], Any], *, buffer_size: int = 1024 * 1024
) -> Any:
    """Hash a file in blocks and return the digest object."""
    hasher = hashlib.new(digest) if isinstance(digest, str) else digest()
    while chunk := fileobj.read(buffer_size):
        hasher.update(chunk)
    return hasher
