"""Small helpers for local paths and fsspec URIs."""

from __future__ import annotations

import hashlib
import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from meldr.core.errors import MissingExtraError

_PROTOCOL_EXTRA = {
    "s3": "s3",
    "s3a": "s3",
    "gs": "gcs",
    "gcs": "gcs",
    "az": "azure",
    "abfs": "azure",
    "abfss": "azure",
    "sftp": "sftp",
}


def _is_local_uri(value: str) -> bool:
    return urlparse(value).scheme in ("", "file")


def _extra_for_uri(uri: str) -> str:
    return _PROTOCOL_EXTRA.get(urlparse(uri).scheme, "fsspec")


@dataclass(frozen=True)
class StoragePath:
    """A local path or a path used by an fsspec filesystem."""

    uri: str
    path: str | Path
    fs: Any = None

    @property
    def is_local(self) -> bool:
        return self.fs is None

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def suffix(self) -> str:
        return Path(self.name).suffix

    @property
    def parent(self) -> StoragePath:
        if self.is_local:
            local = self._local
            return StoragePath(str(local.parent), local.parent)
        parent = posixpath.dirname(str(self.path))
        return StoragePath(self.fs.unstrip_protocol(parent), parent, self.fs)

    @property
    def _local(self) -> Path:
        if not self.is_local:
            raise ValueError("A remote storage path has no local path.")
        return Path(self.path)

    def child(self, name: str) -> StoragePath:
        if self.is_local:
            child = self._local / name
            return StoragePath(str(child), child)
        child_path = posixpath.join(str(self.path), name)
        return StoragePath(self.fs.unstrip_protocol(child_path), child_path, self.fs)

    def is_file(self) -> bool:
        return self._local.is_file() if self.is_local else bool(self.fs.isfile(self.path))

    def is_dir(self) -> bool:
        return self._local.is_dir() if self.is_local else bool(self.fs.isdir(self.path))

    def glob(self, pattern: str) -> list[StoragePath]:
        if self.is_local:
            return [
                StoragePath(str(path), path) for path in self._local.glob(pattern) if path.is_file()
            ]
        prefix = posixpath.join(str(self.path), pattern)
        return [
            StoragePath(self.fs.unstrip_protocol(path), path, self.fs)
            for path in sorted(self.fs.glob(prefix))
            if self.fs.isfile(path)
        ]

    def find(self) -> list[StoragePath]:
        if self.is_local:
            return [
                StoragePath(str(path), path) for path in self._local.rglob("*") if path.is_file()
            ]
        return [
            StoragePath(self.fs.unstrip_protocol(path), path, self.fs)
            for path in sorted(self.fs.find(self.path))
            if self.fs.isfile(path)
        ]

    def mkdir(self) -> None:
        if self.is_local:
            self._local.mkdir(parents=True, exist_ok=True)
        else:
            self.fs.makedirs(self.path, exist_ok=True)

    def open(self, mode: str = "rb", **kwargs: Any) -> Any:
        if self.is_local:
            return self._local.open(mode, **kwargs)
        return self.fs.open(self.path, mode=mode, **kwargs)

    def read_text(self) -> str:
        with self.open("rt", encoding="utf-8") as handle:
            return str(handle.read())

    def read_bytes(self) -> bytes:
        with self.open("rb") as handle:
            return bytes(handle.read())

    def write_text(self, value: str) -> None:
        with self.open("wt", encoding="utf-8") as handle:
            handle.write(value)

    def size(self) -> int:
        return self._local.stat().st_size if self.is_local else int(self.fs.size(self.path))

    def digest(self) -> str:
        digest = hashlib.sha256()
        with self.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def relative_to(self, base: StoragePath) -> str:
        if self.is_local and base.is_local:
            return self._local.relative_to(base._local).as_posix()
        return posixpath.relpath(str(self.path), str(base.path))


def storage_path(value: str | Path, *, feature: str = "file storage") -> StoragePath:
    """Make a path wrapper and load fsspec only for a non-local URI."""
    text = str(value)
    parsed = urlparse(text)
    if _is_local_uri(text):
        local = Path(parsed.path) if parsed.scheme == "file" else Path(text)
        return StoragePath(text, local)
    try:
        import fsspec
    except ImportError as e:
        raise MissingExtraError("fsspec", feature) from e
    try:
        fs, path = fsspec.core.url_to_fs(text)
    except ImportError as e:
        raise MissingExtraError(_extra_for_uri(text), feature) from e
    return StoragePath(text, path, fs)
