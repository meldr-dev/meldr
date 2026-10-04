"""The RawBatch envelope that every connector gives to Meldr."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import polars as pl
import pyarrow as pa
from pydantic import BaseModel, Field

from meldr._compat import UTC, StrEnum, file_digest
from meldr.core.errors import ConnectorError


class Modality(StrEnum):
    """Kind of content in a batch."""

    TABULAR = "tabular"
    JSON = "json"
    DOCUMENT = "document"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class RawRecord(BaseModel):
    """One unit pulled by a connector: a JSON object, a file, or a media asset."""

    record_id: str
    source_id: str
    modality: Modality
    data: dict[str, Any] | None = None
    content: bytes | None = None
    uri: str | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime
    content_hash: str


_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".html": "text/html",
    ".htm": "text/html",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}


def modality_for_path(path: str | Path) -> Modality:
    """Infer the modality of a file from its extension (documents by default)."""
    kind = (mime_for_path(path) or "").split("/")[0]
    return Modality(kind) if kind in ("image", "audio", "video") else Modality.DOCUMENT


def mime_for_path(path: str | Path) -> str | None:
    """Best-effort MIME type from the extension."""
    return _MIME.get(Path(path).suffix.lower())


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    """Hex sha256 of a canonical JSON serialization."""
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode())


def json_records(
    rows: Sequence[Mapping[str, Any]], source_id: str, ids: Sequence[str]
) -> list[RawRecord]:
    """JSON objects as records with the given ids and content hashes."""
    now = datetime.now(UTC)
    return [
        RawRecord(
            record_id=rid,
            source_id=source_id,
            modality=Modality.JSON,
            data=dict(row),
            fetched_at=now,
            content_hash=sha256_json(dict(row)),
        )
        for rid, row in zip(ids, rows, strict=True)
    ]


@dataclass
class RawBatch:
    """What every connector gives to Meldr. Tabular sources set `table`, others `records`."""

    source_id: str
    modality: Modality
    table: pl.DataFrame | None = None
    records: list[RawRecord] = field(default_factory=list)
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        has_table = self.table is not None
        has_records = bool(self.records)
        if has_table == has_records:
            raise ConnectorError(
                f"RawBatch for source '{self.source_id}' must set only one of `table` and "
                "`records`. Use RawBatch.from_polars() for tables or from_dicts()/from_files() "
                "for records."
            )

    @classmethod
    def from_polars(cls, df: pl.DataFrame, source_id: str, **meta: Any) -> RawBatch:
        """Wrap a Polars DataFrame as a tabular batch."""
        return cls(source_id=source_id, modality=Modality.TABULAR, table=df, source_metadata=meta)

    @classmethod
    def from_pandas(cls, df: Any, source_id: str, **meta: Any) -> RawBatch:
        """Wrap a pandas DataFrame as a tabular batch."""
        return cls.from_polars(pl.from_pandas(df), source_id, **meta)

    @classmethod
    def from_arrow(cls, table: pa.Table, source_id: str, **meta: Any) -> RawBatch:
        """Wrap a PyArrow table as a tabular batch."""
        return cls.from_polars(cast(pl.DataFrame, pl.from_arrow(table)), source_id, **meta)

    @classmethod
    def from_dicts(cls, rows: Sequence[Mapping[str, Any]], source_id: str, **meta: Any) -> RawBatch:
        """Wrap JSON objects (for example, API records) as a JSON batch."""
        records = json_records(rows, source_id, [str(i) for i in range(len(rows))])
        return cls(
            source_id=source_id, modality=Modality.JSON, records=records, source_metadata=meta
        )

    @classmethod
    def from_contents(
        cls,
        files: Sequence[tuple[str, bytes]],
        source_id: str,
        **meta: Any,
    ) -> RawBatch:
        """Wrap named in-memory files as document or media records."""
        if not files:
            raise ConnectorError(
                f"from_contents() for source '{source_id}' got no files. "
                "Pass at least one named file."
            )
        now = datetime.now(UTC)
        records = [
            RawRecord(
                record_id=name,
                source_id=source_id,
                modality=modality_for_path(name),
                content=content,
                mime_type=mime_for_path(name),
                metadata={"filename": Path(name).name, "size": len(content)},
                fetched_at=now,
                content_hash=sha256_bytes(content),
            )
            for name, content in files
        ]
        return cls(
            source_id=source_id,
            modality=records[0].modality,
            records=records,
            source_metadata=meta,
        )

    @classmethod
    def from_files(
        cls,
        paths: Sequence[str | Path],
        source_id: str,
        *,
        root: str | Path | None = None,
        **meta: Any,
    ) -> RawBatch:
        """Wrap files (documents or media) as records.

        The extension of the first file sets the batch modality. Record ids are paths relative to
        `root` (default: file names, which must be unique).
        """
        if not paths:
            raise ConnectorError(
                f"from_files() for source '{source_id}' got no paths. Pass at least one path."
            )
        base = Path(root).resolve() if root is not None else None
        ids = [
            Path(p).resolve().relative_to(base).as_posix() if base else Path(p).name for p in paths
        ]
        if len(set(ids)) != len(ids):
            raise ConnectorError(
                f"from_files() for source '{source_id}' got files with the same name. Pass root= "
                "so that record ids are relative paths."
            )
        now = datetime.now(UTC)
        records: list[RawRecord] = []
        for p, rid in zip(paths, ids, strict=True):
            path = Path(p)
            if not path.is_file():
                raise ConnectorError(f"File not found: {path}. Check the path and try again.")
            with path.open("rb") as fh:
                digest = file_digest(fh, "sha256").hexdigest()
            size = path.stat().st_size
            records.append(
                RawRecord(
                    record_id=rid,
                    source_id=source_id,
                    modality=modality_for_path(path),
                    uri=str(path.resolve()),
                    mime_type=mime_for_path(path),
                    fetched_at=now,
                    content_hash=digest,
                    metadata={"filename": path.name, "size": size},
                )
            )
        return cls(
            source_id=source_id,
            modality=records[0].modality,
            records=records,
            source_metadata=meta,
        )
