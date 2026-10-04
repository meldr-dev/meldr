"""Deterministic record IDs: UUIDv5 of the JSON-serialized key values in a table namespace.

This is the scheme that datagrid's `generateRecordId` uses (values JSON-encoded, joined with
U+0001, nulls as empty strings). If you pass a datagrid table stableId as the namespace, string
keys get the same IDs. Floats and dates can serialize differently in Python and JavaScript.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Sequence
from typing import Any

import polars as pl

ROOT_NAMESPACE = uuid.UUID("0b6f4d1c-5a9e-4f55-9d57-6c1a0e8f2b31")
_DELIMITER = "\x01"


def namespace_for(source_id: str, entity: str) -> uuid.UUID:
    """Default namespace: one for each (source, entity), so keys do not collide across sources."""
    return uuid.uuid5(ROOT_NAMESPACE, f"{source_id}/{entity}")


_encode = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"), default=str).encode


def _uuid5(namespace: bytes, name: str) -> str:
    """`str(uuid.uuid5(...))` without building UUID objects."""
    digest = bytearray(hashlib.sha1(namespace + name.encode()).digest()[:16])  # noqa: S324
    digest[6] = (digest[6] & 0x0F) | 0x50  # version 5
    digest[8] = (digest[8] & 0x3F) | 0x80  # RFC 4122 variant
    h = digest.hex()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def _name(key_values: Sequence[Any]) -> str:
    return _DELIMITER.join("" if v is None else _encode(v) for v in key_values)


def record_id(key_values: Sequence[Any], namespace: uuid.UUID) -> str:
    """ID for one record from its primary-key values."""
    return _uuid5(namespace.bytes, _name(key_values))


def key_id_expr(key_columns: Sequence[str], namespace: uuid.UUID) -> pl.Expr:
    """IDs from the key columns, null where a key value is missing or empty (see `content_id`)."""
    keys = list(key_columns)
    if not keys:
        return pl.lit(None, dtype=pl.String)
    ns = namespace.bytes

    def ids(batch: pl.Series) -> pl.Series:
        rows = batch.struct.unnest().iter_rows()
        ids = [None if None in k or "" in k else _uuid5(ns, _name(k)) for k in rows]
        return pl.Series(ids, dtype=pl.String)

    return pl.struct(keys).map_batches(ids, return_dtype=pl.String)


def content_id(source_row: str, namespace: uuid.UUID) -> str:
    """ID of a row without a complete key.

    The ID is stable only while the source values of the row do not change. Because of this, an
    update to a keyless row becomes a new record. Map the primary key for true upserts.
    """
    return _uuid5(namespace.bytes, f"row{_DELIMITER}{source_row}")
