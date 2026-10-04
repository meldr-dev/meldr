"""Caches for deterministic requests: LLM responses by request hash, embeddings by text hash."""

from __future__ import annotations

import sqlite3
import threading
from array import array
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Protocol

from meldr._sql import make_engine
from meldr._tenant import scoped_key, validate_tenant
from meldr.core.errors import ConfigError, MissingExtraError
from meldr.secrets import SecretRef

_IN_LIST = 900  # stays below SQLite's default bound-parameter limit


class ResponseCache(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> None: ...


class VectorCache(Protocol):
    def get_many(self, keys: Sequence[str]) -> dict[str, list[float]]: ...
    def set_many(self, vectors: Mapping[str, Sequence[float]]) -> None: ...


class MemoryCache:
    """Process-local dict cache."""

    def __init__(self, *, tenant: str | None = None) -> None:
        self.tenant = validate_tenant(tenant)
        self._data: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._data.get(scoped_key(self.tenant, key))

    def set(self, key: str, value: str) -> None:
        self._data[scoped_key(self.tenant, key)] = value


def _connect(path: str | Path, ddl: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(ddl)
    conn.commit()
    return conn


class SQLiteResponseCache:
    """Key/value cache in a SQLite file (default `{root}/cache/llm.sqlite`)."""

    def __init__(self, path: str | Path, *, tenant: str | None = None) -> None:
        self.tenant = validate_tenant(tenant)
        self._lock = threading.Lock()
        self._conn = _connect(
            path, "CREATE TABLE IF NOT EXISTS responses (k TEXT PRIMARY KEY, v TEXT NOT NULL)"
        )

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT v FROM responses WHERE k = ?", (scoped_key(self.tenant, key),)
            ).fetchone()
        return None if row is None else str(row[0])

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (k, v) VALUES (?, ?)",
                (scoped_key(self.tenant, key), value),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class SQLiteVectorCache:
    """Embeddings stored as lossless float64 blobs (default `{root}/cache/embeddings.sqlite`)."""

    def __init__(self, path: str | Path, *, tenant: str | None = None) -> None:
        self.tenant = validate_tenant(tenant)
        self._lock = threading.Lock()
        self._conn = _connect(
            path, "CREATE TABLE IF NOT EXISTS vectors (k TEXT PRIMARY KEY, v BLOB NOT NULL)"
        )

    def get_many(self, keys: Sequence[str]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        scoped = [scoped_key(self.tenant, key) for key in keys]
        with self._lock:
            for i in range(0, len(keys), _IN_LIST):
                chunk = scoped[i : i + _IN_LIST]
                marks = ",".join("?" * len(chunk))
                sql = f"SELECT k, v FROM vectors WHERE k IN ({marks})"  # noqa: S608 - only '?' marks
                for k, blob in self._conn.execute(sql, chunk):
                    vector = array("d")
                    vector.frombytes(blob)
                    out[str(k).removeprefix(f"{scoped_key(self.tenant, '')}")] = vector.tolist()
        return out

    def set_many(self, vectors: Mapping[str, Sequence[float]]) -> None:
        if not vectors:
            return
        rows = [(scoped_key(self.tenant, k), array("d", v).tobytes()) for k, v in vectors.items()]
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO vectors (k, v) VALUES (?, ?)", rows)
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class _SQLCacheBase:
    schema_version = 1

    def __init__(self, url: str | SecretRef, feature: str) -> None:
        try:
            import sqlalchemy as sa
        except ImportError as e:  # pragma: no cover - depends on the optional extra
            raise MissingExtraError("sql", feature) from e
        self._sa = sa
        self._engine = make_engine(url, feature)
        self._metadata = sa.MetaData()
        self._schema = sa.Table(
            "meldr_schema",
            self._metadata,
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("version", sa.Integer, nullable=False),
        )
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._engine.begin() as conn:
            self._metadata.create_all(conn)
            current = conn.execute(self._sa.select(self._schema.c.version)).scalar_one_or_none()
            if current is None:
                with suppress(self._sa.exc.IntegrityError):
                    conn.execute(self._schema.insert().values(id=1, version=self.schema_version))
        with self._engine.connect() as conn:
            current = conn.execute(self._sa.select(self._schema.c.version)).scalar_one_or_none()
        if current is not None and int(current) > self.schema_version:
            raise ConfigError(
                f"Database schema version {current} is newer than Meldr version "
                f"{self.schema_version}. Upgrade Meldr before use."
            )

    def close(self) -> None:
        self._engine.dispose()


class SQLResponseCache(_SQLCacheBase):
    """LLM response cache in a SQLAlchemy database."""

    def __init__(self, url: str | SecretRef, *, tenant: str | None = None) -> None:
        super().__init__(url, "SQL response cache")
        self.tenant = validate_tenant(tenant)
        sa = self._sa
        self._table = sa.Table(
            "meldr_response_cache",
            self._metadata,
            sa.Column("k", sa.String(255), primary_key=True),
            sa.Column("v", sa.Text, nullable=False),
        )
        self._metadata.create_all(self._engine)

    def get(self, key: str) -> str | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                self._sa.select(self._table.c.v).where(
                    self._table.c.k == scoped_key(self.tenant, key)
                )
            ).first()
        return None if row is None else str(row[0])

    def set(self, key: str, value: str) -> None:
        with self._engine.begin() as conn:
            scoped = scoped_key(self.tenant, key)
            conn.execute(self._sa.delete(self._table).where(self._table.c.k == scoped))
            conn.execute(self._table.insert().values(k=scoped, v=value))


class SQLVectorCache(_SQLCacheBase):
    """Embedding cache in a SQLAlchemy database."""

    def __init__(self, url: str | SecretRef, *, tenant: str | None = None) -> None:
        super().__init__(url, "SQL embedding cache")
        self.tenant = validate_tenant(tenant)
        sa = self._sa
        self._table = sa.Table(
            "meldr_vector_cache",
            self._metadata,
            sa.Column("k", sa.String(255), primary_key=True),
            sa.Column("v", sa.LargeBinary, nullable=False),
        )
        self._metadata.create_all(self._engine)

    def get_many(self, keys: Sequence[str]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        with self._engine.connect() as conn:
            for i in range(0, len(keys), _IN_LIST):
                chunk = [scoped_key(self.tenant, key) for key in keys[i : i + _IN_LIST]]
                if not chunk:
                    continue
                rows = conn.execute(
                    self._sa.select(self._table.c.k, self._table.c.v).where(
                        self._table.c.k.in_(chunk)
                    )
                ).all()
                for key, blob in rows:
                    vector = array("d")
                    vector.frombytes(blob)
                    prefix = f"{scoped_key(self.tenant, '')}"
                    out[str(key).removeprefix(prefix)] = vector.tolist()
        return out

    def set_many(self, vectors: Mapping[str, Sequence[float]]) -> None:
        if not vectors:
            return
        rows = [
            {"k": scoped_key(self.tenant, key), "v": array("d", vector).tobytes()}
            for key, vector in vectors.items()
        ]
        keys = [row["k"] for row in rows]
        with self._engine.begin() as conn:
            conn.execute(self._sa.delete(self._table).where(self._table.c.k.in_(keys)))
            conn.execute(self._table.insert(), rows)


class TenantResponseCache:
    """Scope a caller-provided response cache with a private key prefix."""

    def __init__(self, cache: ResponseCache, tenant: str) -> None:
        self.cache = cache
        self.tenant = validate_tenant(tenant)
        if self.tenant is None:  # pragma: no cover - the constructor requires a tenant
            raise ConfigError("TenantResponseCache must have a tenant name.")

    def get(self, key: str) -> str | None:
        return self.cache.get(scoped_key(self.tenant, key))

    def set(self, key: str, value: str) -> None:
        self.cache.set(scoped_key(self.tenant, key), value)

    def close(self) -> None:
        close = getattr(self.cache, "close", None)
        if callable(close):
            close()


class TenantVectorCache:
    """Scope a caller-provided embedding cache with a private key prefix."""

    def __init__(self, cache: VectorCache, tenant: str) -> None:
        self.cache = cache
        self.tenant = validate_tenant(tenant)
        if self.tenant is None:  # pragma: no cover - the constructor requires a tenant
            raise ConfigError("TenantVectorCache must have a tenant name.")

    def get_many(self, keys: Sequence[str]) -> dict[str, list[float]]:
        prefix = scoped_key(self.tenant, "")
        stored = self.cache.get_many([scoped_key(self.tenant, key) for key in keys])
        return {key.removeprefix(prefix): value for key, value in stored.items()}

    def set_many(self, vectors: Mapping[str, Sequence[float]]) -> None:
        self.cache.set_many(
            {scoped_key(self.tenant, key): vector for key, vector in vectors.items()}
        )

    def close(self) -> None:
        close = getattr(self.cache, "close", None)
        if callable(close):
            close()
