"""Run log in SQLite (default `{root}/lineage.sqlite`)."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import suppress
from pathlib import Path
from typing import Protocol

from meldr._sql import make_engine
from meldr._tenant import scoped_key, validate_tenant
from meldr.core.errors import ConfigError, MissingExtraError
from meldr.observability import RunReport
from meldr.secrets import SecretRef


class LineageStore(Protocol):
    def record_run(self, report: RunReport) -> None: ...
    def runs(self, source_id: str | None = None, limit: int = 50) -> list[RunReport]: ...
    def get(self, run_id: str) -> RunReport | None: ...


class SQLiteLineageStore:
    def __init__(self, path: str | Path, *, tenant: str | None = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.tenant = validate_tenant(tenant)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(p, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, source_id TEXT, "
            "target TEXT, started_at TEXT, report TEXT NOT NULL)"
        )
        self._conn.commit()

    def record_run(self, report: RunReport) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?)",
                (
                    scoped_key(self.tenant, report.run_id),
                    report.source_id,
                    report.target,
                    report.started_at.isoformat(),
                    report.model_dump_json(),
                ),
            )
            self._conn.commit()

    def runs(self, source_id: str | None = None, limit: int = 50) -> list[RunReport]:
        sql = "SELECT report FROM runs"
        args: tuple[object, ...] = ()
        if source_id is not None:
            sql += " WHERE source_id = ?"
            args = (source_id,)
        if self.tenant is not None:
            sql += " WHERE " if " WHERE " not in sql else " AND "
            sql += " run_id LIKE ?"
            args = (*args, f"{scoped_key(self.tenant, '')}%")
        sql += " ORDER BY started_at DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, (*args, limit)).fetchall()
        return [RunReport.model_validate_json(r[0]) for r in rows]

    def get(self, run_id: str) -> RunReport | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT report FROM runs WHERE run_id = ?",
                (scoped_key(self.tenant, run_id),),
            ).fetchone()
        return RunReport.model_validate_json(row[0]) if row else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class SQLLineageStore:
    """Run reports in a SQLAlchemy database."""

    schema_version = 1

    def __init__(self, url: str | SecretRef, *, tenant: str | None = None) -> None:
        try:
            import sqlalchemy as sa
        except ImportError as e:  # pragma: no cover - depends on the optional extra
            raise MissingExtraError("sql", "SQL lineage storage") from e
        self._sa = sa
        self.tenant = validate_tenant(tenant)
        self._engine = make_engine(url, "SQL lineage storage")
        metadata = sa.MetaData()
        self._runs = sa.Table(
            "meldr_runs",
            metadata,
            sa.Column("run_id", sa.String(255), primary_key=True),
            sa.Column("source_id", sa.String(255), nullable=False),
            sa.Column("target", sa.String(255)),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("report", sa.Text, nullable=False),
        )
        self._schema = sa.Table(
            "meldr_schema",
            metadata,
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("version", sa.Integer, nullable=False),
        )
        self._metadata = metadata
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

    def record_run(self, report: RunReport) -> None:
        with self._engine.begin() as conn:
            key = scoped_key(self.tenant, report.run_id)
            conn.execute(self._sa.delete(self._runs).where(self._runs.c.run_id == key))
            conn.execute(
                self._runs.insert().values(
                    run_id=key,
                    source_id=report.source_id,
                    target=report.target,
                    started_at=report.started_at,
                    report=report.model_dump_json(),
                )
            )

    def runs(self, source_id: str | None = None, limit: int = 50) -> list[RunReport]:
        query = (
            self._sa.select(self._runs.c.report)
            .order_by(self._runs.c.started_at.desc())
            .limit(limit)
        )
        if source_id is not None:
            query = query.where(self._runs.c.source_id == source_id)
        if self.tenant is not None:
            query = query.where(self._runs.c.run_id.like(f"{scoped_key(self.tenant, '')}%"))
        with self._engine.connect() as conn:
            rows = conn.execute(query).all()
        return [RunReport.model_validate_json(str(row[0])) for row in rows]

    def get(self, run_id: str) -> RunReport | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                self._sa.select(self._runs.c.report).where(
                    self._runs.c.run_id == scoped_key(self.tenant, run_id)
                )
            ).first()
        return RunReport.model_validate_json(str(row[0])) if row else None

    def close(self) -> None:
        self._engine.dispose()


class TenantLineageStore:
    """Scope a caller-provided lineage store with a private run ID prefix."""

    def __init__(self, store: LineageStore, tenant: str) -> None:
        self.store = store
        self.tenant = validate_tenant(tenant)
        if self.tenant is None:  # pragma: no cover - the constructor requires a tenant
            raise ConfigError("TenantLineageStore must have a tenant name.")

    def _run_id(self, run_id: str) -> str:
        return scoped_key(self.tenant, run_id)

    def record_run(self, report: RunReport) -> None:
        stored = report.model_copy(update={"run_id": self._run_id(report.run_id)})
        self.store.record_run(stored)

    def runs(self, source_id: str | None = None, limit: int = 50) -> list[RunReport]:
        rows = self.store.runs(source_id, limit=max(limit * 10, 100))
        prefix = scoped_key(self.tenant, "")
        return [
            report.model_copy(update={"run_id": report.run_id.removeprefix(prefix)})
            for report in rows
            if report.run_id.startswith(prefix)
        ][:limit]

    def get(self, run_id: str) -> RunReport | None:
        report = self.store.get(self._run_id(run_id))
        return None if report is None else report.model_copy(update={"run_id": run_id})

    def close(self) -> None:
        close = getattr(self.store, "close", None)
        if callable(close):
            close()
