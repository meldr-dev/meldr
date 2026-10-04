"""Versioned MappingSpec storage: YAML files you can commit and review in PRs."""

from __future__ import annotations

import builtins
import json
import os
import re
import threading
import uuid
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import ValidationError

from meldr._compat import UTC
from meldr._sql import make_engine
from meldr._tenant import scoped_identifier, scoped_key, tenant_token, validate_tenant
from meldr.core.errors import ConfigError, SpecConflictError, SpecValidationError
from meldr.core.mapping import MappingSpec, SpecStatus
from meldr.secrets import SecretRef

_SAFE = re.compile(r"^[A-Za-z0-9_.\-]+$")


class _Unset:
    pass


UNSET = _Unset()
_COMPARE_EXCLUDE = {
    "version",
    "parent_version",
    "created_at",
    "updated_at",
    "approved_by",
}


class MappingStore(Protocol):
    def find_approved(
        self, source_id: str, target: str, fingerprint: str
    ) -> MappingSpec | None: ...
    def latest(self, source_id: str, target: str) -> MappingSpec | None: ...
    def get(self, spec_id: str, version: int) -> MappingSpec: ...
    def save(
        self,
        spec: MappingSpec,
        *,
        expected_latest: int | _Unset | None = UNSET,
        actor: str = "meldr",
        action: str = "save",
    ) -> MappingSpec: ...
    def list(self, source_id: str | None = None) -> list[tuple[str, int, SpecStatus]]: ...


def _same_revision(left: MappingSpec, right: MappingSpec) -> bool:
    return left.model_dump(mode="json", exclude=_COMPARE_EXCLUDE) == right.model_dump(
        mode="json", exclude=_COMPARE_EXCLUDE
    )


def _check_name(value: str, what: str) -> str:
    if not _SAFE.match(value):
        raise SpecValidationError(
            f"{what} '{value}' can contain only letters, digits, '.', '_' and '-'."
        )
    return value


def _split(spec_id: str) -> tuple[str, str]:
    source_id, _, target = spec_id.rpartition(".")
    if not source_id:
        raise SpecValidationError(f"spec_id '{spec_id}' must look like '<source>.<target>'.")
    return source_id, target


class FileMappingStore:
    """`{root}/specs/{source_id}/{target}/v{version:04d}.yaml` plus an `audit.jsonl` log.

    Every save publishes a new version with an atomic hard link. So a save does not overwrite or
    half-write a version, even with more than one writer process.
    """

    def __init__(self, root: str | Path, *, tenant: str | None = None) -> None:
        self.tenant = validate_tenant(tenant)
        self.root = Path(root) / "specs"
        if self.tenant is not None:
            self.root = self.root / self.tenant
        self.root.mkdir(parents=True, exist_ok=True)
        self._audit_path = self.root / "audit.jsonl"
        self._lock = threading.Lock()

    def _dir(self, source_id: str, target: str) -> Path:
        return self.root / _check_name(source_id, "source_id") / _check_name(target, "target")

    def _versions(self, source_id: str, target: str) -> builtins.list[int]:
        d = self._dir(source_id, target)
        if not d.is_dir():
            return []
        return sorted(int(p.stem[1:]) for p in d.glob("v*.yaml") if p.stem[1:].isdigit())

    def _load(self, source_id: str, target: str, version: int) -> MappingSpec:
        path = self._dir(source_id, target) / f"v{version:04d}.yaml"
        if not path.exists():
            raise SpecValidationError(f"Spec {source_id}.{target} v{version} is not in the store.")
        return MappingSpec.from_yaml(path.read_text())

    def find_approved(self, source_id: str, target: str, fingerprint: str) -> MappingSpec | None:
        """Latest version for this fingerprint, if it is fully APPROVED."""
        for version in reversed(self._versions(source_id, target)):
            spec = self._load(source_id, target, version)
            if spec.source_fingerprint == fingerprint:
                return spec if spec.status == SpecStatus.APPROVED else None
        return None

    def latest(self, source_id: str, target: str) -> MappingSpec | None:
        versions = self._versions(source_id, target)
        return self._load(source_id, target, versions[-1]) if versions else None

    def get(self, spec_id: str, version: int) -> MappingSpec:
        return self._load(*_split(spec_id), version)

    def history(self, spec_id: str) -> builtins.list[int]:
        return self._versions(*_split(spec_id))

    def save(
        self,
        spec: MappingSpec,
        *,
        expected_latest: int | _Unset | None = UNSET,
        actor: str = "meldr",
        action: str = "save",
    ) -> MappingSpec:
        """Persist `spec` as the next version and return it with `version` set."""
        d = self._dir(spec.source_id, spec.target_entity)
        d.mkdir(parents=True, exist_ok=True)
        with self._lock:
            while True:
                versions = self._versions(spec.source_id, spec.target_entity)
                previous = versions[-1] if versions else None
                if expected_latest is not UNSET and previous != expected_latest:
                    raise SpecConflictError(
                        f"Spec {spec.spec_id} changed. Expected version {expected_latest!r}. "
                        f"Latest version is {previous!r}. Read the latest spec and try again."
                    )
                if previous is not None:
                    try:
                        latest = self._load(spec.source_id, spec.target_entity, previous)
                    except (SpecValidationError, ValidationError):
                        latest = None
                    if latest is not None and _same_revision(spec, latest):
                        return latest
                nxt = (previous or 0) + 1
                now = datetime.now(UTC)
                parent = spec.parent_version if spec.parent_version is not None else previous
                saved = spec.model_copy(
                    update={"version": nxt, "parent_version": parent, "updated_at": now}
                )
                tmp = d / f".tmp-{uuid.uuid4().hex}.yaml"
                tmp.write_text(saved.to_yaml())
                try:
                    os.link(tmp, d / f"v{nxt:04d}.yaml")
                    break
                except FileExistsError:
                    if expected_latest is not UNSET:
                        raise SpecConflictError(
                            f"A different writer changed spec {spec.spec_id} during the save. "
                            "Read the latest spec and try again."
                        ) from None
                    continue  # a different writer took this version number
                finally:
                    tmp.unlink()
            self._audit(
                {
                    "spec_id": saved.spec_id,
                    "version": nxt,
                    "action": action,
                    "actor": actor,
                    "status": str(saved.status),
                    "at": now.isoformat(),
                }
            )
            return saved

    def _audit(self, entry: dict[str, Any]) -> None:
        with self._audit_path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")

    def audit_log(self, spec_id: str | None = None) -> builtins.list[dict[str, Any]]:
        if not self._audit_path.exists():
            return []
        rows = [json.loads(line) for line in self._audit_path.read_text().splitlines() if line]
        return [r for r in rows if spec_id is None or r["spec_id"] == spec_id]

    def list(self, source_id: str | None = None) -> builtins.list[tuple[str, int, SpecStatus]]:
        out: builtins.list[tuple[str, int, SpecStatus]] = []
        for src_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if source_id is not None and src_dir.name != source_id:
                continue
            for tgt_dir in sorted(p for p in src_dir.iterdir() if p.is_dir()):
                versions = self._versions(src_dir.name, tgt_dir.name)
                if versions:
                    spec = self._load(src_dir.name, tgt_dir.name, versions[-1])
                    out.append((spec.spec_id, spec.version, spec.status))
        return out


class SQLMappingStore:
    """Versioned mapping specs in a SQLAlchemy database."""

    schema_version = 1

    def __init__(self, url: str | SecretRef, *, tenant: str | None = None) -> None:
        try:
            import sqlalchemy as sa
        except ImportError as e:  # pragma: no cover - depends on the optional extra
            from meldr.core.errors import MissingExtraError

            raise MissingExtraError("sql", "SQL mapping storage") from e
        self._sa = sa
        self._engine = make_engine(url, "SQL mapping storage")
        self.tenant = validate_tenant(tenant)
        self._lock = threading.Lock()
        metadata = sa.MetaData()
        self._specs = sa.Table(
            "meldr_specs",
            metadata,
            sa.Column("source_id", sa.String(255), primary_key=True),
            sa.Column("target", sa.String(255), primary_key=True),
            sa.Column("version", sa.Integer, primary_key=True),
            sa.Column("spec_id", sa.String(511), nullable=False),
            sa.Column("fingerprint", sa.String(128), nullable=False),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("body", sa.Text, nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Index("ix_meldr_specs_fingerprint", "source_id", "target", "fingerprint"),
        )
        self._audit_table = sa.Table(
            "meldr_audit",
            metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("spec_id", sa.String(511), nullable=False),
            sa.Column("version", sa.Integer, nullable=False),
            sa.Column("action", sa.String(64), nullable=False),
            sa.Column("actor", sa.String(255), nullable=False),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("at", sa.DateTime(timezone=True), nullable=False),
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
        sa = self._sa
        with self._engine.begin() as conn:
            self._metadata.create_all(conn)
            current = conn.execute(sa.select(self._schema.c.version)).scalar_one_or_none()
            if current is None:
                with suppress(sa.exc.IntegrityError):
                    conn.execute(self._schema.insert().values(id=1, version=self.schema_version))
        with self._engine.connect() as conn:
            current = conn.execute(sa.select(self._schema.c.version)).scalar_one_or_none()
        if current is not None and int(current) > self.schema_version:
            raise ConfigError(
                f"Database schema version {current} is newer than Meldr version "
                f"{self.schema_version}. Upgrade Meldr before use."
            )

    def _row_spec(self, row: Any) -> MappingSpec:
        return MappingSpec.from_yaml(str(row._mapping["body"]))

    def _source_key(self, source_id: str) -> str:
        return scoped_key(self.tenant, source_id)

    def _audit_key(self, spec_id: str) -> str:
        return scoped_key(self.tenant, spec_id)

    def _latest_connection(self, conn: Any, source_id: str, target: str) -> MappingSpec | None:
        row = conn.execute(
            self._sa.select(self._specs)
            .where(
                self._specs.c.source_id == self._source_key(source_id),
                self._specs.c.target == target,
            )
            .order_by(self._specs.c.version.desc())
            .limit(1)
        ).first()
        return self._row_spec(row) if row is not None else None

    def find_approved(self, source_id: str, target: str, fingerprint: str) -> MappingSpec | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                self._sa.select(self._specs)
                .where(
                    self._specs.c.source_id == self._source_key(source_id),
                    self._specs.c.target == target,
                    self._specs.c.fingerprint == fingerprint,
                )
                .order_by(self._specs.c.version.desc())
                .limit(1)
            ).first()
        if row is None:
            return None
        spec = self._row_spec(row)
        return spec if spec.status == SpecStatus.APPROVED else None

    def latest(self, source_id: str, target: str) -> MappingSpec | None:
        with self._engine.connect() as conn:
            return self._latest_connection(conn, source_id, target)

    def get(self, spec_id: str, version: int) -> MappingSpec:
        source_id, target = _split(spec_id)
        with self._engine.connect() as conn:
            row = conn.execute(
                self._sa.select(self._specs).where(
                    self._specs.c.source_id == self._source_key(source_id),
                    self._specs.c.target == target,
                    self._specs.c.version == version,
                )
            ).first()
        if row is None:
            raise SpecValidationError(f"Spec {spec_id} v{version} is not in the store.")
        return self._row_spec(row)

    def history(self, spec_id: str) -> builtins.list[int]:
        source_id, target = _split(spec_id)
        with self._engine.connect() as conn:
            rows = conn.execute(
                self._sa.select(self._specs.c.version)
                .where(
                    self._specs.c.source_id == self._source_key(source_id),
                    self._specs.c.target == target,
                )
                .order_by(self._specs.c.version)
            ).all()
        return [int(row[0]) for row in rows]

    def save(
        self,
        spec: MappingSpec,
        *,
        expected_latest: int | _Unset | None = UNSET,
        actor: str = "meldr",
        action: str = "save",
    ) -> MappingSpec:
        for attempt in range(5):
            try:
                with self._engine.begin() as conn:
                    previous = self._latest_connection(conn, spec.source_id, spec.target_entity)
                    previous_version = previous.version if previous is not None else None
                    if expected_latest is not UNSET and previous_version != expected_latest:
                        raise SpecConflictError(
                            f"Spec {spec.spec_id} changed. Expected version {expected_latest!r}. "
                            f"Latest version is {previous_version!r}. Read the latest spec and "
                            "try again."
                        )
                    if previous is not None and _same_revision(spec, previous):
                        return previous
                    next_version = (previous_version or 0) + 1
                    parent = (
                        spec.parent_version if spec.parent_version is not None else previous_version
                    )
                    saved = spec.model_copy(
                        update={
                            "version": next_version,
                            "parent_version": parent,
                            "updated_at": datetime.now(UTC),
                        }
                    )
                    conn.execute(
                        self._specs.insert().values(
                            source_id=self._source_key(saved.source_id),
                            target=saved.target_entity,
                            version=saved.version,
                            spec_id=saved.spec_id,
                            fingerprint=saved.source_fingerprint,
                            status=saved.status.value,
                            body=saved.to_yaml(),
                            created_at=saved.created_at,
                        )
                    )
                    conn.execute(
                        self._audit_table.insert().values(
                            spec_id=self._audit_key(saved.spec_id),
                            version=saved.version,
                            action=action,
                            actor=actor,
                            status=saved.status.value,
                            at=datetime.now(UTC),
                        )
                    )
                    return saved
            except self._sa.exc.IntegrityError as e:
                if expected_latest is not UNSET or attempt == 4:
                    raise SpecConflictError(
                        f"A different writer changed spec {spec.spec_id} during the save. "
                        "Read the latest spec and try again."
                    ) from e
        raise AssertionError("unreachable")  # pragma: no cover

    def audit_log(self, spec_id: str | None = None) -> builtins.list[dict[str, Any]]:
        with self._engine.connect() as conn:
            query = self._sa.select(self._audit_table).order_by(self._audit_table.c.id)
            if spec_id is not None:
                query = query.where(self._audit_table.c.spec_id == self._audit_key(spec_id))
            elif self.tenant is not None:
                query = query.where(self._audit_table.c.spec_id.like(f"{self._audit_key('')}%"))
            rows = conn.execute(query).mappings().all()
        result = [dict(row) for row in rows]
        if self.tenant is not None:
            for row in result:
                row["spec_id"] = str(row["spec_id"]).removeprefix(f"{scoped_key(self.tenant, '')}")
        return result

    def list(self, source_id: str | None = None) -> builtins.list[tuple[str, int, SpecStatus]]:
        with self._engine.connect() as conn:
            query = self._sa.select(self._specs).order_by(
                self._specs.c.source_id, self._specs.c.target, self._specs.c.version
            )
            if source_id is not None:
                query = query.where(self._specs.c.source_id == self._source_key(source_id))
            rows = conn.execute(query).all()
        latest: dict[str, MappingSpec] = {}
        for row in rows:
            spec = self._row_spec(row)
            latest[spec.spec_id] = spec
        return [(spec.spec_id, spec.version, spec.status) for spec in latest.values()]

    def close(self) -> None:
        self._engine.dispose()


class TenantMappingStore:
    """Scope a caller-provided mapping store and keep its public spec IDs."""

    def __init__(self, store: MappingStore, tenant: str) -> None:
        self.store = store
        checked = validate_tenant(tenant)
        if checked is None:  # pragma: no cover - the constructor requires a tenant
            raise ConfigError("TenantMappingStore must have a tenant name.")
        self.tenant = checked

    def _source(self, source_id: str) -> str:
        return scoped_identifier(self.tenant, source_id)

    def _spec_for_store(self, spec: MappingSpec) -> MappingSpec:
        source_id = self._source(spec.source_id)
        return spec.model_copy(
            update={"source_id": source_id, "spec_id": f"{source_id}.{spec.target_entity}"}
        )

    def _spec_for_user(self, spec: MappingSpec) -> MappingSpec:
        source_id = spec.source_id.removeprefix(f"{tenant_token(self.tenant)}__")
        return spec.model_copy(
            update={"source_id": source_id, "spec_id": f"{source_id}.{spec.target_entity}"}
        )

    def find_approved(self, source_id: str, target: str, fingerprint: str) -> MappingSpec | None:
        spec = self.store.find_approved(self._source(source_id), target, fingerprint)
        return None if spec is None else self._spec_for_user(spec)

    def latest(self, source_id: str, target: str) -> MappingSpec | None:
        spec = self.store.latest(self._source(source_id), target)
        return None if spec is None else self._spec_for_user(spec)

    def get(self, spec_id: str, version: int) -> MappingSpec:
        source_id, _, target = spec_id.rpartition(".")
        stored = self.store.get(f"{self._source(source_id)}.{target}", version)
        return self._spec_for_user(stored)

    def save(
        self,
        spec: MappingSpec,
        *,
        expected_latest: int | _Unset | None = UNSET,
        actor: str = "meldr",
        action: str = "save",
    ) -> MappingSpec:
        saved = self.store.save(
            self._spec_for_store(spec),
            expected_latest=expected_latest,
            actor=actor,
            action=action,
        )
        return self._spec_for_user(saved)

    def list(self, source_id: str | None = None) -> builtins.list[tuple[str, int, SpecStatus]]:
        rows = self.store.list(self._source(source_id) if source_id is not None else None)
        prefix = f"{tenant_token(self.tenant)}__"
        return [
            (spec_id.removeprefix(prefix), version, status)
            for spec_id, version, status in rows
            if spec_id.startswith(prefix)
        ]

    def history(self, spec_id: str) -> builtins.list[int]:
        history = getattr(self.store, "history")  # noqa: B009
        return cast(
            builtins.list[int],
            history(f"{self._source(spec_id.rpartition('.')[0])}.{spec_id.rpartition('.')[2]}"),
        )

    def audit_log(self, spec_id: str | None = None) -> builtins.list[dict[str, Any]]:
        audit = getattr(self.store, "audit_log")  # noqa: B009
        rows = audit(
            None
            if spec_id is None
            else f"{self._source(spec_id.rpartition('.')[0])}.{spec_id.rpartition('.')[2]}"
        )
        prefix = f"{tenant_token(self.tenant)}__"
        return [{**row, "spec_id": str(row["spec_id"]).removeprefix(prefix)} for row in rows]

    def close(self) -> None:
        close = getattr(self.store, "close", None)
        if callable(close):
            close()
