"""Canonical entities and the schema registry."""

from __future__ import annotations

import builtins
import datetime as dt
import decimal
import types
import typing
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from meldr._yaml import read_mapping
from meldr.core.dtypes import DataType
from meldr.core.errors import ConfigError


class CanonicalField(BaseModel):
    """One field of a canonical entity."""

    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: DataType
    description: str = ""
    required: bool = False
    synonyms: list[str] = Field(default_factory=list)
    unit: str | None = None
    vocabulary: list[str] | None = None
    pattern: str | None = None
    pii: bool = False
    examples: list[str] = Field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None


# Keys read from Pydantic `json_schema_extra` (on each field, and on the model config).
_FIELD_EXTRAS = (
    "synonyms",
    "unit",
    "vocabulary",
    "pattern",
    "pii",
    "examples",
    "minimum",
    "maximum",
)
_ENTITY_EXTRAS = (
    "primary_key",
    "embedding_template",
    "sequence_field",
    "extraction",
    "relations",
)


class EntityExtraction(BaseModel):
    """Extraction settings for one canonical entity."""

    model_config = ConfigDict(extra="forbid")

    cardinality: Literal["one", "many"] = "one"


class CanonicalEntity(BaseModel):
    """A versioned canonical entity definition."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    description: str = ""
    fields: list[CanonicalField]
    primary_key: list[str] = Field(default_factory=list)
    sequence_field: str | None = None
    embedding_template: str | None = None
    extraction: EntityExtraction = Field(default_factory=EntityExtraction)
    relations: list[str] = Field(default_factory=list)

    @field_validator("version")
    @classmethod
    def _semver(cls, v: str) -> str:
        parts = v.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError(f"version '{v}' must be semver like 1.0.0")
        return v

    @model_validator(mode="after")
    def _check(self) -> CanonicalEntity:
        names = [f.name for f in self.fields]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate field names: {sorted(dupes)}")
        missing = [k for k in self.primary_key if k not in names]
        if missing:
            raise ValueError(f"primary_key references unknown fields: {missing}")
        if self.sequence_field is not None and self.sequence_field not in names:
            raise ValueError(f"sequence_field '{self.sequence_field}' is not a field")
        if len(set(self.relations)) != len(self.relations):
            raise ValueError("relations must not contain duplicates")
        return self

    def field(self, name: str) -> CanonicalField:
        """Return the field called `name` or raise KeyError."""
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"Entity '{self.name}' has no field '{name}'.")

    def field_names(self) -> list[str]:
        """Field names in declaration order."""
        return [f.name for f in self.fields]

    @classmethod
    def from_pydantic(cls, model: type[BaseModel], *, version: str = "1.0.0") -> CanonicalEntity:
        """Build an entity from a Pydantic model.

        `json_schema_extra` can set synonyms and the other `_FIELD_EXTRAS` keys.
        """
        fields: list[CanonicalField] = []
        for name, info in model.model_fields.items():
            extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
            dtype, optional = _dtype_from_annotation(info.annotation)
            field_extra = {k: v for k, v in extra.items() if k in _FIELD_EXTRAS}
            fields.append(
                CanonicalField.model_validate(
                    {
                        "name": name,
                        "dtype": dtype,
                        "description": info.description or "",
                        "required": info.is_required() and not optional,
                        **field_extra,
                    }
                )
            )
        config_extra = model.model_config.get("json_schema_extra")
        meta = config_extra if isinstance(config_extra, dict) else {}
        return cls.model_validate(
            {
                "name": model.__name__,
                "version": version,
                "description": (model.__doc__ or "").strip(),
                "fields": fields,
                **{k: v for k, v in meta.items() if k in _ENTITY_EXTRAS},
            }
        )

    @classmethod
    def from_linkml(
        cls, schema: dict[str, Any] | str | Path, class_name: str, *, version: str | None = None
    ) -> CanonicalEntity:
        """One class of a LinkML schema (see meldr.core.linkml for the supported features)."""
        from meldr.core.linkml import from_linkml

        return from_linkml(schema, class_name, version=version)

    @classmethod
    def from_json_schema(
        cls, schema: dict[str, Any], *, name: str | None = None, version: str = "1.0.0"
    ) -> CanonicalEntity:
        """Build an entity from a flat JSON Schema object (draft 2020-12 keywords).

        Meldr metadata uses `x-` keys:

        - On properties: `x-synonyms`, `x-unit`, `x-pii`, and `x-dtype` (for example, "decimal").
        - At the root: `x-version`, `x-primary-key`, `x-sequence-field`, and
          `x-embedding-template`.
        """
        props = schema.get("properties")
        if not isinstance(props, dict) or not props:
            raise ConfigError("The JSON Schema must be an object with `properties`.")
        required = set(schema.get("required", []))
        fields = []
        for field_name, spec in props.items():
            types = spec.get("type", "string")
            types = [t for t in types if t != "null"] if isinstance(types, list) else [types]
            dtype = (
                DataType(spec["x-dtype"])
                if "x-dtype" in spec
                else _JSON_FORMATS.get(spec.get("format", ""))
                or _JSON_TYPES.get(types[0] if types else "string", DataType.STRING)
            )
            enum = spec.get("enum")
            fields.append(
                CanonicalField(
                    name=field_name,
                    dtype=dtype,
                    description=spec.get("description") or spec.get("title") or "",
                    required=field_name in required,
                    synonyms=list(spec.get("x-synonyms", [])),
                    unit=spec.get("x-unit"),
                    vocabulary=[str(v) for v in enum if v is not None] if enum else None,
                    pattern=spec.get("pattern"),
                    pii=bool(spec.get("x-pii", False)),
                    examples=[str(v) for v in spec.get("examples", [])],
                    minimum=spec.get("minimum"),
                    maximum=spec.get("maximum"),
                )
            )
        try:
            return cls(
                name=name or schema.get("title") or "Entity",
                version=schema.get("x-version", version),
                description=schema.get("description", ""),
                fields=fields,
                primary_key=list(schema.get("x-primary-key", [])),
                sequence_field=schema.get("x-sequence-field"),
                embedding_template=schema.get("x-embedding-template"),
            )
        except ValidationError as e:
            raise ConfigError(f"Invalid entity from JSON Schema:\n{e}") from e

    def to_json_schema(self) -> dict[str, Any]:
        """The entity as a flat JSON Schema object (round-trips through from_json_schema)."""
        props: dict[str, Any] = {}
        for f in self.fields:
            json_type, fmt = _JSON_OUT[f.dtype]
            p: dict[str, Any] = {"type": json_type}
            if fmt:
                p["format"] = fmt
            if f.dtype == DataType.DECIMAL:
                p["x-dtype"] = "decimal"
            optional: dict[str, Any] = {
                "description": f.description or None,
                "enum": f.vocabulary,
                "pattern": f.pattern,
                "minimum": f.minimum,
                "maximum": f.maximum,
                "examples": f.examples or None,
                "x-synonyms": f.synonyms or None,
                "x-unit": f.unit,
                "x-pii": f.pii or None,
            }
            p.update({k: v for k, v in optional.items() if v is not None})
            props[f.name] = p
        out: dict[str, Any] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": self.name,
            "x-version": self.version,
            "type": "object",
            "properties": props,
            "required": [f.name for f in self.fields if f.required],
        }
        if self.description:
            out["description"] = self.description
        if self.primary_key:
            out["x-primary-key"] = self.primary_key
        if self.sequence_field:
            out["x-sequence-field"] = self.sequence_field
        if self.embedding_template:
            out["x-embedding-template"] = self.embedding_template
        return out

    @classmethod
    def from_yaml(cls, path: str | Path) -> CanonicalEntity:
        """Load an entity from a YAML file."""
        raw = read_mapping(path, "entity file")
        try:
            return cls.model_validate(raw)
        except ValidationError as e:
            raise ConfigError(f"Invalid entity definition in {path}:\n{e}") from e


_JSON_TYPES = {
    "string": DataType.STRING,
    "integer": DataType.INTEGER,
    "number": DataType.FLOAT,
    "boolean": DataType.BOOLEAN,
    "array": DataType.ARRAY,
    "object": DataType.JSON,
}
_JSON_FORMATS = {"date": DataType.DATE, "date-time": DataType.DATETIME}
_JSON_OUT: dict[DataType, tuple[str, str | None]] = {
    DataType.STRING: ("string", None),
    DataType.INTEGER: ("integer", None),
    DataType.FLOAT: ("number", None),
    DataType.DECIMAL: ("number", None),
    DataType.BOOLEAN: ("boolean", None),
    DataType.DATE: ("string", "date"),
    DataType.DATETIME: ("string", "date-time"),
    DataType.JSON: ("object", None),
    DataType.ARRAY: ("array", None),
    DataType.BINARY: ("string", "binary"),
    DataType.UNKNOWN: ("string", None),
}

_PY_TO_DTYPE: dict[Any, DataType] = {
    str: DataType.STRING,
    int: DataType.INTEGER,
    float: DataType.FLOAT,
    decimal.Decimal: DataType.DECIMAL,
    bool: DataType.BOOLEAN,
    dt.date: DataType.DATE,
    dt.datetime: DataType.DATETIME,
    bytes: DataType.BINARY,
    dict: DataType.JSON,
}


def _dtype_from_annotation(ann: Any) -> tuple[DataType, bool]:
    optional = False
    origin = typing.get_origin(ann)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        optional = len(args) < len(typing.get_args(ann))
        ann = args[0] if len(args) == 1 else ann
        origin = typing.get_origin(ann)
    if origin in (list, builtins.list):
        return DataType.ARRAY, optional
    if origin is dict:
        return DataType.JSON, optional
    return _PY_TO_DTYPE.get(ann, DataType.UNKNOWN), optional


def _semver_key(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split("."))


class SchemaRegistry:
    """In-memory registry of canonical entities by name and version."""

    def __init__(self) -> None:
        self._entities: dict[str, dict[str, CanonicalEntity]] = {}

    def register(self, entity: CanonicalEntity) -> None:
        """Add an entity version. A second `register()` of the same entity does nothing."""
        versions = self._entities.setdefault(entity.name, {})
        existing = versions.get(entity.version)
        if existing is not None and existing != entity:
            raise ConfigError(
                f"Entity {entity.name} {entity.version} is already registered with a different "
                "definition. Bump the version to change it."
            )
        versions[entity.version] = entity

    def get(self, name: str, version: str | None = None) -> CanonicalEntity:
        """Return the given `version`, or the latest when `version` is None."""
        versions = self._entities.get(name)
        if not versions:
            known = ", ".join(sorted(self._entities)) or "none"
            raise ConfigError(f"Unknown entity '{name}'. Registered entities: {known}.")
        if version is None:
            return versions[max(versions, key=_semver_key)]
        if version not in versions:
            raise ConfigError(
                f"Entity '{name}' has no version {version}. Registered versions: "
                f"{sorted(versions)}."
            )
        return versions[version]

    def load_dir(self, path: str | Path) -> None:
        """Register every *.yaml / *.yml entity in a directory."""
        p = Path(path)
        if not p.is_dir():
            raise ConfigError(f"Cannot find schema directory {p}. Fix `schemas.paths`.")
        for f in sorted([*p.glob("*.yaml"), *p.glob("*.yml")]):
            self.register(CanonicalEntity.from_yaml(f))

    def load_linkml(
        self, path: str | Path, classes: Iterable[str] | None = None
    ) -> builtins.list[str]:
        """Register LinkML classes (default: every non-abstract, non-mixin class)."""
        from meldr.core.linkml import _load, from_linkml

        data = _load(path)
        names = (
            list(classes)
            if classes is not None
            else [
                n
                for n, c in (data.get("classes") or {}).items()
                if not (c or {}).get("abstract") and not (c or {}).get("mixin")
            ]
        )
        for n in names:
            self.register(from_linkml(data, n))
        return names

    def list(self) -> builtins.list[tuple[str, str]]:
        """All (name, version) pairs."""
        return sorted((n, v) for n, vs in self._entities.items() for v in vs)
