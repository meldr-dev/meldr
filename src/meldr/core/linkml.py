"""Read LinkML schemas (https://linkml.io) as canonical entities, without the LinkML runtime.

Supported: classes with `slots`/`attributes`/`slot_usage`, `is_a` and `mixins` inheritance,
enums (`permissible_values` -> vocabulary), `identifier` (-> primary key), `required`,
`multivalued`, `pattern`, `minimum_value`/`maximum_value`, `aliases` (-> synonyms), `unit`,
and slot `examples`. The reader does not follow imports of other schema files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from meldr._yaml import read_mapping
from meldr.core.canonical import CanonicalEntity, CanonicalField
from meldr.core.dtypes import DataType
from meldr.core.errors import ConfigError

_RANGES = {
    "string": DataType.STRING,
    "str": DataType.STRING,
    "uri": DataType.STRING,
    "uriorcurie": DataType.STRING,
    "curie": DataType.STRING,
    "ncname": DataType.STRING,
    "integer": DataType.INTEGER,
    "int": DataType.INTEGER,
    "float": DataType.FLOAT,
    "double": DataType.FLOAT,
    "decimal": DataType.DECIMAL,
    "boolean": DataType.BOOLEAN,
    "bool": DataType.BOOLEAN,
    "date": DataType.DATE,
    "datetime": DataType.DATETIME,
}


def _load(schema: dict[str, Any] | str | Path) -> dict[str, Any]:
    return schema if isinstance(schema, dict) else read_mapping(schema, "LinkML schema")


def _ancestry(classes: dict[str, Any], name: str, seen: set[str] | None = None) -> list[str]:
    """Class names, root first: the `is_a` chain, then mixins, then `name`."""
    seen = seen or set()
    if name in seen or name not in classes:
        return []
    seen.add(name)
    cls = classes[name] or {}
    order: list[str] = []
    for parent in [cls.get("is_a"), *(cls.get("mixins") or [])]:
        if parent:
            order += [c for c in _ancestry(classes, parent, seen) if c not in order]
    return [*order, name]


def _slot_specs(schema: dict[str, Any], class_name: str) -> dict[str, dict[str, Any]]:
    classes = schema.get("classes") or {}
    global_slots = schema.get("slots") or {}
    specs: dict[str, dict[str, Any]] = {}
    for name in _ancestry(classes, class_name):
        cls = classes[name] or {}
        for slot in cls.get("slots") or []:
            specs.setdefault(slot, dict(global_slots.get(slot) or {}))
        for slot, attrs in (cls.get("attributes") or {}).items():
            specs[slot] = {**specs.get(slot, {}), **(attrs or {})}
        for slot, usage in (cls.get("slot_usage") or {}).items():
            specs[slot] = {**specs.get(slot, {}), **(usage or {})}
    return specs


def _field(name: str, spec: dict[str, Any], schema: dict[str, Any]) -> CanonicalField:
    enums = schema.get("enums") or {}
    range_ = spec.get("range") or schema.get("default_range") or "string"
    vocabulary = None
    if range_ in enums:
        vocabulary = [str(v) for v in ((enums[range_] or {}).get("permissible_values") or {})]
        dtype = DataType.STRING
    else:
        dtype = _RANGES.get(str(range_).lower(), DataType.STRING)
    if spec.get("multivalued"):
        dtype = DataType.ARRAY
    unit = spec.get("unit") or {}
    return CanonicalField(
        name=name,
        dtype=dtype,
        description=spec.get("description") or spec.get("title") or "",
        required=bool(spec.get("required") or spec.get("identifier")),
        synonyms=[str(a) for a in spec.get("aliases") or []],
        unit=unit.get("ucum_code") or unit.get("symbol") if isinstance(unit, dict) else None,
        vocabulary=vocabulary,
        pattern=spec.get("pattern"),
        examples=[str(e.get("value")) for e in spec.get("examples") or [] if isinstance(e, dict)],
        minimum=spec.get("minimum_value"),
        maximum=spec.get("maximum_value"),
    )


def from_linkml(
    schema: dict[str, Any] | str | Path, class_name: str, *, version: str | None = None
) -> CanonicalEntity:
    """One LinkML class as a canonical entity (with the schema `version` if it is semver)."""
    data = _load(schema)
    if class_name not in (data.get("classes") or {}):
        known = ", ".join(sorted(data.get("classes") or {})) or "none"
        raise ConfigError(f"LinkML schema has no class '{class_name}'. Classes: {known}.")
    specs = _slot_specs(data, class_name)
    schema_version = str(data.get("version", ""))
    semver = schema_version.count(".") == 2 and schema_version.replace(".", "").isdigit()
    return CanonicalEntity(
        name=class_name,
        version=version or (schema_version if semver else "1.0.0"),
        description=(data["classes"][class_name] or {}).get("description", ""),
        fields=[_field(n, s, data) for n, s in specs.items()],
        primary_key=[n for n, s in specs.items() if s.get("identifier")],
    )
