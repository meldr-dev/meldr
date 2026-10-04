"""Result containers returned by the Meldr."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any, cast

import polars as pl

from meldr.core.mapping import FieldMapping, MappingSpec, ReviewItem
from meldr.core.segments import ContentSegment, ExtractedRecord, Relation
from meldr.observability import RunReport


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_json_safe(v) for v in sorted(value, key=str)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _frame_rows(
    frame: pl.DataFrame,
    *,
    limit: int,
    pii_fields: tuple[str, ...] = (),
    reveal_pii: bool = False,
) -> list[dict[str, Any]]:
    if limit < 0:
        raise ValueError("limit must be zero or greater.")
    hidden = {"_meldr_source_row"}
    columns = [c for c in frame.columns if c not in hidden and not c.startswith("_meldr_loss__")]
    rows = frame.select(columns).head(limit).to_dicts()
    for row in rows:
        if not reveal_pii:
            for field in pii_fields:
                if field in row and row[field] is not None:
                    row[field] = "***"
    return [_json_safe(row) for row in rows]


@dataclass
class MeldrResult:
    data: pl.DataFrame
    quarantine: pl.DataFrame
    spec: MappingSpec
    pending_review: list[FieldMapping]
    report: RunReport
    pii_fields: tuple[str, ...] = ()

    def to_dict(
        self, *, include_rows: bool = False, limit: int = 100, reveal_pii: bool = False
    ) -> dict[str, Any]:
        """Return a JSON-safe summary, with rows only when requested."""
        result: dict[str, Any] = {
            "rows_valid": self.data.height,
            "rows_quarantined": self.quarantine.height,
            "values_lost": dict(self.report.values_lost),
            "warnings": list(self.report.warnings),
            "spec_id": self.spec.spec_id,
            "spec_version": self.spec.version,
            "pending_fields": [field.target for field in self.pending_review],
            "llm_calls": self.report.llm_calls,
        }
        if include_rows:
            result["rows"] = _frame_rows(
                self.data, limit=limit, pii_fields=self.pii_fields, reveal_pii=reveal_pii
            )
            result["quarantine"] = _frame_rows(
                self.quarantine, limit=limit, pii_fields=self.pii_fields, reveal_pii=reveal_pii
            )
        return cast(dict[str, Any], _json_safe(result))


@dataclass
class ExtractResult:
    records: list[ExtractedRecord]
    data: pl.DataFrame
    quarantine: pl.DataFrame
    segments: list[ContentSegment]
    report: RunReport
    relations: list[Relation] = field(default_factory=list)

    def to_dict(
        self, *, include_rows: bool = False, limit: int = 100, reveal_pii: bool = False
    ) -> dict[str, Any]:
        """Return a JSON-safe summary of an extraction."""
        result: dict[str, Any] = {
            "rows_valid": self.data.height,
            "rows_quarantined": self.quarantine.height,
            "warnings": list(self.report.warnings),
            "llm_calls": self.report.llm_calls,
            "relations": len(self.relations),
        }
        if include_rows:
            result["rows"] = _frame_rows(self.data, limit=limit, reveal_pii=reveal_pii)
            result["quarantine"] = _frame_rows(self.quarantine, limit=limit, reveal_pii=reveal_pii)
            result["records"] = [record.model_dump(mode="json") for record in self.records[:limit]]
        return cast(dict[str, Any], _json_safe(result))


@dataclass
class HarmonizeResult:
    """The caller-owned result of one black-box harmonization run."""

    data: pl.DataFrame
    quarantine: pl.DataFrame
    review: list[ReviewItem]
    children: dict[str, pl.DataFrame]
    report: RunReport
    spec: MappingSpec | None = None
    relations: pl.DataFrame = field(default_factory=pl.DataFrame)

    def to_dict(
        self, *, include_rows: bool = False, limit: int = 100, reveal_pii: bool = False
    ) -> dict[str, Any]:
        """Return a JSON-safe summary of one harmonization run."""
        result: dict[str, Any] = {
            "rows_valid": self.data.height,
            "rows_quarantined": self.quarantine.height,
            "review_items": len(self.review),
            "children": {name: frame.height for name, frame in self.children.items()},
            "relations": self.relations.height,
            "warnings": list(self.report.warnings),
            "llm_calls": self.report.llm_calls,
        }
        if self.spec is not None:
            result["spec_id"] = self.spec.spec_id
            result["spec_version"] = self.spec.version
        if include_rows:
            result["rows"] = _frame_rows(self.data, limit=limit, reveal_pii=reveal_pii)
            result["quarantine"] = _frame_rows(self.quarantine, limit=limit, reveal_pii=reveal_pii)
            result["review"] = [
                item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                for item in self.review[:limit]
            ]
        return cast(dict[str, Any], _json_safe(result))
