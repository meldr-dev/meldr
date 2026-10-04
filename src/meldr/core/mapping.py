"""The declarative MappingSpec: how one source maps to one canonical entity."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from meldr._compat import StrEnum
from meldr._yaml import load as load_yaml
from meldr.core.dtypes import DataType


class TransformStep(BaseModel):
    """One whitelisted transform op with typed args."""

    model_config = ConfigDict(extra="forbid")

    op: str
    args: dict[str, Any] = Field(default_factory=dict)


class FieldStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"


class Origin(StrEnum):
    HEURISTIC = "heuristic"
    LLM = "llm"
    HUMAN = "human"


class DryRunStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: int
    loss_rate: float
    invalid_rate: float
    failed_examples: int


class FieldMapping(BaseModel):
    """The source paths and transform steps that make one target field."""

    model_config = ConfigDict(extra="forbid")

    target: str
    sources: list[str]
    steps: list[TransformStep] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    origin: Origin
    status: FieldStatus = FieldStatus.PROPOSED
    review_reason: str | None = None
    carried_from: int | None = None
    dryrun: DryRunStats | None = None


class Alternative(BaseModel):
    """A ranked candidate source for a target (shown to reviewers)."""

    model_config = ConfigDict(extra="forbid")

    path: str
    score: float


class ReviewCandidate(BaseModel):
    """One source path that a reviewer can choose."""

    model_config = ConfigDict(extra="forbid")

    source: str
    confidence: float = Field(ge=0.0, le=1.0)


class ReviewItem(BaseModel):
    """One mapping decision that must have a caller-owned review."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "uncertain_field",
        "unmapped_required",
        "new_values",
        "low_confidence_extraction",
    ]
    target: str
    candidates: list[ReviewCandidate] = Field(default_factory=list)
    spec_id: str
    version: int
    count: int | None = Field(default=None, ge=0)


class ReviewDecision(BaseModel):
    """One caller decision for a review item."""

    model_config = ConfigDict(extra="forbid")

    target: str
    action: Literal["approve", "reject", "edit"]
    changes: dict[str, Any] | str | None = None
    reason: str = "rejected"


class GeneratorInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    transforms_version: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    meldr_version: str


class SpecStatus(StrEnum):
    DRAFT = "draft"
    PARTIAL = "partial"
    APPROVED = "approved"


class MappingSpec(BaseModel):
    """A versioned mapping from one source shape to one canonical entity."""

    model_config = ConfigDict(extra="forbid")

    spec_id: str
    version: int
    parent_version: int | None = None
    source_id: str
    source_fingerprint: str
    source_type_signature: dict[str, DataType]
    target_entity: str
    target_version: str
    fields: list[FieldMapping]
    unmapped_sources: list[str] = Field(default_factory=list)
    unmapped_targets: list[str] = Field(default_factory=list)
    required_targets: list[str] = Field(default_factory=list)
    candidates: dict[str, list[Alternative]] = Field(default_factory=dict)
    generator: GeneratorInfo
    created_at: datetime
    updated_at: datetime
    approved_by: str | None = None

    @property
    def status(self) -> SpecStatus:
        """APPROVED when every required target is approved and nothing is PROPOSED."""
        approved = {f.target for f in self.fields if f.status == FieldStatus.APPROVED}
        if not approved:
            return SpecStatus.DRAFT
        any_proposed = any(f.status == FieldStatus.PROPOSED for f in self.fields)
        if not any_proposed and set(self.required_targets) <= approved:
            return SpecStatus.APPROVED
        return SpecStatus.PARTIAL

    def approved_fields(self) -> list[FieldMapping]:
        """Field mappings with status APPROVED."""
        return [f for f in self.fields if f.status == FieldStatus.APPROVED]

    def runnable_fields(self, *, include_proposed: bool = False) -> list[FieldMapping]:
        """Approved mappings that have sources (also proposed ones, if `include_proposed`)."""
        allowed = (
            {FieldStatus.APPROVED, FieldStatus.PROPOSED}
            if include_proposed
            else {FieldStatus.APPROVED}
        )
        return [f for f in self.fields if f.sources and f.status in allowed]

    def pending_fields(self) -> list[FieldMapping]:
        """Field mappings that wait for review, most decision-relevant first.

        Required targets come first. Then come the fields with the smallest margin between the
        two best candidates (margin sampling). A reviewer adds the most value on these decisions.
        """
        required = set(self.required_targets)

        def margin(f: FieldMapping) -> float:
            scores = [a.score for a in self.candidates.get(f.target, [])[:2]]
            return scores[0] - scores[1] if len(scores) == 2 else 1.0

        pending = [f for f in self.fields if f.status == FieldStatus.PROPOSED]
        return sorted(pending, key=lambda f: (f.target not in required, margin(f)))

    def mapping_for(self, target: str) -> FieldMapping | None:
        """The mapping for `target`, or None if the spec has no mapping for it."""
        return next((f for f in self.fields if f.target == target), None)

    def to_yaml(self) -> str:
        """Serialize to YAML (stable key order)."""
        data = self.model_dump(mode="json", exclude_none=True)
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, text: str) -> MappingSpec:
        """Parse a spec from YAML text."""
        return cls.model_validate(load_yaml(text))
