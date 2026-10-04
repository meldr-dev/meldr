"""Data classes, data policies, and effective-policy computation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from meldr._compat import StrEnum
from meldr.core.errors import PolicyViolation


class DataClass(StrEnum):
    """The kind of information that a request has."""

    SCHEMA = "schema"
    MASKED_SAMPLES = "masked_samples"
    SAMPLES = "samples"
    CONTENT = "content"


ALL_CLASSES: frozenset[DataClass] = frozenset(DataClass)
VENDOR_DEFAULT: frozenset[DataClass] = frozenset({DataClass.SCHEMA, DataClass.MASKED_SAMPLES})


class DataPolicy(BaseModel):
    """Set of data classes a provider (or source) can receive."""

    model_config = ConfigDict(extra="forbid")

    allowed: set[DataClass]

    @classmethod
    def default_for(cls, *, is_local: bool) -> DataPolicy:
        """Vendor providers: schema + masked samples. Local providers: everything."""
        return cls(allowed=set(ALL_CLASSES if is_local else VENDOR_DEFAULT))

    def allows(self, data_class: DataClass) -> bool:
        return data_class in self.allowed

    def intersect(self, other: DataPolicy | None) -> DataPolicy:
        if other is None:
            return self
        return DataPolicy(allowed=self.allowed & other.allowed)


class SplitPolicy(BaseModel):
    """Source policy with separate lists for vendor and local providers."""

    model_config = ConfigDict(extra="forbid")

    vendor: list[DataClass]
    local: list[DataClass]


SourcePolicy = list[DataClass] | SplitPolicy


def effective_policy(
    provider_policy: DataPolicy, source_policy: SourcePolicy | None, *, is_local: bool
) -> DataPolicy:
    """Provider policy ∩ source policy (split policies pick by `is_local`)."""
    if source_policy is None:
        return provider_policy
    chosen = (
        (source_policy.local if is_local else source_policy.vendor)
        if isinstance(source_policy, SplitPolicy)
        else source_policy
    )
    return provider_policy.intersect(DataPolicy(allowed=set(chosen)))


def check(policy: DataPolicy, contains: set[DataClass], *, provider: str) -> None:
    """Raise PolicyViolation when `contains` is not a subset of `policy`."""
    denied = sorted(contains - policy.allowed)
    if denied:
        raise PolicyViolation(
            f"The request contains data classes {denied} that provider '{provider}' cannot "
            f"receive (it can receive: {sorted(policy.allowed)}). Use a local provider "
            "(openai_compatible). Or, if your security team approves, widen the data_policy for "
            "this provider/source."
        )
