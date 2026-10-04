"""Whitelisted transform ops and their registry."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, ValidationError

from meldr.core.dtypes import DataType
from meldr.core.errors import SpecValidationError, validation_details
from meldr.core.mapping import FieldMapping, TransformStep

# Bump when a built-in op changes behavior after a release. Specs record the version they used.
TRANSFORMS_VERSION = "1"


class OpArgs(BaseModel):
    """Base class for typed op arguments."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class OpDef:
    """A registered transform op."""

    name: str
    args_model: type[OpArgs]
    arity: Literal["single", "multi"]
    build: Callable[[list[pl.Expr], OpArgs], pl.Expr]
    description: str
    nulls_intentionally: bool = False
    output_dtype: DataType | None = None


class TransformRegistry:
    """The only source of transforms. The LLM sees what `describe_for_llm` returns, no more."""

    def __init__(self) -> None:
        self._ops: dict[str, OpDef] = {}

    def register(self, op: OpDef) -> None:
        """Add or replace an op."""
        self._ops[op.name] = op

    def get(self, name: str) -> OpDef:
        """Return a registered op or raise SpecValidationError."""
        op = self._ops.get(name)
        if op is None:
            raise SpecValidationError(
                f"Unknown transform op '{name}'. Registered ops: {', '.join(sorted(self._ops))}."
            )
        return op

    def names(self) -> list[str]:
        return sorted(self._ops)

    def parse_args(self, step: TransformStep) -> OpArgs:
        """Validate one step's args."""
        op = self.get(step.op)
        try:
            return op.args_model.model_validate(step.args)
        except ValidationError as e:
            raise SpecValidationError(
                f"Invalid args for op '{step.op}': {validation_details(e)}"
            ) from e

    def validate(self, mapping: FieldMapping) -> list[OpArgs]:
        """Validate every step of a mapping, and check the arity rules."""
        if not mapping.sources:
            raise SpecValidationError(
                f"Mapping for '{mapping.target}' has no sources. Leave the target unmapped instead."
            )
        parsed: list[OpArgs] = []
        for i, step in enumerate(mapping.steps):
            op = self.get(step.op)
            if op.arity == "multi" and i != 0:
                raise SpecValidationError(
                    f"Op '{step.op}' combines more than one source, so it must be the first step "
                    f"(mapping for '{mapping.target}')."
                )
            parsed.append(self.parse_args(step))
        first_multi = bool(mapping.steps) and self.get(mapping.steps[0].op).arity == "multi"
        if len(mapping.sources) > 1 and not first_multi:
            raise SpecValidationError(
                f"Mapping for '{mapping.target}' has {len(mapping.sources)} sources, so its first "
                "step must be a multi-source op (concat, coalesce, template)."
            )
        if first_multi:
            check = getattr(parsed[0], "check_sources", None)
            if callable(check):
                check(len(mapping.sources))
        return parsed

    def output_dtype(self, mapping: FieldMapping, source_dtype: DataType) -> DataType:
        """Best-effort logical dtype after the steps (before the cast to the target dtype)."""
        dtype = source_dtype
        for step in mapping.steps:
            op = self.get(step.op)
            if step.op == "cast":
                dtype = DataType(step.args.get("dtype", dtype))
            elif op.output_dtype is not None:
                dtype = op.output_dtype
        return dtype

    def describe_for_llm(self) -> str:
        """Compact catalog: name, arity, args schema, description."""
        keep = ("type", "default", "items", "enum", "anyOf")
        lines = []
        for name in sorted(self._ops):
            op = self._ops[name]
            schema = op.args_model.model_json_schema()
            defs = schema.get("$defs", {})
            args = {}
            for k, v in schema.get("properties", {}).items():
                ref = v.get("$ref", "").rsplit("/", 1)[-1]
                args[k] = {kk: vv for kk, vv in {**defs.get(ref, {}), **v}.items() if kk in keep}
            lines.append(
                f"- {name} [{op.arity}]: {op.description} args={json.dumps(args, default=str)}"
            )
        return "\n".join(lines)


def default_registry() -> TransformRegistry:
    """A registry with every built-in op."""
    from meldr.transforms.ops import BUILTIN_OPS

    reg = TransformRegistry()
    for op in BUILTIN_OPS:
        reg.register(op)
    return reg
