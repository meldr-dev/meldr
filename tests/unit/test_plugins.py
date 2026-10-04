"""Entry-point plugins reach the Meldr: sinks, ops, and matchers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from meldr import Meldr, MeldrConfig, RawBatch
from meldr import plugins as plugins_mod
from meldr.core.dtypes import DataType
from meldr.core.errors import ConfigError
from meldr.indexing.sinks import TABLE_SINKS, VECTOR_SINKS, register_sink
from meldr.indexing.sinks.memory import MemoryVectorSink
from meldr.indexing.sinks.parquet import ParquetSink
from meldr.transforms.registry import OpArgs, OpDef


class _EntryPoint:
    def __init__(self, name: str, obj: Any) -> None:
        self.name = name
        self._obj = obj

    def load(self) -> Any:
        return self._obj


def test_entry_point_plugins_are_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def matcher(schema: Any, entity: Any) -> dict[str, Any]:
        calls.append(entity.name)
        return {}

    shout = OpDef(
        name="shout",
        args_model=OpArgs,
        arity="single",
        description="Uppercase with an exclamation mark.",
        build=lambda e, a: e[0].cast(pl.String).str.to_uppercase() + "!",
        output_dtype=DataType.STRING,
    )
    found = {
        "meldr.sinks": [
            _EntryPoint("lake", lambda cfg, root: ParquetSink(root / "lake")),
            _EntryPoint("vector.memory_plugin", lambda cfg, root: MemoryVectorSink()),
        ],
        "meldr.ops": [_EntryPoint("shout", shout)],
        "meldr.matchers": [_EntryPoint("mine", matcher)],
    }
    monkeypatch.setattr(plugins_mod, "_loaded", {})
    monkeypatch.setattr(plugins_mod, "entry_points", lambda group: found.get(group, []))
    config = MeldrConfig.from_dict(
        {
            "storage": {"root": str(tmp_path)},
            "sinks": {
                "tables": {"type": "lake"},
                "vectors": {"type": "memory_plugin"},
            },
        }
    )
    mr = Meldr(config)
    assert isinstance(mr.table_sink, ParquetSink)
    assert isinstance(mr.vector_sink, MemoryVectorSink)
    assert "shout" in mr.ops.names()
    mr.propose(RawBatch.from_polars(pl.DataFrame({"x": ["a"]}), "s"), "Person")
    assert calls == ["Person"]


def test_sink_registries_separate_kinds_and_reject_same_kind_collisions() -> None:
    table_name = "test_table_registry"
    vector_name = "test_vector_registry"
    register_sink(table_name, lambda cfg, root: ParquetSink(root / "table"), kind="table")
    register_sink(vector_name, lambda cfg, root: MemoryVectorSink(), kind="vector")

    assert table_name in TABLE_SINKS
    assert vector_name in VECTOR_SINKS
    assert table_name not in VECTOR_SINKS
    assert vector_name not in TABLE_SINKS

    with pytest.raises(ConfigError):
        register_sink(table_name, lambda cfg, root: ParquetSink(root / "other"), kind="table")
    with pytest.raises(ConfigError):
        register_sink("duckdb", lambda cfg, root: ParquetSink(root / "other"), kind="table")
