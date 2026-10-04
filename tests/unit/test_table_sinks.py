"""One behavioural contract for every table sink: idempotent upserts, sequence ordering, schema
evolution, guarded deletes, and a full meld round trip."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from meldr.core.errors import SinkError
from meldr.core.types import RawBatch
from meldr.indexing.sinks.base import RECORD_KEY, SEQUENCE, TableSink
from meldr.pipeline.meldr import Meldr
from tests.helpers import make_config

KEY = list(RECORD_KEY)


def _duckdb(tmp: Path) -> TableSink:
    from meldr.indexing.sinks.duckdb import DuckDBSink

    return DuckDBSink(tmp / "h.duckdb")


def _parquet(tmp: Path) -> TableSink:
    from meldr.indexing.sinks.parquet import ParquetSink

    return ParquetSink(tmp / "parquet")


def _sql(tmp: Path) -> TableSink:
    from meldr.indexing.sinks.sql import SQLTableSink

    return SQLTableSink(f"sqlite:///{tmp / 'h.db'}")


def _delta(tmp: Path) -> TableSink:
    pytest.importorskip("deltalake")
    from meldr.indexing.sinks.delta import DeltaSink

    return DeltaSink(tmp / "delta")


def _iceberg(tmp: Path) -> TableSink:
    pytest.importorskip("pyiceberg")
    from meldr.indexing.sinks.iceberg import IcebergSink

    (tmp / "warehouse").mkdir()
    catalog = {"type": "sql", "uri": f"sqlite:///{tmp / 'catalog.db'}"}
    return IcebergSink("test", properties={**catalog, "warehouse": f"file://{tmp / 'warehouse'}"})


SINKS: dict[str, Callable[[Path], TableSink]] = {
    "duckdb": _duckdb,
    "parquet": _parquet,
    "sql": _sql,
    "delta": _delta,
    "iceberg": _iceberg,
}


@pytest.fixture(params=sorted(SINKS))
def sink(request: pytest.FixtureRequest, tmp_path: Path) -> TableSink:
    return SINKS[request.param](tmp_path)


def rows(ids: list[str], values: list[Any], seq: list[int | None] | None = None) -> pl.DataFrame:
    df = pl.DataFrame({"_meldr_source_id": ["s"] * len(ids), "_meldr_record_id": ids, "v": values})
    return df.with_columns(pl.Series(SEQUENCE, seq, dtype=pl.Int64)) if seq is not None else df


def stored(sink: TableSink, column: str = "v") -> list[Any]:
    return sink.read("Org").sort("_meldr_record_id")[column].to_list()


def test_upsert_is_idempotent_and_latest_wins(sink: TableSink) -> None:
    sink.write("Org", rows(["1", "2"], [1, 2]), key=KEY)
    sink.write("Org", rows(["1", "2"], [1, 2]), key=KEY)
    assert stored(sink) == [1, 2]
    sink.write("Org", rows(["2", "3", "3"], [20, 0, 30]), key=KEY)
    assert stored(sink) == [1, 20, 30]


def test_older_sequence_never_overwrites_newer(sink: TableSink) -> None:
    sink.write("Org", rows(["1", "2"], [1, 2], seq=[5, 5]), key=KEY)
    sink.write("Org", rows(["1", "2"], [10, 20], seq=[6, 4]), key=KEY)
    assert stored(sink) == [10, 2]
    sink.write("Org", rows(["1"], [99], seq=[None]), key=KEY)
    assert stored(sink) == [10, 2]


def test_new_columns_are_added(sink: TableSink) -> None:
    sink.write("Org", rows(["1"], [1]), key=KEY)
    later = rows(["2"], [2]).with_columns(pl.lit("x").alias("extra"))
    sink.write("Org", later, key=KEY)
    assert stored(sink, "extra") == [None, "x"]


def test_delete_respects_sequence(sink: TableSink) -> None:
    sink.write("Org", rows(["1", "2", "3"], [1, 2, 3], seq=[5, 5, 5]), key=KEY)
    sink.delete("Org", rows(["1", "2"], [0, 0], seq=[5, 4]).select([*KEY, SEQUENCE]))
    assert stored(sink, "_meldr_record_id") == ["2", "3"]
    sink.delete("Missing", rows(["1"], [0]).select(KEY))
    assert sink.read("Missing").height == 0


def test_delete_without_sequence(sink: TableSink) -> None:
    sink.write("Org", rows(["1", "2"], [1, 2]), key=KEY)
    sink.delete("Org", rows(["1", "9"], [0, 0]).select(KEY))
    assert stored(sink, "_meldr_record_id") == ["2"]


def test_custom_key_and_unsafe_names(sink: TableSink) -> None:
    links = pl.DataFrame({"a": ["x", "x"], "b": ["y", "z"], "score": [0.5, 0.9]})
    sink.write("Org__entity_links", links, key=["a", "b"])
    sink.write("Org__entity_links", links.with_columns(pl.lit(1.0).alias("score")), key=["a", "b"])
    assert sink.read("Org__entity_links").sort("b")["score"].to_list() == [1.0, 1.0]
    with pytest.raises(SinkError):
        sink.write("bad name;", links, key=["a", "b"])


def test_meld_round_trip(
    request: pytest.FixtureRequest, tmp_path: Path, erp_df: pl.DataFrame
) -> None:
    for name, make in SINKS.items():
        root = tmp_path / name
        root.mkdir()
        mr = Meldr(make_config(root / "state"), table_sink=make(root))
        batch = RawBatch.from_polars(erp_df, "erp")
        first = mr.meld(batch, "Organization")
        mr.meld(batch, "Organization")
        saved = mr.table_sink.read("Organization")
        assert saved.height == first.data.height, name
        assert set(first.data.columns) <= set(saved.columns), name
