"""Contract checks for third-party connectors and sinks."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from meldr.core.errors import SinkError
from meldr.indexing.sinks.base import (
    RECORD_KEY,
    SEQUENCE,
    IndexItem,
    TableSink,
    VectorSink,
)

TableSinkFactory = Callable[[Path], TableSink]
VectorSinkFactory = Callable[[Path], VectorSink]


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@contextmanager
def _root(root: str | Path | None) -> Iterator[Path]:
    if root is not None:
        yield Path(root)
        return
    with TemporaryDirectory(prefix="meldr-test-") as path:
        yield Path(path)


def check_table_sink(factory: TableSinkFactory, *, root: str | Path | None = None) -> None:
    """Run the table sink contract against a factory."""
    with _root(root) as path:
        sink = factory(path)
        key = list(RECORD_KEY)
        first = pl.DataFrame(
            {"_meldr_source_id": ["s", "s"], "_meldr_record_id": ["1", "2"], "value": [1, 2]}
        )
        sink.write("Entity", first, key=key)
        sink.write("Entity", first, key=key)
        _check(sink.read("Entity").height == 2, "A repeated table write must not add rows.")

        newer = first.with_columns(pl.lit(3).alias(SEQUENCE))
        older = first.with_columns(pl.lit(2).alias(SEQUENCE))
        sink.write("Entity", newer, key=key)
        sink.write("Entity", older, key=key)
        _check(
            sink.read("Entity").sort("_meldr_record_id")["value"].to_list() == [1, 2],
            "A newer sequence must win over an older sequence.",
        )

        extra = first.head(1).with_columns(pl.lit("x").alias("extra"))
        sink.write("Entity", extra, key=key)
        _check("extra" in sink.read("Entity").columns, "A sink must add new columns.")

        sink.delete("Entity", first.head(1).select(key))
        _check(sink.read("Entity").height == 1, "A sink must delete matching keys.")
        try:
            sink.write("bad name;", first, key=key)
        except SinkError:
            pass
        else:
            raise AssertionError("A sink must reject unsafe table names.")


def check_vector_sink(factory: VectorSinkFactory, *, root: str | Path | None = None) -> None:
    """Run the vector sink contract against a factory."""
    with _root(root) as path:
        sink = factory(path)
        items = [
            IndexItem(id="a", vector=[1.0, 0.0], text="a", metadata={"source": "x"}),
            IndexItem(id="b", vector=[0.9, 0.1], text="b", metadata={"source": "y"}),
            IndexItem(id="c", vector=[0.0, 1.0], text="c", metadata={"source": "z"}),
        ]
        sink.upsert("records", items)
        sink.upsert("records", items)
        hits = sink.search("records", [1.0, 0.0], k=3)
        _check([hit.id for hit in hits] == ["a", "b", "c"], "A vector sink must rank by cosine.")
        filtered = sink.search("records", [1.0, 0.0], k=3, filters={"source": ["y", "z"]})
        _check([hit.id for hit in filtered] == ["b", "c"], "A vector sink must apply filters.")
        _check(sink.search("missing", [1.0, 0.0], k=3) == [], "A missing collection is empty.")
