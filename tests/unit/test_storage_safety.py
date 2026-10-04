"""Defensive storage: brief source reads, tagged queries, no silent overwrites, locked sinks."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import sqlalchemy as sa

from meldr.api import apply_mapping, propose_mapping
from meldr.connectors.sql import SQLConnector
from meldr.core.canonical import CanonicalEntity
from meldr.core.errors import SinkError
from meldr.core.mapping import TransformStep
from meldr.core.types import RawBatch
from meldr.eval.tables import entity_from_table
from meldr.indexing.sinks.base import entity_docs
from meldr.indexing.sinks.duckdb import DuckDBSink
from meldr.mapping import review
from meldr.pipeline.meldr import Meldr


def _vendors(tmp_path: Path) -> str:
    db = tmp_path / "erp.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE vendors (id TEXT, name TEXT)")
        conn.executemany("INSERT INTO vendors VALUES (?, ?)", [("1", "A"), ("2", "B"), ("3", "C")])
    return f"sqlite:///{db}"


def test_sql_source_connection_is_released_before_batches(tmp_path: Path) -> None:
    connector = SQLConnector("erp */ drop", _vendors(tmp_path), batch_size=2)
    engine = connector._get_engine()
    statements: list[str] = []

    @sa.event.listens_for(engine, "before_cursor_execute")
    def capture(conn: Any, cursor: Any, statement: str, *args: Any) -> None:
        statements.append(statement)

    batches = connector.pull(table="vendors")
    first = next(batches)
    assert engine.pool.checkedout() == 0  # nothing downstream runs inside the source transaction
    assert [first.table.height, *(b.table.height for b in batches)] == [2, 1]  # type: ignore[union-attr]
    select = next(s for s in statements if "FROM vendors" in s)
    assert "/* meldr source=erp____drop */" in select


def test_failed_field_quarantines_rows_instead_of_writing_nulls() -> None:
    source = pl.DataFrame({"id": ["1", "2"], "name": ["Acme", "Beta"]})
    entity = entity_from_table(pl.DataFrame({"id": ["1"], "name": ["x"]}), "E")
    spec = review.approve(propose_mapping(source, entity, source_id="s"), user="t", above=0.0)
    broken = TransformStep(op="regex_extract", args={"pattern": "("})
    fields = [
        f.model_copy(update={"steps": [broken]}) if f.target == "name" else f for f in spec.fields
    ]
    out = apply_mapping(source, spec.model_copy(update={"fields": fields}), entity)
    assert out.data.height == 0 and out.quarantine.height == 2
    assert out.error_counts == {"value_lost:name": 2}


def test_duckdb_sink_is_read_only_and_local(tmp_path: Path) -> None:
    sink = DuckDBSink(tmp_path / "h.duckdb")
    sink.write("Org", pl.DataFrame({"v": [1]}), key=["v"])
    (tmp_path / "secret.csv").write_text("a\n1\n")
    assert sink.sql("SELECT v FROM Org")["v"].to_list() == [1]
    with pytest.raises(SinkError):
        sink.sql("DROP TABLE Org")
    with pytest.raises(SinkError):
        sink.sql(f"SELECT * FROM read_csv('{tmp_path / 'secret.csv'}')")


def test_harmonized_tables_describe_themselves(
    mr: Meldr, org: CanonicalEntity, erp_df: pl.DataFrame
) -> None:
    table_doc, docs = entity_docs(org)
    assert "Organization" in table_doc and "Personal data." in docs["email"]
    assert docs["org_type"].startswith("Relationship to us") and "One of:" in docs["org_type"]
    mr.meld(RawBatch.from_polars(erp_df, "erp"), "Organization")
    assert isinstance(mr.table_sink, DuckDBSink)
    comments = dict(
        mr.table_sink.sql(
            "SELECT column_name, comment FROM duckdb_columns() WHERE table_name = 'Organization'"
        ).iter_rows()
    )
    assert comments["legal_name"] == docs["legal_name"]
    assert comments["_meldr_record_id"].startswith("Record ID within a source")
