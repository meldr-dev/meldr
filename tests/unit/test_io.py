"""Connectors and sinks."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import httpx
import polars as pl
import pytest
import respx
import sqlalchemy as sa

from meldr._sql import make_engine
from meldr.config import AuthConfig, PaginationConfig
from meldr.connectors.files import FileConnector
from meldr.connectors.rest import RestConnector
from meldr.connectors.sql import SQLConnector, check_read_only
from meldr.core.errors import ConnectorError, MissingExtraError
from meldr.core.types import Modality
from meldr.indexing.render import render
from meldr.indexing.sinks.base import IndexItem
from meldr.indexing.sinks.memory import MemoryVectorSink
from meldr.secrets import SecretRef


def test_file_connector_formats(tmp_path: Path) -> None:
    pl.DataFrame({"a": list(range(5))}).write_csv(tmp_path / "t.csv")
    pl.DataFrame({"a": [1, 2]}).write_parquet(tmp_path / "t.parquet")
    (tmp_path / "t.json").write_text(json.dumps([{"a": {"b": 1}}]))
    (tmp_path / "t.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
    (tmp_path / "notes.md").write_text("# Notes")
    batches = list(FileConnector("f", tmp_path, batch_size=2).pull())
    kinds = [
        (b.modality, b.table.height if b.table is not None else len(b.records)) for b in batches
    ]
    assert kinds.count((Modality.TABULAR, 2)) == 3  # 5-row CSV in 2+2+1, parquet 2
    assert (Modality.JSON, 1) in kinds and (Modality.JSON, 2) in kinds
    assert batches[-1].modality == Modality.DOCUMENT
    with pytest.raises(ConnectorError):
        list(FileConnector("f", tmp_path / "missing").pull())


def test_file_connector_reads_memory_uri() -> None:
    fsspec = pytest.importorskip("fsspec")
    fs = fsspec.filesystem("memory")
    with fs.open("bucket/remote.csv", "wb") as handle:
        handle.write(b"id,name\n1,Acme\n2,Beta\n")
    batches = list(FileConnector("remote", "memory://bucket", batch_size=1).pull())
    assert [batch.table["id"].to_list() for batch in batches] == [["1"], ["2"]]


def test_sql_connector_read_only(tmp_path: Path) -> None:
    db = tmp_path / "erp.db"
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE "vendors" (id TEXT, name TEXT)')
        conn.executemany("INSERT INTO vendors VALUES (?, ?)", [("1", "A"), ("2", "B")])
    conn = SQLConnector("erp", f"sqlite:///{db}", batch_size=1)
    assert [b.table.height for b in conn.pull(table="vendors")] == [1, 1]  # type: ignore[union-attr]
    q = list(conn.pull(query="SELECT name FROM vendors WHERE name = 'A'"))
    assert q[0].table["name"].to_list() == ["A"]  # type: ignore[index]
    for bad in [
        "DELETE FROM vendors",
        "SELECT 1; DROP TABLE vendors",
        "WITH x AS (DELETE FROM vendors) SELECT 1",
        "PRAGMA table_info(vendors)",
    ]:
        with pytest.raises(ConnectorError):
            check_read_only(bad)
    check_read_only("SELECT 'drop table' AS s -- delete\n")


def test_missing_warehouse_dialect_names_its_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise sa.exc.NoSuchModuleError("missing dialect")

    monkeypatch.setattr(sa, "create_engine", fail)
    with pytest.raises(MissingExtraError, match=r"meldr\[snowflake\]"):
        make_engine("snowflake://user:pass@example/db", "the SQL connector")


@respx.mock
def test_rest_cursor_auth_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HZ_TOKEN", "tok")
    route = respx.get("https://api.example.com/v2/vendors")
    route.side_effect = [
        httpx.Response(429, headers={"retry-after": "0"}),
        httpx.Response(200, json={"data": [{"id": 1}], "meta": {"next": "c2"}}),
        httpx.Response(200, json={"data": [{"id": 2}], "meta": {"next": None}}),
    ]
    rest = RestConnector(
        "api",
        "https://api.example.com/v2",
        endpoint="vendors",
        retry_base_s=0,
        auth=AuthConfig(type="bearer", token=SecretRef("${env:HZ_TOKEN}")),
        pagination=PaginationConfig(type="cursor", next_path="meta.next"),
        records_path="data",
    )
    batches = list(rest.pull())
    assert [b.records[0].data for b in batches] == [{"id": 1}, {"id": 2}]
    assert route.calls[-1].request.url.params["cursor"] == "c2"
    assert route.calls[-1].request.headers["authorization"] == "Bearer tok"


@respx.mock
def test_rest_refuses_off_host_redirect() -> None:
    respx.get("https://api.example.com/x").mock(
        return_value=httpx.Response(302, headers={"location": "https://evil.example.org/x"})
    )
    with pytest.raises(ConnectorError, match=r"evil\.example\.org"):
        list(RestConnector("api", "https://api.example.com", endpoint="x").pull())


@respx.mock
def test_rest_combines_pages_into_batches() -> None:
    route = respx.get("https://api.example.com/items")
    route.side_effect = [
        httpx.Response(200, json=[{"id": 1}, {"id": 2}]),
        httpx.Response(200, json=[{"id": 3}, {"id": 4}]),
        httpx.Response(200, json=[{"id": 5}]),
        httpx.Response(200, json=[]),
    ]
    rest = RestConnector(
        "api",
        "https://api.example.com",
        endpoint="items",
        pagination=PaginationConfig(type="page"),
        batch_size=3,
    )
    batches = list(rest.pull())
    assert [[r.record_id for r in b.records] for b in batches] == [
        ["0:0", "0:1", "1:0", "1:1"],
        ["2:0"],
    ]


@respx.mock
def test_rest_link_header_pagination() -> None:
    respx.get("https://api.example.com/items", params={"page": "2"}).mock(
        return_value=httpx.Response(200, json=[{"id": 2}])
    )
    respx.get("https://api.example.com/items").mock(
        return_value=httpx.Response(
            200,
            json=[{"id": 1}],
            headers={"link": '<https://api.example.com/items?page=2>; rel="next"'},
        )
    )
    rest = RestConnector(
        "api",
        "https://api.example.com",
        endpoint="items",
        pagination=PaginationConfig(type="link_header"),
    )
    assert [b.records[0].data["id"] for b in rest.pull()] == [1, 2]  # type: ignore[index]


def test_memory_vector_sink_filters() -> None:
    sink = MemoryVectorSink()
    sink.upsert(
        "c",
        [
            IndexItem(id="a", vector=[1, 0], text="a", metadata={"src": "x"}),
            IndexItem(id="b", vector=[0.9, 0.1], text="b", metadata={"src": "y"}),
        ],
    )
    assert [h.id for h in sink.search("c", [1, 0], k=5, filters={"src": "y"})] == ["b"]


def test_template_sandbox() -> None:
    assert render("{{ name }} ({{ kind }})", {"name": "Acme", "kind": None}, ["name"]) == "Acme ()"
    escaped = render("{{ name.__class__.__mro__ }}", {"name": "x"}, ["name"])
    assert "object" not in escaped
    assert render(None, {"a": 1, "b": None}, ["a", "b"]) == "a: 1"
