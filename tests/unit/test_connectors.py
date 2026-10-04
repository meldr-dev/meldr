"""Tests for Python and dlt connector adapters."""

from __future__ import annotations

import sys
import types
from typing import Any

import polars as pl
import pytest

from meldr.config import MeldrConfig
from meldr.connectors.base import build_connector
from meldr.connectors.dlt import DltConnector
from meldr.connectors.python import PythonConnector
from meldr.core.errors import ConnectorError
from meldr.core.types import Modality


def test_python_connector_batches_rows_and_accepts_params() -> None:
    def source(limit: int) -> list[dict[str, Any]]:
        return [{"id": i} for i in range(limit)]

    connector = PythonConnector("python", source, batch_size=2)
    batches = list(connector.pull(limit=5))
    assert [batch.table["id"].to_list() for batch in batches] == [[0, 1], [2, 3], [4]]


def test_python_connector_accepts_frames_and_json_rows() -> None:
    frames = list(PythonConnector("frames", pl.DataFrame({"id": [1, 2]}), batch_size=1).pull())
    assert [batch.table["id"].to_list() for batch in frames] == [[1], [2]]
    records = list(
        PythonConnector(
            "json",
            [{"id": 1}, {"id": 2}],
            batch_size=2,
            modality=Modality.JSON,
        ).pull()
    )
    assert [record.data for record in records[0].records] == [{"id": 1}, {"id": 2}]


def test_python_connector_rejects_scalar() -> None:
    with pytest.raises(ConnectorError, match="must yield"):
        list(PythonConnector("python", "not rows").pull())


def test_python_connector_from_config() -> None:
    cfg = MeldrConfig.from_dict(
        {
            "sources": {
                "python": {
                    "connector": "python",
                    "options": {"source": [{"id": "1"}]},
                }
            }
        }
    )
    connector = build_connector("python", cfg.sources["python"], 10)
    assert connector.pull().__next__().table["id"].to_list() == ["1"]  # type: ignore[index]


def test_dlt_connector_selects_a_resource(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_dlt = types.ModuleType("dlt")
    fake_dlt.source = object()
    monkeypatch.setitem(sys.modules, "dlt", fake_dlt)

    class Source:
        def with_resources(self, name: str) -> list[dict[str, int]]:
            assert name == "items"
            return [{"id": 1}, {"id": 2}]

    connector = DltConnector("dlt", Source(), resource="items", batch_size=2)
    batches = list(connector.pull())
    assert batches[0].table["id"].to_list() == [1, 2]  # type: ignore[index]
