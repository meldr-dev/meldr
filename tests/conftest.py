from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from meldr import Meldr
from meldr.core.canonical import SchemaRegistry
from meldr.domains import pack_path
from tests.helpers import make_config


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Unit tests must never open real sockets (respx/MockTransport work in-process)."""
    if request.node.get_closest_marker("live_llm"):
        return

    def guard(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network access is not allowed in tests.")

    monkeypatch.setattr(socket.socket, "connect", guard)


@pytest.fixture
def registry() -> SchemaRegistry:
    reg = SchemaRegistry()
    reg.load_dir(pack_path("generic"))
    return reg


@pytest.fixture
def org(registry: SchemaRegistry):  # type: ignore[no-untyped-def]
    return registry.get("Organization")


@pytest.fixture
def erp_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "VNDR_ID": ["V1", "V2", "V3", "V4"],
            "VNDR_NM": ["Acme Corp", "Beta LLC", "Gamma Inc", "Delta Co"],
            "VNDR_TYP_CD": ["S", "V", "C", "S"],
            "REV_CENTS": [100000, 250000, None, 99],
            "EST_DT": ["03/14/2001", "12/25/1999", "01/22/2010", None],
            "CONTACT_EMAIL": ["a@b.com", "c@d.org", "e@f.net", "g@h.io"],
            "CTRY": ["US", "US", "CA", "GB"],
            "ACTV_FLG": ["Y", "N", "Y", "Y"],
            "LAST_UPD_USR": ["x", "y", "z", "w"],
        }
    )


@pytest.fixture
def mr(tmp_path: Path) -> Meldr:
    return Meldr(make_config(tmp_path / "state"))
