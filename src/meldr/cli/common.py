"""Shared CLI helpers: load the config and select the batches."""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NoReturn, ParamSpec, TypeVar

import typer
from rich.console import Console

from meldr.config import MeldrConfig
from meldr.connectors.files import FileConnector
from meldr.core.errors import MeldrError
from meldr.core.types import RawBatch
from meldr.pipeline.meldr import Meldr

console = Console()
err = Console(stderr=True)

ConfigOpt = typer.Option(None, "--config", "-c", help="Path to meldr.yaml.")
SourceOpt = typer.Option(None, "--source", "-s", help="Configured source name.")
FileOpt = typer.Option(None, "--file", "-f", help="Ad-hoc file or folder instead of a source.")
SourceIdOpt = typer.Option(None, "--source-id", help="source_id for --file (default: file stem).")
TableOpt = typer.Option(None, "--table", help="SQL table for sql sources.")
ParamOpt = typer.Option([], "--param", "-p", help="Extra pull() params as key=value.")


def load_meldr(config: Path | None) -> Meldr:
    path = config or (Path("meldr.yaml") if Path("meldr.yaml").exists() else None)
    return Meldr.from_config(path) if path else Meldr(MeldrConfig())


def parse_params(params: list[str], table: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in params:
        if "=" not in p:
            raise typer.BadParameter(f"--param expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k] = v
    if table:
        out["table"] = table
    return out


def batches(
    mr: Meldr,
    source: str | None,
    file: Path | None,
    source_id: str | None,
    params: dict[str, Any],
) -> Iterator[RawBatch]:
    if source and not file:
        yield from mr.connectors[source].pull(**params)
    elif file and not source:
        connector = FileConnector(
            source_id or file.stem, file, batch_size=mr.config.execution.batch_size
        )
        yield from connector.pull(**params)
    else:
        raise typer.BadParameter("Pass --source or --file, but not the two.")


def first_batch(
    mr: Meldr,
    source: str | None,
    file: Path | None,
    source_id: str | None,
    params: dict[str, Any],
) -> RawBatch:
    for b in batches(mr, source, file, source_id, params):
        return b
    raise MeldrError("The source returned no data.")


def fail(e: Exception) -> NoReturn:
    err.print(f"[red]error:[/red] {e}")
    raise typer.Exit(code=1)


P = ParamSpec("P")
R = TypeVar("R")


def cli_errors(fn: Callable[P, R]) -> Callable[P, R]:
    """Report a MeldrError on one line and exit with code 1, without a traceback."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(*args, **kwargs)
        except MeldrError as e:
            fail(e)

    return wrapper
