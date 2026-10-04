"""Meldr command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

try:
    import typer
    from rich.table import Table
except ImportError as e:
    from meldr.core.errors import MissingExtraError

    raise MissingExtraError("cli", "The Meldr CLI") from e

from meldr._version import __version__
from meldr.cli.common import (
    ConfigOpt,
    FileOpt,
    ParamOpt,
    SourceIdOpt,
    SourceOpt,
    TableOpt,
    batches,
    cli_errors,
    console,
    first_batch,
    load_meldr,
    parse_params,
)
from meldr.cli.eval import eval_cmd
from meldr.cli.review import review_cmd, spec_table, specs_app
from meldr.mapping.discovery import entity_to_yaml
from meldr.observability import configure_logging

app = typer.Typer(
    name="meldr",
    help="Harmonize data from all connector types into canonical entities.",
    no_args_is_help=True,
)
app.command("review")(review_cmd)
app.command("eval")(eval_cmd)
app.add_typer(specs_app, name="specs")


def _version(value: bool) -> None:
    if value:
        console.print(f"meldr {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version, is_eager=True, help="Print the version."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="JSON logs on stderr."),
) -> None:
    """Meldr CLI."""
    if verbose:
        configure_logging()


@app.command()
@cli_errors
def profile(
    source: str | None = SourceOpt,
    file: Path | None = FileOpt,
    source_id: str | None = SourceIdOpt,
    table: str | None = TableOpt,
    param: list[str] = ParamOpt,
    as_json: bool = typer.Option(False, "--json", help="Print the SourceSchema as JSON."),
    config: Path | None = ConfigOpt,
) -> None:
    """Profile the first batch of a source (types, patterns, hints)."""
    mr = load_meldr(config)
    schema = mr.profile(first_batch(mr, source, file, source_id, parse_params(param, table)))
    if as_json:
        console.print_json(schema.model_dump_json())
        return
    t = Table(
        "path",
        "dtype",
        "inferred",
        "nulls",
        "distinct",
        "patterns",
        "hints",
        title=f"{schema.source_id} · {schema.row_count} rows · {schema.fingerprint[:12]}",
    )
    for f in schema.fields:
        t.add_row(
            f.path,
            str(f.dtype),
            str(f.inferred_dtype),
            f"{f.null_ratio:.0%}",
            f"{f.distinct_ratio:.0%}",
            ", ".join(f.patterns),
            ", ".join(k for k in f.hints if k != "top_values"),
        )
    console.print(t)


@app.command("map")
@cli_errors
def map_cmd(
    target: str = typer.Option(..., "--target", "-t", help="Canonical entity name."),
    source: str | None = SourceOpt,
    file: Path | None = FileOpt,
    source_id: str | None = SourceIdOpt,
    table: str | None = TableOpt,
    param: list[str] = ParamOpt,
    config: Path | None = ConfigOpt,
) -> None:
    """Propose (or refresh) a mapping spec for a source and save it as a new version."""
    mr = load_meldr(config)
    batch = first_batch(mr, source, file, source_id, parse_params(param, table))
    console.print(spec_table(mr.propose(batch, target)))


@app.command()
@cli_errors
def run(
    target: str = typer.Option(..., "--target", "-t"),
    source: str | None = SourceOpt,
    file: Path | None = FileOpt,
    source_id: str | None = SourceIdOpt,
    table: str | None = TableOpt,
    param: list[str] = ParamOpt,
    write: bool = typer.Option(True, "--write/--no-write"),
    index: bool = typer.Option(False, "--index/--no-index", help="Embed rows for search()."),
    config: Path | None = ConfigOpt,
) -> None:
    """Harmonize every batch of a source into the table sink."""
    mr = load_meldr(config)
    for batch in batches(mr, source, file, source_id, parse_params(param, table)):
        r = mr.meld(batch, target, write=write, index=index).report
        console.print(
            f"{r.source_id} → {r.target}: in {r.rows_in}, out {r.rows_out}, quarantined "
            f"{r.rows_quarantined}, pending fields {r.fields_pending}, LLM calls "
            f"{r.llm_calls}, spec {r.spec_id} v{r.spec_version}"
            + (" (cached)" if r.spec_cache_hit else "")
        )
        for w in r.warnings:
            console.print(f"  [yellow]warning[/yellow] {w}")


@app.command()
@cli_errors
def extract(
    target: str | None = typer.Option(None, "--target", "-t"),
    source: str | None = SourceOpt,
    file: Path | None = FileOpt,
    source_id: str | None = SourceIdOpt,
    config: Path | None = ConfigOpt,
) -> None:
    """Segment documents/media, extract records with evidence, and index them."""
    mr = load_meldr(config)
    for batch in batches(mr, source, file, source_id, {}):
        res = mr.extract(batch, target)
        console.print(
            f"{len(res.segments)} segments, {res.data.height} valid records, "
            f"{res.quarantine.height} quarantined, LLM calls {res.report.llm_calls}"
        )
        for rec in res.records:
            verified = sum(e.verified for e in rec.evidence.values())
            console.print(f"  {rec.record_id}: {verified}/{len(rec.evidence)} evidence verified")


@app.command()
@cli_errors
def search(
    query: str,
    entity: str | None = typer.Option(None, "--entity", "-e"),
    k: int = typer.Option(10, "-k"),
    config: Path | None = ConfigOpt,
) -> None:
    """Vector search of the indexed records and segments."""
    for h in load_meldr(config).search(query, entity=entity, k=k):
        loc = {k: v for k, v in h.metadata.items() if k in ("page", "start_ms", "end_ms")}
        console.print(f"{h.score:.3f}  {h.id}  {json.dumps(loc)}\n    {h.text[:160]}")


@app.command()
@cli_errors
def explain(
    entity: str,
    record_id: str,
    field: str,
    source_id: str | None = typer.Option(None, "--source-id"),
    config: Path | None = ConfigOpt,
) -> None:
    """Show which source fields and steps made one harmonized value."""
    exp = load_meldr(config).explain(entity, record_id, field, source_id=source_id)
    console.print_json(exp.model_dump_json(exclude={"run"}))


@app.command()
@cli_errors
def discover(
    name: str = typer.Argument(..., help="Name for the draft entity."),
    file: list[Path] = typer.Option(..., "--file", "-f", help="Source files (repeatable)."),
    config: Path | None = ConfigOpt,
) -> None:
    """Draft a canonical entity YAML from source files (the command does not register it)."""
    mr = load_meldr(config)
    sample = [first_batch(mr, None, f, f.stem, {}) for f in file]
    console.print(entity_to_yaml(mr.discover_schema(sample, name)))


@app.command()
def plugins() -> None:
    """List installed entry-point plugins."""
    from meldr.plugins import GROUPS, discover

    for g in GROUPS:
        console.print(f"{g}: {', '.join(sorted(discover(g))) or '-'}")


if __name__ == "__main__":  # pragma: no cover
    app()
