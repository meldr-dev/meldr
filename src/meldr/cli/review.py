"""`meldr review` and `meldr specs ...`."""

from __future__ import annotations

import getpass
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import typer
import yaml
from rich.table import Table

from meldr.cli.common import (
    ConfigOpt,
    FileOpt,
    ParamOpt,
    SourceIdOpt,
    SourceOpt,
    TableOpt,
    cli_errors,
    console,
    first_batch,
    load_meldr,
    parse_params,
)
from meldr.core.errors import MeldrError
from meldr.core.mapping import FieldStatus, MappingSpec
from meldr.mapping import drift, review
from meldr.mapping.sssom import to_sssom
from meldr.mapping.store import FileMappingStore
from meldr.pipeline.meldr import Meldr

specs_app = typer.Typer(help="List, show, diff, and refresh mapping specs.", no_args_is_help=True)


def _file_store(mr: Meldr) -> FileMappingStore:
    if not isinstance(mr.store, FileMappingStore):
        raise MeldrError("The CLI works only with the FileMappingStore.")
    return mr.store


def _store(config: Path | None) -> FileMappingStore:
    return _file_store(load_meldr(config))


def _latest(store: FileMappingStore, spec_id: str, version: int | None = None) -> MappingSpec:
    if version is not None:
        return store.get(spec_id, version)
    history = store.history(spec_id)
    if not history:
        raise MeldrError(f"No spec '{spec_id}'. See `meldr specs list`.")
    return store.get(spec_id, history[-1])


def _alternatives(spec: MappingSpec, target: str, chosen: list[str]) -> str:
    others = [a for a in spec.candidates.get(target, []) if a.path not in chosen][:2]
    return ", ".join(f"{a.path} ({a.score:.2f})" for a in others)


def spec_table(spec: MappingSpec) -> Table:
    table = Table(
        "target",
        "sources",
        "steps",
        "confidence",
        "status",
        "review reason",
        "alternatives",
        title=f"{spec.spec_id} v{spec.version} ({spec.status})",
    )
    decided = [f for f in spec.fields if f.status != FieldStatus.PROPOSED]
    for f in [*spec.pending_fields(), *decided]:
        color = {"approved": "green", "proposed": "yellow", "rejected": "red"}[str(f.status)]
        table.add_row(
            f.target,
            ", ".join(f.sources),
            " → ".join(s.op for s in f.steps) or "copy",
            f"{f.confidence:.2f}",
            f"[{color}]{f.status}[/{color}] ({f.origin})",
            f.review_reason or "",
            _alternatives(spec, f.target, f.sources),
        )
    for t in spec.unmapped_targets:
        table.add_row(t, "", "", "", "[dim]unmapped[/dim]", "", _alternatives(spec, t, []))
    return table


def _edit_in_editor(spec: MappingSpec, target: str) -> str:
    current = spec.mapping_for(target)
    body = (
        current.model_dump(mode="json", include={"sources", "steps"})
        if current
        else {"sources": [], "steps": []}
    )
    editor = shlex.split(os.environ.get("EDITOR", "vi"))
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as fh:
        fh.write(f"# Edit the mapping for {target}; save and close to apply.\n")
        fh.write(yaml.safe_dump(body, sort_keys=False))
        path = Path(fh.name)
    try:
        subprocess.run([*editor, str(path)], check=True)  # noqa: S603
        return path.read_text()
    finally:
        path.unlink(missing_ok=True)


@cli_errors
def review_cmd(
    spec_id: str = typer.Argument(..., help="Spec id, for example erp_db.Organization"),
    approve_above: float | None = typer.Option(None, help="Approve proposed fields ≥ this."),
    approve: list[str] = typer.Option([], "--approve", help="Approve a target (repeatable)."),
    reject: list[str] = typer.Option([], "--reject", help="Reject a target (repeatable)."),
    edit: str | None = typer.Option(None, "--edit", help="Edit one target in $EDITOR."),
    edit_yaml: str | None = typer.Option(None, "--set", help="Inline YAML for --edit."),
    user: str | None = typer.Option(None, "--user", help="Reviewer (default: OS user)."),
    config: Path | None = ConfigOpt,
) -> None:
    """Show a spec and approve, reject, or edit its fields. Every change is a new version."""
    mr = load_meldr(config)
    store = _file_store(mr)
    spec = _latest(store, spec_id)
    who = user or getpass.getuser()
    changed = spec
    if approve_above is not None or approve:
        changed = review.approve(changed, user=who, targets=approve or None, above=approve_above)
    for t in reject:
        changed = review.reject(changed, t, user=who)
    if edit:
        text = edit_yaml if edit_yaml is not None else _edit_in_editor(changed, edit)
        changed = review.edit(changed, edit, text, user=who, ops=mr.ops)
    if changed is not spec:
        changed = store.save(changed, actor=who, action="review")
        console.print(f"Saved {changed.spec_id} v{changed.version} ({changed.status}).")
    console.print(spec_table(changed))


@specs_app.command("list")
@cli_errors
def specs_list(
    source: str | None = typer.Option(None, "--source", help="Filter by source id."),
    config: Path | None = ConfigOpt,
) -> None:
    """Latest version and status of every spec."""
    table = Table("spec_id", "version", "status")
    for spec_id, version, status in _store(config).list(source):
        table.add_row(spec_id, str(version), str(status))
    console.print(table)


@specs_app.command("show")
@cli_errors
def specs_show(
    spec_id: str,
    version: int | None = typer.Option(None, "--version", "-v"),
    as_yaml: bool = typer.Option(False, "--yaml", help="Print the YAML file."),
    config: Path | None = ConfigOpt,
) -> None:
    """Show one spec version (latest by default)."""
    spec = _latest(_store(config), spec_id, version)
    console.print(spec.to_yaml() if as_yaml else spec_table(spec))


@specs_app.command("export")
@cli_errors
def specs_export(
    spec_id: str,
    version: int | None = typer.Option(None, "--version", "-v"),
    fmt: str = typer.Option("sssom", "--format", help="sssom (SSSOM/TSV mapping set)."),
    include_proposed: bool = typer.Option(False, help="Also export fields pending review."),
    config: Path | None = ConfigOpt,
) -> None:
    """Export a spec for exchange with other tools (the YAML spec stays the source of truth)."""
    if fmt != "sssom":
        raise MeldrError(f"Unknown format '{fmt}'. Available: sssom.")
    spec = _latest(_store(config), spec_id, version)
    typer.echo(to_sssom(spec, include_proposed=include_proposed), nl=False)


@specs_app.command("diff")
@cli_errors
def specs_diff(
    spec_id: str,
    from_version: int = typer.Option(..., "--from"),
    to_version: int = typer.Option(..., "--to"),
    config: Path | None = ConfigOpt,
) -> None:
    """Field-level changes between two versions."""
    store = _store(config)
    changes = drift.spec_diff(store.get(spec_id, from_version), store.get(spec_id, to_version))
    if not changes:
        console.print("No field changes.")
    for c in changes:
        console.print(json.dumps(c))


@specs_app.command("refresh-values")
@cli_errors
def specs_refresh_values(
    spec_id: str,
    source: str | None = SourceOpt,
    file: Path | None = FileOpt,
    source_id: str | None = SourceIdOpt,
    table: str | None = TableOpt,
    param: list[str] = ParamOpt,
    config: Path | None = ConfigOpt,
) -> None:
    """Propose map_values additions for categorical values the spec has not seen."""
    mr = load_meldr(config)
    store = _file_store(mr)
    spec = _latest(store, spec_id)
    batch = first_batch(mr, source, file, source_id or spec.source_id, parse_params(param, table))
    updated, unseen = drift.refresh_values(
        spec, mr.to_table(batch), mr.registry.get(spec.target_entity)
    )
    if updated is None:
        console.print("No new categorical values.")
        return
    saved = store.save(updated, action="refresh-values")
    for t, values in unseen.items():
        console.print(f"{t}: {len(values)} new value(s)")
    console.print(f"Saved {saved.spec_id} v{saved.version}; review it with `meldr review`.")
