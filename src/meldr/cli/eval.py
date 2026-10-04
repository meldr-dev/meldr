"""`meldr eval`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from meldr._async import run_sync
from meldr.cli.common import ConfigOpt, cli_errors, console, load_meldr
from meldr.core.errors import MeldrError
from meldr.eval.metrics import Case
from meldr.eval.runner import (
    GatewayFactory,
    builtin_factories,
    check_baseline,
    run_eval,
    run_table_eval,
)
from meldr.llm.gateway import LLMGateway


def _fixed(gateway: LLMGateway | None) -> GatewayFactory:
    def factory(_case: Case) -> LLMGateway | None:
        return gateway

    return factory


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


@cli_errors
def eval_cmd(
    suite: str = typer.Option("synthetic", help="synthetic, or tables (also set --path)."),
    path: Path | None = typer.Option(
        None, "--path", help="Table-pair benchmark root (Valentine layout) for --suite tables."
    ),
    variants: int = typer.Option(10, help="Variants for each entity."),
    providers: str = typer.Option(
        "heuristic,fake", help="Comma list: heuristic, fake, fake-heuristic, or config providers."
    ),
    entities: str = typer.Option("Organization,Project,Person,Event", help="Comma list."),
    rows: int = typer.Option(200, help="Rows for each variant."),
    seed: int = typer.Option(7),
    out: Path = typer.Option(Path("eval/reports"), help="Directory for the JSON report."),
    check: Path | None = typer.Option(None, "--check-baseline", help="Fail on regression."),
    target_precision: float = typer.Option(
        0.97, help="Precision the recommended auto-approve threshold must guarantee."
    ),
    config: Path | None = ConfigOpt,
) -> None:
    """Compare providers on a benchmark and write a JSON report."""
    if suite not in ("synthetic", "tables") or (suite == "tables" and path is None):
        raise MeldrError("Use --suite synthetic, or --suite tables --path <benchmark dir>.")
    mr = load_meldr(config)
    builtins = builtin_factories()
    factories = {
        name: builtins.get(name) or _fixed(mr.gateway(name))
        for name in (p.strip() for p in providers.split(","))
        if name
    }
    if path is not None and suite == "tables":
        job = run_table_eval(
            path, factories, cfg=mr.config.mapping, target_precision=target_precision
        )
    else:
        ents = [mr.registry.get(e.strip()) for e in entities.split(",") if e.strip()]
        job = run_eval(
            ents,
            factories,
            variants=variants,
            rows=rows,
            seed=seed,
            cfg=mr.config.mapping,
            target_precision=target_precision,
        )
    report = run_sync(job)
    table = Table(
        "provider",
        "precision",
        "recall",
        "F1",
        "Hit@1",
        "Hit@5",
        "MRR",
        "value acc",
        "auto-approve P",
        "auto-approved",
        "LLM calls",
        "tokens in/out",
        "cost USD",
        "seconds",
        title=f"meldr eval · {report.suite} · {report.variants} cases",
    )
    for r in report.results:
        table.add_row(
            r.provider,
            f"{r.precision:.3f}",
            f"{r.recall:.3f}",
            f"{r.f1:.3f}",
            _fmt(r.hit_at_1),
            _fmt(r.hit_at_5),
            _fmt(r.mrr),
            _fmt(r.value_accuracy),
            _fmt(r.auto_approve_precision),
            str(r.auto_approved),
            str(r.llm_calls),
            f"{r.input_tokens}/{r.output_tokens}",
            "-" if r.est_cost_usd is None else f"{r.est_cost_usd:.4f}",
            f"{r.seconds:.1f}",
        )
    console.print(table)
    for provider, cal in report.calibration.items():
        if cal.threshold is None:
            console.print(
                f"{provider}: no threshold reaches {cal.target_precision:.0%} precision with 95% "
                f"confidence on {cal.points} proposals; keep reviewing everything."
            )
        else:
            console.print(
                f"{provider}: auto_approve_threshold {cal.threshold} guarantees precision >= "
                f"{cal.lower_bound:.3f} (95% bound; observed {cal.precision:.3f}) and would "
                f"auto-approve {cal.auto_rate:.0%} of {cal.points} proposals."
            )
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval-{report.created_at.replace(':', '').split('.')[0]}.json"
    path.write_text(report.to_json())
    console.print(f"Report: {path}")
    if check is not None:
        failures = check_baseline(report, check)
        if failures:
            for f in failures:
                console.print(f"[red]regression:[/red] {f}")
            raise typer.Exit(code=1)
        console.print("[green]Baseline check passed.[/green]")
