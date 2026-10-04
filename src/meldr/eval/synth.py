"""Synthetic benchmark: canonical rows -> perturbed source variants with ground truth (§4.17).

It uses a seeded, dependency-free generator (see DECISIONS.md), so `meldr eval` works on a
core install.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import polars as pl

from meldr._compat import UTC
from meldr.core.canonical import CanonicalEntity, CanonicalField
from meldr.core.dtypes import DataType, to_polars
from meldr.core.mapping import TransformStep
from meldr.mapping.normalize import tokens

_COMPANY_A = (
    "Acme Globex Initech Umbrella Stark Wayne Hooli Vandelay Soylent Tyrell Cyberdyne Wonka "
    "Gringotts Oscorp Aperture"
).split()
_COMPANY_B = (
    "Builders Electric Holdings Partners Supply Concrete Steel Mechanical Roofing Logistics "
    "Engineering Plumbing"
).split()
_COMPANY_C = "Inc LLC Ltd Corp Co GmbH".split()
_FIRST = (
    "Ana Ben Chen Dara Eli Fatima Goran Hana Ivan Jia Kofi Lena Mateo Nia Omar Priya Quinn Rosa "
    "Sven Tariq"
).split()
_LAST = (
    "Garcia Smith Nguyen Okafor Kowalski Rossi Tanaka Silva Novak Haddad Larsen Mehta Dubois Kim "
    "Moreau"
).split()
_CITIES = "Denver Austin Boston Seattle Chicago Toronto Leeds Lyon Porto Munich Osaka Perth".split()
_COUNTRIES = "US CA GB FR PT DE JP AU".split()
_JOBS = (
    "Project Manager,Site Engineer,Estimator,Superintendent,Architect,Safety Officer,Buyer"
).split(",")
_WORDS = "North Tower Bridge Harbor Plaza Campus Station Depot Mill Clinic Library Arena".split()

ABBREV = {
    "name": "nm",
    "number": "no",
    "date": "dt",
    "amount": "amt",
    "type": "typ",
    "code": "cd",
    "revenue": "rev",
    "organization": "org",
    "customer": "cust",
    "vendor": "vndr",
    "project": "proj",
    "description": "desc",
    "address": "addr",
    "quantity": "qty",
    "identifier": "id",
    "established": "est",
    "status": "stat",
    "active": "actv",
    "phone": "tel",
    "country": "ctry",
    "city": "cty",
    "company": "co",
    "account": "acct",
}
DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d.%m.%Y", "%Y%m%d", "%b %d %Y"]
NOISE = ["LAST_UPD_USR", "ROW_VER", "etl_batch_id", "created_ts", "legacy_flag"]


@dataclass
class Variant:
    """One synthetic source with its ground truth."""

    name: str
    entity: CanonicalEntity
    source: pl.DataFrame
    truth: dict[str, list[str]]
    truth_steps: dict[str, list[TransformStep]] = field(default_factory=dict)
    expected: pl.DataFrame = field(default_factory=pl.DataFrame)
    perturbations: list[str] = field(default_factory=list)


def _value(f: CanonicalField, entity: str, i: int, rng: random.Random) -> Any:
    toks = set(tokens(f.name))
    if f.vocabulary:
        return rng.choice(f.vocabulary)
    if f.dtype == DataType.DATE:
        return date(1970, 1, 1) + timedelta(days=rng.randint(0, 20_000))
    if f.dtype == DataType.DATETIME:
        return datetime(2020, 1, 1, tzinfo=UTC) + timedelta(seconds=rng.randint(0, 10**8))
    if f.dtype in (DataType.FLOAT, DataType.DECIMAL):
        return round(rng.uniform(1_000, 5_000_000), 2)
    if f.dtype == DataType.INTEGER:
        return rng.randint(1, 10_000)
    if f.dtype == DataType.BOOLEAN:
        return rng.random() < 0.7
    if "identifier" in toks or f.name.endswith("_id"):
        return f"{entity[:3].upper()}{i:05d}"
    if "email" in toks:
        return f"{rng.choice(_FIRST).lower()}.{rng.choice(_LAST).lower()}{i}@example.com"
    if toks & {"phone", "mobile", "telephone"}:
        return f"+1{rng.randint(200, 999)}{rng.randint(1000000, 9999999)}"
    if "city" in toks or "location" in toks:
        return rng.choice(_CITIES)
    if "country" in toks:
        return rng.choice(_COUNTRIES)
    if toks & {"job", "title"} and entity == "Person":
        return rng.choice(_JOBS)
    if toks & {"full"} or (entity == "Person" and "name" in toks):
        return f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
    if toks & {"organization", "client", "contractor", "company", "legal"} or (
        entity == "Organization" and "name" in toks
    ):
        return f"{rng.choice(_COMPANY_A)} {rng.choice(_COMPANY_B)} {rng.choice(_COMPANY_C)}"
    return f"{rng.choice(_WORDS)} {rng.choice(_WORDS)} {i}"


def canonical_rows(entity: CanonicalEntity, n: int, seed: int) -> pl.DataFrame:
    """Typed canonical rows (5% nulls in optional fields)."""
    rng = random.Random(seed)
    cols: dict[str, list[Any]] = {f.name: [] for f in entity.fields}
    for i in range(n):
        for f in entity.fields:
            null = not f.required and f.name not in entity.primary_key and rng.random() < 0.05
            cols[f.name].append(None if null else _value(f, entity.name, i, rng))
    return pl.DataFrame(cols, schema={f.name: to_polars(f.dtype) for f in entity.fields})


def _case(words: list[str], style: str) -> str:
    if style == "screaming":
        return "_".join(w.upper() for w in words)
    if style == "camel":
        return words[0].lower() + "".join(w.capitalize() for w in words[1:])
    if style == "title":
        return " ".join(w.capitalize() for w in words)
    return "_".join(w.lower() for w in words)


def _rename(f: CanonicalField, rng: random.Random, perturb: list[str]) -> str:
    choice = rng.random()
    base = f.name
    if f.synonyms and choice < 0.45:
        base = rng.choice(f.synonyms)
        perturb.append(f"synonym:{f.name}")
    words = list(tokens(base)) or [base]
    if rng.random() < 0.4:
        words = [ABBREV.get(w, w) for w in words]
        perturb.append(f"abbrev:{f.name}")
    style = rng.choice(["snake", "screaming", "camel", "title"])
    name = _case(words, style)
    if rng.random() < 0.15:
        name = rng.choice(["tbl_", "src_"]) + name
    return name


def _render(
    f: CanonicalField, col: pl.Series, rng: random.Random, name: str, perturb: list[str]
) -> tuple[str, pl.Series, list[TransformStep]]:
    """Encode canonical values in a source format and return (name, series, truth steps)."""
    values = col.to_list()
    if f.vocabulary:
        style = rng.choice(["same", "codes", "title"])
        if style == "codes":
            perturb.append(f"codes:{f.name}")
            codes = {v: v[:3].upper() for v in f.vocabulary}
            mapping = {c: v for v, c in codes.items()}
            out = [None if v is None else codes[v] for v in values]
            step = TransformStep(op="map_values", args={"mapping": mapping, "default": None})
            return name, pl.Series(name, out, dtype=pl.String), [step]
        if style == "title":
            out = [None if v is None else v.replace("_", " ").title() for v in values]
            mapping = {v.replace("_", " ").title(): v for v in f.vocabulary}
            step = TransformStep(op="map_values", args={"mapping": mapping, "default": None})
            return name, pl.Series(name, out, dtype=pl.String), [step]
        return name, pl.Series(name, values, dtype=pl.String), []
    if f.dtype == DataType.DATE:
        fmt = rng.choice(DATE_FORMATS)
        perturb.append(f"date_format:{f.name}:{fmt}")
        out = [None if v is None else v.strftime(fmt) for v in values]
        return (
            name,
            pl.Series(name, out, dtype=pl.String),
            [TransformStep(op="parse_date", args={"formats": [fmt]})],
        )
    if f.dtype == DataType.DATETIME:
        fmt = rng.choice(["%Y-%m-%dT%H:%M:%S%#z", "%Y-%m-%d %H:%M:%S"])
        out = [None if v is None else v.strftime(fmt.replace("%#z", "Z")) for v in values]
        return (
            name,
            pl.Series(name, out, dtype=pl.String),
            [TransformStep(op="parse_datetime", args={"formats": [fmt]})],
        )
    if f.dtype in (DataType.FLOAT, DataType.DECIMAL):
        style = rng.choice(["plain", "scaled", "formatted"])
        if style == "scaled" and f.unit == "USD":
            perturb.append(f"cents:{f.name}")
            out = [None if v is None else round(v * 100) for v in values]
            return (
                f"{name}_cents",
                pl.Series(f"{name}_cents", out, dtype=pl.Int64),
                [TransformStep(op="scale", args={"factor": 0.01})],
            )
        if style == "scaled" and f.unit == "m2":
            perturb.append(f"sqft:{f.name}")
            out = [None if v is None else v / 0.09290304 for v in values]
            return (
                f"{name}_sqft",
                pl.Series(f"{name}_sqft", out, dtype=pl.Float64),
                [TransformStep(op="unit_convert", args={"from_unit": "ft2", "to_unit": "m2"})],
            )
        if style == "formatted":
            perturb.append(f"number_format:{f.name}")
            sym = "$" if f.unit == "USD" else ""
            out = [None if v is None else f"{sym}{v:,.2f}" for v in values]
            return name, pl.Series(name, out, dtype=pl.String), [TransformStep(op="parse_number")]
        return name, pl.Series(name, values, dtype=pl.Float64), []
    if f.dtype == DataType.BOOLEAN:
        t, fl = rng.choice([("Y", "N"), ("yes", "no"), ("true", "false")])
        out = [None if v is None else (t if v else fl) for v in values]
        return name, pl.Series(name, out, dtype=pl.String), [TransformStep(op="parse_bool")]
    return name, pl.Series(name, values, dtype=to_polars(f.dtype)), []


def make_variant(entity: CanonicalEntity, canonical: pl.DataFrame, seed: int, idx: int) -> Variant:
    rng = random.Random(seed * 1000 + idx)
    perturb: list[str] = []
    optional = [f for f in entity.fields if not f.required and f.name not in entity.primary_key]
    dropped = set(rng.sample([f.name for f in optional], k=min(len(optional), rng.randint(0, 2))))
    columns: list[pl.Series] = []
    truth: dict[str, list[str]] = {}
    steps: dict[str, list[TransformStep]] = {}
    used: set[str] = set()
    for f in entity.fields:
        if f.name in dropped:
            continue
        name = _rename(f, rng, perturb)
        while name.casefold() in used:
            name += "_2"
        col_name, series, truth_steps = _render(f, canonical[f.name], rng, name, perturb)
        used.add(col_name.casefold())
        columns.append(series.alias(col_name))
        truth[f.name] = [col_name]
        steps[f.name] = truth_steps
    height = canonical.height
    for noise in rng.sample(NOISE, k=rng.randint(1, 3)):
        if noise.casefold() in used:
            continue
        values = ["".join(rng.choices(string.ascii_lowercase, k=6)) for _ in range(height)]
        columns.append(pl.Series(noise, values, dtype=pl.String))
    rng.shuffle(columns)
    return Variant(
        name=f"{entity.name}-v{idx:02d}",
        entity=entity,
        source=pl.DataFrame(columns),
        truth=truth,
        truth_steps=steps,
        expected=canonical.select(list(truth)),
        perturbations=perturb,
    )


def make_variants(
    entity: CanonicalEntity, n: int, *, rows: int = 200, seed: int = 7
) -> list[Variant]:
    """`n` perturbed source variants for `entity` (deterministic for a given seed)."""
    canonical = canonical_rows(entity, rows, seed)
    return [make_variant(entity, canonical, seed, i) for i in range(n)]
