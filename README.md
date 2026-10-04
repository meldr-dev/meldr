# Meldr

[![PyPI version](https://img.shields.io/pypi/v/meldr.svg?color=blue)](https://pypi.org/project/meldr/)
[![Python versions](https://img.shields.io/pypi/pyversions/meldr.svg)](https://pypi.org/project/meldr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Harmonize data from all connector types (SQL, REST APIs, files, documents, media) into canonical entities.

An LLM (Claude, OpenAI, or a self-hosted OpenAI-compatible model) **proposes** mappings one time for each source shape. Meldr saves the approved mappings as versioned YAML and **runs them deterministically** with Polars. Meldr validates the results, quarantines the bad ones, stores the results with lineage, and indexes them for search. When you run a known source again, Meldr makes zero LLM calls.

Start with the [concepts guide](docs/concepts.md) for the main ideas and terms.

## Black-box use

Use `Meldr.harmonize()` when your application already fetches and stores data.
Meldr does not fetch data or write result rows in this path.
It returns correct rows, quarantine rows, review items, child rows, and relations.

```python
def harmonize_rows(mr, rows, download):
    return mr.harmonize(
        rows,
        target="Organization",
        documents={"contract_url": "Project"},
        texts={"notes": "Event"},
        resolve=download,  # your function: link -> bytes
    )
```

The `resolve` function belongs to your application.
Meldr calls it only for declared link columns.
Use `result.children` for records from text or documents.
Use `result.relations` for approved relationship rows.
Save these frames in your own data store.

## Install

```bash
pip install meldr
pip install 'meldr[anthropic]'        # add one optional provider
pip install 'meldr[sql,s3]'            # add SQL and S3 support
pip install 'meldr[all]'              # common provider, SQL, and document extras
```

Python 3.10+ is necessary.

## Quickstart

```python
import polars as pl
from meldr import Meldr, MeldrConfig, RawBatch

config = MeldrConfig.from_dict(
    {
        "storage": {"root": ".meldr"},
        "llm": {"providers": {"fake": {"type": "fake"}}},  # offline stand-in; use anthropic/openai
    }
)
mr = Meldr(config)

vendors = pl.DataFrame(
    {
        "VNDR_ID": ["V1", "V2"],
        "VNDR_NM": ["Acme Corp", "Beta LLC"],
        "VNDR_TYP_CD": ["S", "V"],
        "REV_CENTS": [100000, 250000],
        "EST_DT": ["03/14/2001", "12/25/1999"],
    }
)
result = mr.meld(RawBatch.from_polars(vendors, "erp_db"), target="Organization")

print(result.data)  # canonical rows + _meldr_* lineage columns
print(result.report.llm_calls)  # 0 on the next run of the same source shape
for field in result.pending_review:  # low-confidence fields wait for a human
    print(field.target, field.sources, field.confidence, field.review_reason)
```

Meldr writes the spec to `.meldr/specs/erp_db/Organization/v0001.yaml`. Commit it and review it like code.

### Embedding in another pipeline

If your pipeline already has rows and its own storage, use the two plain functions. They work without a config, a spec store, or sinks:

```python
from meldr import SchemaRegistry, apply_mapping, pack_path, propose_mapping
from meldr.mapping import review

registry = SchemaRegistry()
registry.load_dir(pack_path("generic"))
org = registry.get("Organization")

spec = propose_mapping(vendors, org, source_id="erp_db", llm=mr.gateway("fake"))  # None: heuristics
spec = review.approve(spec, user="ana", above=0.85)  # or load an approved spec from YAML
out = apply_mapping(vendors, spec, org)
out.data, out.quarantine, out.values_lost  # canonical rows, rejected rows with reasons, counts
```

`_meldr_record_id` is a UUIDv5 of the primary-key values, so it is deterministic and stable across re-runs. It uses the scheme of datagrid's `generateRecordId`. To get matching IDs for string keys, pass a datagrid table stableId (a UUID) as `record_namespace`. Numeric and date keys can serialize differently.

Documents follow the same path through `mr.extract(RawBatch.from_files([...], "docs"), "Project")`. For each extracted value, the model gives a quote that supports the value. Meldr checks the quote against the source text. If the quote is missing or is not in the source text, Meldr halves the confidence of the value and adds a warning. `mr.search("electrical subcontractors", entity="Organization")` returns hits with their lineage.

## How matching works

Matching uses the retrieve-then-rerank design of ReMatch, Magneto, Matchmaker, and SCHEMORA. Candidate retrieval runs first, and LLM reranking runs next:

1. **Candidates** get a score from these signals: name (abbreviation-aware, with a penalty when the names share no word), type, value patterns, vocabulary, embeddings (optional), and **instance evidence**. Instance evidence compares source values with the `examples` of a target field. This signal separates look-alike names. For example, `total_changes` can contain counts or budget dollars. The spec keeps the top 5 candidates for each target, so reviewers see the alternatives.
2. **The LLM decides** among the candidates. It chooses the sources and the transform steps. Set `mapping.self_consistency: 3` to send the request three times, each time with the fields in a different order. Meldr then takes a majority vote, and the level of agreement scales the confidence.
3. **Dry run and self-correction** use sample rows. Then **auto-approval** occurs only if the confidence, dry-run quality, PII, ambiguity, and heuristic agreement checks all pass. If two targets use the same source column, they go to review, unless they split the column (`split_part`, `regex_extract`).
4. **Meldr learns from approved mappings** (`mapping.learn_from_approved`). When a human approves a column name for a target, that column name becomes a top candidate for every future source.

Canonical entities load from YAML, Pydantic models, JSON Schema (`from_json_schema` / `to_json_schema`), or **LinkML** (`CanonicalEntity.from_linkml(path, "Person")`, or `schemas.linkml: [schema.yaml]` in the config). Fields can declare `vocabulary`, `pattern`, `minimum`/`maximum`, `unit`, `pii`, and `examples`.

Transforms are a closed set of 24 deterministic ops (list them with `meldr.default_registry().describe_for_llm()`):
- Parsing: dates, datetimes, numbers, booleans, and Unix or Excel serial timestamps (`from_epoch`).
- Arithmetic and units: linear formulas (`scale` with `factor` and `offset`), `round`, and unit conversion.
- Strings: trim, case, `pad`, regex extract/replace, split, concat, and template.
- Codes and flags: vocabulary maps (`map_values`), keyword rules (`map_contains`), and sentinel nulls.

You can export approved specs to the **SSSOM** mapping standard: `meldr specs export erp_db.Organization --format sssom`.

## Benchmarks

```bash
meldr eval --variants 10 --providers heuristic,fake                    # synthetic (bundled)
git clone --depth 1 https://github.com/delftdata/valentine /tmp/valentine  # real NYC Open Data pairs
meldr eval --suite tables --path /tmp/valentine/experiments/data --providers heuristic,claude
```

Heuristic-only results (no LLM) on the 10 Valentine/NYU table pairs (77 true matches, some with deliberate name traps): precision 0.750, recall 0.623, F1 0.681, HitRate@1 0.792, HitRate@3 0.935, HitRate@5 0.987, MRR 0.870. HitRate@5 means that the correct column is one of the 5 candidates that go to the LLM. CI fails if F1, HitRate@1, or MRR drops by more than 0.02. Run it with your LLM provider to measure end-to-end accuracy.

Because confidence scores rank mappings but are not probabilities, do not guess `auto_approve_threshold`. Every eval run prints the lowest threshold that has a precision of at least `--target-precision` (default 0.97) at 95% confidence. It also prints how much work that threshold auto-approves.

When there are no errors, at least 125 labeled matches are necessary to prove 97%. Until your benchmark is that large, the report tells you to continue to review everything. The default `auto_approve_threshold` (0.90) is not calibrated. Set it from this report.

## Configuration

`Meldr.from_config("meldr.yaml")` loads a YAML file. Credential fields (`api_key`, `url`, `token`, `password`, `client_secret`) must be `${env:…}` or `${vault:…}` references. Meldr rejects literal values. Meldr resolves the references in REST `headers`/`params` and in sink `options` when it builds the connector or sink.

```yaml
storage: {root: ./.meldr}
llm:
  providers:
    claude: {type: anthropic, model: <model>, api_key: "${env:ANTHROPIC_API_KEY}"}
    local:  {type: openai_compatible, base_url: http://localhost:8000/v1, model: <model>,
             data_policy: [schema, masked_samples, samples, content]}
  routing: {mapping: claude, extraction: local}
mapping: {auto_approve_threshold: 0.90, on_breaking_drift: block}
sources:
  erp_db: {connector: sql, url: "${vault:secret/data/erp#readonly_url}", data_policy: [schema]}
  vendors_api:
    connector: rest
    base_url: https://api.example.com/v2
    endpoint: vendors
    auth: {type: bearer, token: "${env:VENDORS_API_TOKEN}"}
    pagination: {type: cursor, next_path: meta.next_cursor}
    records_path: data
sinks: {tables: {type: duckdb}, vectors: {type: lancedb}}
```

Canonical entities are YAML files (see `src/meldr/domains/generic`: Organization, Person, Project, Event) or Pydantic models that you pass to `CanonicalEntity.from_pydantic`.

## Storage: bring your own database

Meldr is a library, so you choose where results go. Pick one table sink and, optionally, one vector sink. You can use each table sink with each vector sink:

| Kind | Name | Install extra | Status |
|---|---|---|---|
| Table sink | `duckdb`, `parquet` | core | tested in CI |
| Table sink | `sql` with SQLite | `meldr[sql]` | tested in CI |
| Table sink | `sql` with Snowflake, BigQuery, or Databricks | `meldr[sql,snowflake]`, `meldr[sql,bigquery]`, or `meldr[sql,databricks]` | same code path |
| Table sink | `delta`, `iceberg` | `meldr[delta]`, `meldr[iceberg]` | tested in CI |
| Vector sink | `memory`, `chroma` | `meldr[chromadb]` for Chroma | tested in CI |
| Vector sink | `lancedb`, `qdrant`, `pgvector` | `meldr[lancedb]`, `meldr[qdrant]`, or `meldr[pgvector]` | tested in CI |
| Vector sink | Pinecone, Weaviate, OpenSearch, Milvus, Elasticsearch | your plugin package | plugin |
| Connector | files with local paths or fsspec URIs | `meldr[fsspec]`, `meldr[s3]`, `meldr[gcs]`, `meldr[azure]`, or `meldr[sftp]` | tested in CI |
| Connector | `sql` with SQLAlchemy | `meldr[sql]` | tested in CI |
| Connector | `rest`, `python` | core | tested in CI |
| Connector | `dlt` with a fake source test | `meldr[dlt]` | same code path |

```yaml
sinks:
  tables: {type: sql, url: "${env:WAREHOUSE_URL}", namespace: harmonized}   # or:
  # tables: {type: delta, path: s3://lake/meldr, options: {AWS_REGION: us-east-1}}
  # tables: {type: iceberg, namespace: meldr, options: {catalog: prod, uri: "${env:CATALOG_URI}"}}
  vectors: {type: pgvector, url: "${env:PG_URL}"}
```

Every table sink upserts by `(_meldr_source_id, _meldr_record_id)`, lets a newer `_meldr_sequence` win, adds new columns, and passes the same conformance suite (`tests/unit/test_table_sinks.py`). To add a different store, use `meldr.register_sink("mydb", factory, kind="table")` or a `meldr.sinks` entry point. Implement `write(entity, df, *, key)`, `read(entity)`, and `delete(entity, keys)`.

Cloud stores and warehouses use the same code path. CI does not use live cloud services.

If your store keeps column comments, also implement `describe(...)`. Run `meldr.check_table_sink()` or `meldr.check_vector_sink()` before you publish a plugin. Read the [extension guide](docs/extending.md) for connector and sink examples. Pass `table_sink=` / `vector_sink=` objects to `Meldr` directly, or skip sinks entirely with `apply_mapping()`.

## CLI

```bash
meldr profile --source erp_db --table vendors
meldr map     --target Organization --source erp_db --table vendors
meldr review  erp_db.Organization --approve-above 0.85     # or --approve/--reject/--edit TARGET
meldr run     --target Organization --source erp_db --table vendors
meldr specs   list | show | diff --from 2 --to 3 | refresh-values
meldr extract --target Project --file ./docs
meldr search  "electrical subcontractors in Colorado" --entity Organization
meldr eval    --variants 10 --providers heuristic,fake,claude
meldr -v run  ...                                          # JSON logs on stderr (configure_logging)
```

## Guarantees

- **Meldr does not run model output as code.** Transforms come only from the whitelisted `TransformRegistry`, and LLM-supplied regexes run only in Polars' linear-time engine.
- **The gateway enforces data policy before each I/O operation.** By default, vendor providers get the schema and masked samples. A vendor provider gets document content only after an explicit opt-in. If a request contains a data class outside the policy, the request fails with `PolicyViolation`.
- **Re-runs converge, and the run order does not change the result.** Record IDs come from the primary key, and Meldr upserts every table by key. `<Entity>` keeps each record's latest valid version, so a bad update does not replace good data. `<Entity>__quarantine` keeps a failure only until a newer valid version arrives. If you declare `sequence_field` (for example `updated_at`) on an entity, a replay of older extracts cannot replace newer data. Each spec pins its transforms version, and Meldr publishes spec versions atomically and does not overwrite them.
- **Bad rows do not crash a run.** They go to `<Entity>__quarantine` with their failure causes, and Meldr counts the values lost in transformation. If a value in a row fails to transform, Meldr quarantines the row and does not write the row with a null. This rule also applies when the mapping of a field fails completely.
- **Meldr reads source databases politely.** The SQL connector runs one SELECT with the tag `/* meldr source=<id> */`, so you can identify it in `pg_stat_activity` and slow-query logs. On Postgres, MySQL/MariaDB, and SQLite, the SELECT runs inside a read-only transaction, and on other databases your role must enforce read-only access. The connector stages the results to local Parquet and closes the connection before mapping, LLM, embedding, or write work starts. As a result, DBA-set `idle_in_transaction_session_timeout` limits do not stop a run early. Even so, give the connector a read-only role.
- **Harmonized tables explain themselves.** Meldr stores the description, unit, vocabulary, and PII flag of each canonical field as column comments (DuckDB, SQL dialects that support comments, Delta column metadata, Iceberg column docs). This lets analysts and text-to-SQL agents read the table correctly. `DuckDBSink.sql()` (`meldr.indexing.sinks.duckdb`) accepts only one read-only query, and the connection cannot read other files or attach databases.
- **Memory stays bounded.** CSVs stream in `execution.batch_size` batches. Meldr reads CSV values as strings, so it keeps leading zeros. Parquet streams by row group, and JSONL streams line by line. The REST connector fetches pages as you iterate and combines them into batches. Indexing embeds 2,048 rows at a time.
- **Re-runs are cheap.** A source shape with an approved spec skips profiling and the LLM entirely. On a laptop, a 50,000 × 207 batch maps and validates in about 0.1 s.
- **Logs contain field names, counts, and IDs only.**
- **Heavy dependencies are optional extras.** If an extra is missing, you get `MissingExtraError` with the install command.

## Security

### SQL sources

The SQL connector accepts only one SELECT or WITH statement. A keyword filter stops statements that change data. This filter is not a full security boundary.

On Postgres, MySQL, MariaDB, and SQLite, the query also runs in a read-only transaction. On Snowflake, BigQuery, Databricks, SQL Server, and Oracle, only the keyword filter applies. A SELECT can call a function that changes data.

Give each SQL source a database user that can only read. Put the URL of that user in a secret reference, as the example in the configuration section shows.

### Known advisories in optional extras

The core install does not contain these packages.

- **`meldr[chromadb]`:** `chromadb` 1.5.9 has advisories CVE-2026-45829, CVE-2026-45830, CVE-2026-45831, and CVE-2026-45833. The problems are in the HTTP API of the Chroma server. The Meldr Chroma sink uses only the in-process client, so Meldr does not use that API. Do not open a Chroma server to a network that you do not trust. At this time, no fixed version of `chromadb` is available.
- **`meldr[databricks]`:** `databricks-sql-connector` 4.6.0 accepts only `oauthlib` versions before 4.0.0. `oauthlib` 3.3.1 has advisory CVE-2026-49265, a time side-channel problem in the PKCE check of an OAuth server. Meldr is not an OAuth server, so this problem does not apply to Meldr. Without the `databricks` extra, pip installs `oauthlib` 4.0.0, which has the fix.

## Development

```bash
uv sync --all-extras
uv run ruff format --check . && uv run ruff check . && uv run mypy src && uv run pytest -m "not live_llm"
```
