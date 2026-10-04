# Meldr concepts

This guide explains the main idea behind Meldr.
It starts with one row.
It then shows the full data path.

## The problem

Different systems describe the same thing in different ways.

One system can call a company `VNDR_NM`.
One more system can call it `supplier_name`.
A third system can call it `company`.

The values can also have different forms.
One system can store money in cents.
A different system can store money in dollars.
One system can write a date as `03/14/2001`.
A different system can write it as `2001-03-14`.

People want one clear record.
Meldr maps the different forms to that record.

## One row from start to finish

This example has one source row.

```text
Source row
  VNDR_ID  = V1
  VNDR_NM  = Acme Corp
  REV_CENTS = 100000

Approved mapping
  VNDR_ID           -> org_id
  VNDR_NM           -> legal_name
  REV_CENTS / 100   -> annual_revenue

Output row
  org_id          = V1
  legal_name      = Acme Corp
  annual_revenue  = 1000.00
```

The source keeps its original names.
The output uses the canonical names.
The mapping rule explains each change.

The same input and the same approved spec give the same output.
This makes a repeat run safe to inspect and compare.

## The two times in Meldr

Meldr has a design time and a data time.

### Design time

Meldr studies a source shape.
It reads field names, types, patterns, and samples.
It finds possible target fields.
An LLM can suggest a mapping.
Heuristics can suggest a mapping without an LLM.

A person can review the suggestion.
The person can approve or change fields.
Meldr saves the approved mapping as a versioned `MappingSpec`.

The LLM suggests data.
It does not run code.
The transform registry runs the approved operations.

### Data time

Meldr reads a new batch.
It finds the saved spec for the source shape.
It applies the approved operations with Polars.
It checks each output row against the canonical entity.

Meldr writes correct rows to a table sink.
It sends bad rows to a quarantine table.
It records the cause of each failure.
It keeps lineage with each row.

When the source shape stays the same, Meldr can skip the LLM.
The next run uses the saved spec.

## The main words

### Source

A source is a system that gives data to Meldr.
Examples include a SQL database, a REST API, a file, or a document.

Each source has a source ID.
The source ID helps Meldr find its saved specs and lineage.

### RawBatch

`RawBatch` is the input envelope.
It holds a table or a list of records.
It also holds the source ID and source metadata.

Connectors return `RawBatch` objects.
Your code can make one with `RawBatch.from_polars()` or `RawBatch.from_dicts()`.

### Canonical entity

A canonical entity is the shared shape for one real-world thing.
Examples include `Organization`, `Person`, `Project`, and `Event`.

An entity has fields, types, rules, and a version.
It can also declare a primary key, a sequence field, and PII fields.

The entity says what the output must mean.
It does not say how each source names that meaning.

### MappingSpec

`MappingSpec` is the saved mapping plan.
It names source fields, target fields, and transform steps.
It also stores confidence, review status, source shape, and version data.

Meldr treats a spec as data.
Meldr validates the spec before it runs.
Meldr does not turn model text into Python code.

### Transform

A transform changes a source value into the target form.
Examples include trim, parse number, scale, date parsing, and value maps.

Transforms come from the `TransformRegistry`.
The registry contains only approved operations.
LLM output can select an operation and its typed arguments.
The LLM cannot add a new operation at run time.

### Quarantine

Quarantine holds rows that fail a check.
The row stays separate from the correct output.
The quarantine row keeps a cause and source context.

Bad data does not stop the full run.
You can fix the source or the mapping and run it again.

### Lineage

Lineage tells you where an output value came from.
Meldr stores the source ID, record ID, spec ID, spec version, and run ID.
It also stores the time of ingestion.

Use `Meldr.explain()` to inspect a stored row.
Use the lineage store to inspect run reports.

## The public paths

### A script, notebook, or CLI

Use the `Meldr` facade.
It manages local specs, sinks, caches, and lineage.

```python
import polars as pl

from meldr import Meldr, MeldrConfig, RawBatch

config = MeldrConfig.from_dict(
    {
        "storage": {"root": ".meldr"},
        "llm": {"providers": {"fake": {"type": "fake"}}},
    }
)

vendors = pl.DataFrame(
    {
        "VNDR_ID": ["V1"],
        "VNDR_NM": ["Acme Corp"],
        "REV_CENTS": [100000],
    }
)

with Meldr(config) as mr:
    result = mr.meld(RawBatch.from_polars(vendors, "erp"), "Organization")

print(result.to_dict())
```

The first run can call the fake provider.
The fake provider stays inside the process.
The next run can use the saved spec with zero LLM calls.

### An existing data pipeline

Use the plain functions when your pipeline owns storage.

```python
from meldr import apply_mapping, propose_mapping
from meldr.mapping import review

spec = propose_mapping(rows, entity, source_id="erp", llm=gateway)
approved = review.approve(spec, user="ana", above=0.85)
applied = apply_mapping(rows, approved, entity, run_id="task-123")
```

`propose_mapping()` suggests a spec.
`apply_mapping()` runs an approved spec.
`apply_mapping()` does not call an LLM.
Your pipeline can write `applied.data` to its own database.

### An API and a worker

The API can accept a request and send a task to a queue.
The worker can call Meldr.
The queue owns retries, partitions, and job status.

Use one Meldr instance for each worker process.
Use a shared SQL store when many workers share specs.
Use a shared table sink or a user database for rows.

FastAPI, Django, and Flask can start the task.
Celery, RQ, Arq, or a queue worker can run the task.
Read `docs/integrations.md` for the process lifetime examples.

## The black-box path

Your application fetches the data.
Your application keeps the raw data.
Meldr starts when you give it the data.

Use `harmonize()` for this path.
It returns data to your application.
It does not write rows to a sink by default.
It does not index rows by default.

```python
result = mr.harmonize(
    rows,
    target="Organization",
    documents={"contract_url": "Project"},
    texts={"notes": "Event"},
    resolve=download,
)
```

The `resolve` function belongs to your application.
It changes a link into bytes.
Meldr calls it only for columns that you declare.

For a known source shape, Meldr uses the approved spec.
It skips profiling and proposal work.
It makes zero model calls for the mapping.

For a new source shape, Meldr proposes a spec.
High-confidence fields can pass automatic approval.
Other fields become review items.
Use `mr.apply_review()` to save a reviewed spec version.

Use `result.data` for rows that pass checks.
Use `result.quarantine` for rows that fail checks.
Use `result.children` for records from text and documents.
Use `result.relations` for approved relationship rows.
Your application can save each result in its own data store.

An entity can set `extraction.cardinality` to `many`.
Meldr then returns one child row for each item in the content.
An entity can also list approved relation predicates.
Meldr sends a relation with an unknown predicate to quarantine.

## What happens to a bad row

Meldr does not replace a bad value with a silent null.

The executor counts values lost during a transform.
The validator checks required fields and field limits.
The pipeline sends the failed row to quarantine.
The result reports the failure cause.

This gives the operator three facts:

- The row that failed.
- The field or rule that failed.
- The source value and lineage necessary for review.

## What changes at each size

### One script

Use local file specs and a local sink.
Make one `Meldr` instance.
Close it with a context manager.

### One API service

Make one instance for each process.
Call `ameld()` for a short async request.
Send a long job to a worker.
Move slow store and sink calls away from the API loop.

### Many workers

Use a shared SQL spec store.
Use a shared output sink.
Let the queue split a large source into partitions.
Let each worker apply the same approved spec.

Meldr owns mapping and validation.
The host system owns task status and retry policy.

## A short glossary

- **Source:** the system that gives data to Meldr.
- **Entity:** the shared output shape.
- **Spec:** the saved mapping plan.
- **Transform:** one approved value change.
- **Review:** a human check of a mapping.
- **Quarantine:** a safe place for failed rows.
- **Lineage:** facts that show where a value came from.
- **Sink:** the place that stores output rows.

## The main lesson

Meldr learns how a source maps to a shared model.
A person can approve the mapping.
Meldr then applies the same approved rules to new batches.
It checks the output and records its history.
