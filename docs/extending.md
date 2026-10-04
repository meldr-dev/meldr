# Extend Meldr

Meldr uses small protocols for connectors and sinks. You can add a system without a change to
the core library. Use a Python entry point to load your extension from an installed package.

## Write a connector

A connector has a source ID, a modality, and a `pull()` method. The method yields `RawBatch`
objects. The `python` connector can wrap a callable or an iterable when no package is necessary.

```python
from collections.abc import Iterator
from typing import Any

from meldr import Connector, RawBatch


class MyConnector:
    modality = "tabular"

    def __init__(self, source_id: str, rows: list[dict[str, Any]]) -> None:
        self.source_id = source_id
        self.rows = rows

    def pull(self, **params: Any) -> Iterator[RawBatch]:
        yield RawBatch.from_dicts(self.rows, self.source_id)
```

Add a factory that reads `SourceConfig`:

```python
def from_config(source_id: str, cfg: Any, batch_size: int) -> MyConnector:
    rows = cfg.options["rows"]
    return MyConnector(source_id, rows)
```

Declare the entry point in `pyproject.toml`:

```toml
[project.entry-points."meldr.connectors"]
my_source = "my_package.connector:from_config"
```

The factory must return a connector. It must not make a network call during construction.

## Write a table sink

A table sink has `write()`, `read()`, and `delete()` methods. `write()` upserts by the key. A
newer `_meldr_sequence` must win. The sink must reject unsafe table names.

```python
from pathlib import Path
from typing import Any

from meldr import TableSink, register_sink


def make_table_sink(config: Any, root: Path) -> TableSink:
    return MyTableSink(config.url, root)


register_sink("my_table_store", make_table_sink, kind="table")
```

Use `check_table_sink()` before you publish the package:

```python
from meldr import check_table_sink

check_table_sink(lambda root: MyTableSink("test", root))
```

## Write a vector sink

A vector sink has `upsert()` and `search()` methods. It must support a collection name, a vector,
`k`, and scalar filters. A list filter means one of the listed values.

```python
from meldr import VectorSink, check_vector_sink, register_sink


def make_vector_sink(config: Any, root: Path) -> VectorSink:
    return MyVectorSink(config.url)


register_sink("my_vector_store", make_vector_sink, kind="vector")
check_vector_sink(lambda root: MyVectorSink("test"))
```

The factory receives the configured sink section and the Meldr storage root. Resolve secret
references inside the sink. Do not log the resolved value.

## Use a plugin entry point

Sink entry-point names use `table.` or `vector.`. A name without a prefix is a table sink for
compatibility with the old registration form.

```toml
[project.entry-points."meldr.sinks"]
"table.my_table_store" = "my_package.sinks:make_table_sink"
"vector.my_vector_store" = "my_package.sinks:make_vector_sink"
```

The package user installs the plugin. Meldr loads it when it builds a `Meldr`.
