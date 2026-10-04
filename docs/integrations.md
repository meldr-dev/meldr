# Backend integrations

Meldr works in an API process and in a worker process.

For a short request, an API handler can call `ameld()`.
For a long job, the handler must send a task to a queue.
The worker calls `meld()` or `run()`.
The queue owns retries, task state, and idempotency.

Make one `Meldr` instance for each process.
Do not make one instance for each request.
Close the instance when the process stops.
Use a SQL store and a SQL table sink when many workers share state.
The default file store is for one script or one local worker.

## A shared worker function

This function shows the small unit that a queue task can call.
The task receives a source name and connector parameters.
It does not put a large table in the queue message.

```python
from typing import Any

from meldr import Meldr, MeldrResult

_meldr: Meldr | None = None


def get_meldr() -> Meldr:
    global _meldr
    if _meldr is None:
        _meldr = Meldr.from_config("meldr.yaml")
    return _meldr


def run_source(
    source: str,
    target: str,
    params: dict[str, Any],
    idempotency_key: str,
) -> dict[str, Any]:
    results: list[MeldrResult] = get_meldr().run(source, target, **params)
    return {
        "idempotency_key": idempotency_key,
        "reports": [result.to_dict() for result in results],
    }
```

The key is a queue value.
Meldr returns it in the task result.
The queue or the application must reject a second task when it must not run twice.
The table sink still upserts rows by the Meldr record key.
Set an entity `sequence_field` when source updates can arrive out of order.

## FastAPI

Use a lifespan function for one instance in one Uvicorn worker.
Use `await mr.ameld()` in an async handler.
For a long source, send a queue task from the handler.
The example below shows a short request path for a test client.

```python
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Annotated

from fastapi import FastAPI, Header, Request

from meldr import Meldr, RawBatch


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.meldr = Meldr.from_config("meldr.yaml")
    try:
        yield
    finally:
        await app.state.meldr.aclose()


app = FastAPI(lifespan=lifespan)


@app.post("/harmonize")
async def harmonize(
    rows: list[dict[str, Any]],
    request: Request,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
) -> dict[str, Any]:
    result = await request.app.state.meldr.ameld(
        RawBatch.from_dicts(rows, "api"),
        "Organization",
        write=False,
        index=False,
    )
    return {"idempotency_key": idempotency_key, "result": result.to_dict()}
```

The client sends the header value as `idempotency_key`.
For a queue path, replace the call to `ameld()` with `queue.enqueue()`.
Pass a source or file reference, not the rows.
Keep the same key when the client retries the request.

FastAPI's test client runs the lifespan when it is used as a context manager.
The integration test uses the fake provider and does not use a network.

## Django

Use `AppConfig.ready()` to make one instance in each Django process.
Register `close()` with `atexit`.
Do not make the instance in a view.
For a Celery prefork worker, use the Celery process signal in the next section.

```python
import atexit

from django.apps import AppConfig
from django.conf import settings

from meldr import Meldr


class HarmonizationConfig(AppConfig):
    name = "harmonization"
    meldr: Meldr | None = None

    def ready(self) -> None:
        if self.meldr is None:
            self.meldr = Meldr.from_config(settings.MELDR_CONFIG)
            atexit.register(self.meldr.close)
```

A Django view can read the header and send a task.
The header value becomes `idempotency_key`.

```python
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from .tasks import harmonize_task


@require_POST
def start_harmonization(request):
    idempotency_key = request.headers["Idempotency-Key"]
    task = harmonize_task.delay(
        "erp",
        "Organization",
        {"table": "vendors"},
        idempotency_key=idempotency_key,
    )
    return JsonResponse({"task_id": task.id, "idempotency_key": idempotency_key}, status=202)
```

## Flask

Use an app factory.
Store the instance in `app.extensions`.
Register `close()` when the factory makes the instance.
Do not close the shared instance in `teardown_appcontext`.
That hook runs after each request.

```python
import atexit

from flask import Flask, jsonify, request

from meldr import Meldr


def create_app() -> Flask:
    app = Flask(__name__)
    mr = Meldr.from_config("meldr.yaml")
    app.extensions["meldr"] = mr
    atexit.register(mr.close)

    @app.post("/harmonize")
    def start_harmonization():
        idempotency_key = request.headers["Idempotency-Key"]
        task = harmonize_task.delay(
            "erp",
            "Organization",
            {"table": "vendors"},
            idempotency_key=idempotency_key,
        )
        return jsonify(task_id=task.id, idempotency_key=idempotency_key), 202

    return app
```

Use the Flask test client with an app made by `create_app()`.
Close the instance in the test teardown.
The integration test uses the fake provider and does not use a network.

## Celery

Make the instance after Celery forks a worker.
When a worker stops, set `acks_late=True` so Celery sends the task again.
Use transient errors in the retry list.
Do not retry a review or spec conflict without a new decision.

```python
from celery import shared_task
from celery.signals import worker_process_init, worker_process_shutdown

from meldr import Meldr

_meldr: Meldr | None = None


@worker_process_init.connect
def start_meldr(**_: object) -> None:
    global _meldr
    _meldr = Meldr.from_config("meldr.yaml")


@worker_process_shutdown.connect
def stop_meldr(**_: object) -> None:
    global _meldr
    if _meldr is not None:
        _meldr.close()
        _meldr = None


@shared_task(bind=True, acks_late=True)
def harmonize_task(
    self,
    source: str,
    target: str,
    params: dict[str, object],
    *,
    idempotency_key: str | None = None,
) -> dict[str, object]:
    key = idempotency_key or self.request.id
    if _meldr is None:
        raise RuntimeError("The Meldr worker instance is not ready.")
    return run_source(source, target, params, key)
```

Celery keeps the task ID when it retries a task.
An API handler can set the task ID to its `Idempotency-Key`.
The task still must read data from a durable source.
Do not pass rows in a broker message.
Use a Celery chord when one source has many partitions.

For a Django project, keep the Celery signal code in the worker module.
Do not rely on a `Meldr` instance made before the fork.

## Configuration for many workers

Use one shared SQL URL for the spec store.
Use a shared SQL table sink or an object store for output.
Do not use the default local DuckDB file from two worker processes.

```yaml
stores:
  type: sql
  url: "${env:MELDR_DB_URL}"
sinks:
  tables:
    type: sql
    url: "${env:MELDR_DB_URL}"
    namespace: harmonized
```

Use a tenant value when one shared store serves separate customers.
Pass the same tenant to each worker.
Keep secrets as environment references.

## Choose a pattern

- Use `Meldr()` in a script or a notebook.
- Use FastAPI lifespan for short async calls.
- Use a Django `AppConfig` or a Flask app factory for process lifetime.
- Use Celery for long work, retries, and partitions.
- Use `apply_mapping()` in a worker when the worker already has an approved spec.
- Return `to_dict()` from a task.
  It returns counts and warnings without rows by default.
