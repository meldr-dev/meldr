from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.testclient import TestClient
from flask import Flask, jsonify, request

from meldr import Meldr, RawBatch
from tests.helpers import make_config

ROWS = [{"org_id": "O1", "legal_name": "Acme Inc", "country": "US"}]


def make_fastapi_app(root: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.meldr = Meldr(make_config(root))
        try:
            yield
        finally:
            await app.state.meldr.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.post("/harmonize")
    async def harmonize(
        body: list[dict[str, Any]],
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        if idempotency_key is None:
            raise HTTPException(status_code=400, detail="Idempotency-Key is required.")
        result = await request.app.state.meldr.ameld(
            RawBatch.from_dicts(body, "api"),
            "Organization",
            write=False,
            index=False,
        )
        return {"idempotency_key": idempotency_key, "result": result.to_dict()}

    return app


def make_flask_app(root: Path) -> Flask:
    app = Flask(__name__)
    app.extensions["meldr"] = Meldr(make_config(root))

    @app.post("/harmonize")
    def harmonize() -> tuple[Any, int] | Any:
        idempotency_key = request.headers.get("Idempotency-Key")
        if idempotency_key is None:
            return jsonify(error="Idempotency-Key is required."), 400
        body = request.get_json()
        if not isinstance(body, list):
            return jsonify(error="The request body must be a list of rows."), 400
        result = app.extensions["meldr"].meld(
            RawBatch.from_dicts(body, "api"),
            "Organization",
            write=False,
            index=False,
        )
        return jsonify(idempotency_key=idempotency_key, result=result.to_dict())

    return app


def test_fastapi_test_client_uses_lifespan_and_async_meld(tmp_path: Path) -> None:
    with TestClient(make_fastapi_app(tmp_path / "fastapi")) as client:
        response = client.post(
            "/harmonize",
            headers={"Idempotency-Key": "request-1"},
            json=ROWS,
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["idempotency_key"] == "request-1"
    assert payload["result"]["rows_valid"] == 1


def test_flask_test_client_uses_app_factory_and_sync_meld(tmp_path: Path) -> None:
    app = make_flask_app(tmp_path / "flask")
    try:
        with app.test_client() as client:
            response = client.post(
                "/harmonize",
                headers={"Idempotency-Key": "request-2"},
                json=ROWS,
            )
    finally:
        app.extensions["meldr"].close()

    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert payload["idempotency_key"] == "request-2"
    assert payload["result"]["rows_valid"] == 1
