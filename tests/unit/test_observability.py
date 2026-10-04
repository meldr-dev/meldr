from __future__ import annotations

from pathlib import Path

import polars as pl
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from meldr import LLMGateway, Meldr, RawBatch, request_id_context
from meldr.llm.providers.fake import FakeProvider
from meldr.observability import safe_log
from meldr.pipeline.oracle import heuristic_oracle
from tests.helpers import make_config


def test_request_id_and_otel_spans(tmp_path: Path) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    mr = Meldr(
        make_config(tmp_path / "state"),
        llm=LLMGateway(FakeProvider(mode="oracle", oracle=heuristic_oracle), retry_base_s=0.0),
    )
    batch = RawBatch.from_polars(
        pl.DataFrame({"org_id": ["1"], "legal_name": ["Acme Inc"], "country": ["US"]}),
        "api",
    )
    with request_id_context("request-1"):
        logged = safe_log("test", rows=1)
        result = mr.meld(batch, "Organization", index=False)
    mr.close()

    spans = exporter.get_finished_spans()
    names = [span.name for span in spans]
    assert "meldr.run" in names
    assert "meldr.llm.generate" in names
    assert "meldr.sink.write" in names
    assert result.report.rows_out == 1
    assert logged["request_id"] == "request-1"
    assert all(span.attributes["meldr.request_id"] == "request-1" for span in spans)
