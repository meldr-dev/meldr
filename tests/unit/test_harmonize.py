from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from meldr import CanonicalEntity, CanonicalField, DataType, LLMGateway, Meldr
from meldr.api import propose_mapping
from meldr.core.canonical import EntityExtraction
from meldr.core.errors import ConfigError
from meldr.core.types import RawBatch
from meldr.llm.providers.fake import FakeProvider
from meldr.mapping import review
from tests.helpers import make_config


def test_harmonize_accepts_a_table_without_writing(tmp_path: Path, erp_df: pl.DataFrame) -> None:
    mr = Meldr(make_config(tmp_path / "state"))
    lineage_calls: list[object] = []
    mr.lineage.record_run = lineage_calls.append  # type: ignore[method-assign]

    result = mr.harmonize(erp_df, target="Organization", source_id="erp")

    assert result.data.height > 0
    assert result.quarantine.height >= 0
    assert result.spec is not None
    assert result.report.rows_in == erp_df.height
    assert lineage_calls == []


def test_harmonize_accepts_json_rows(tmp_path: Path) -> None:
    mr = Meldr(make_config(tmp_path / "state"))

    result = mr.harmonize(
        [{"org_id": "O1", "company": "Acme"}],
        target="Organization",
        source_id="api",
    )

    assert result.report.rows_in == 1


def test_harmonize_does_not_fetch_links_without_a_caller_resolver(tmp_path: Path) -> None:
    mr = Meldr(make_config(tmp_path / "state"))
    rows = pl.DataFrame({"id": ["P1"], "document_url": ["https://files.test/a.md"]})

    with pytest.raises(ConfigError, match="resolve"):
        mr.harmonize(
            rows,
            target="Organization",
            source_id="api",
            documents={"document_url": "Project"},
        )


def test_harmonize_routes_file_bytes_to_extraction(tmp_path: Path) -> None:
    mr = Meldr(make_config(tmp_path / "state"))

    result = mr.harmonize(
        [("note.md", b"# Acme\n\nA short note.")],
        source_id="documents",
    )

    assert result.spec is None
    assert result.report.target is None
    assert result.report.rows_in == 1


def test_harmonize_skips_profiling_for_an_approved_shape(
    tmp_path: Path, erp_df: pl.DataFrame
) -> None:
    mr = Meldr(make_config(tmp_path / "state"))
    first = mr.meld(RawBatch.from_polars(erp_df, "erp"), "Organization", write=False, index=False)
    approved = mr.store.save(review.approve(first.spec, user="test", above=0.0))

    def fail_profile(*args: object, **kwargs: object) -> object:
        raise AssertionError("the known shape must not be profiled")

    mr._profile = fail_profile  # type: ignore[method-assign]
    result = mr.harmonize(erp_df, target="Organization", source_id="erp")

    assert result.spec == approved
    assert result.report.spec_cache_hit is True
    assert result.report.llm_calls == 0


def test_review_is_serializable_and_saved_as_a_new_version(
    tmp_path: Path, erp_df: pl.DataFrame
) -> None:
    mr = Meldr(make_config(tmp_path / "state"))
    first = mr.meld(RawBatch.from_polars(erp_df, "erp"), "Organization", write=False, index=False)

    items = review.items(first.spec)
    assert items and all(item.spec_id == first.spec.spec_id for item in items)
    assert all(item.model_dump(mode="json")["target"] for item in items)

    target = items[0].target
    saved = mr.apply_review(
        first.spec,
        [{"target": target, "action": "approve"}],
        user="ana",
    )

    assert saved.version == first.spec.version + 1
    assert saved.mapping_for(target).status.value == "approved"  # type: ignore[union-attr]


def _content_registry() -> tuple[CanonicalEntity, CanonicalEntity]:
    parent = CanonicalEntity(
        name="Parent",
        version="1.0.0",
        fields=[CanonicalField(name="id", dtype=DataType.STRING, required=True)],
    )
    child = CanonicalEntity(
        name="Note",
        version="1.0.0",
        fields=[CanonicalField(name="text", dtype=DataType.STRING, required=True)],
    )
    return parent, child


def test_documents_use_the_caller_resolver_once_and_link_parent_rows(tmp_path: Path) -> None:
    parent, child = _content_registry()
    from meldr.core.canonical import SchemaRegistry

    registry = SchemaRegistry()
    registry.register(parent)
    registry.register(child)
    rows = pl.DataFrame(
        {
            "id": ["P1", "P2"],
            "contract_url": ["https://files.test/a.md", "https://files.test/a.md"],
        }
    )
    spec = review.approve(
        propose_mapping(rows.select("id"), parent, source_id="api"),
        user="test",
        targets=["id"],
    )
    provider = FakeProvider(
        responses=[
            {
                "values": {"text": "Acme"},
                "evidence": {
                    "text": {
                        "segment_id": "api:0:contract_url:0",
                        "quote": "Acme",
                    }
                },
                "confidence": {"text": 0.99},
            }
        ]
    )
    mr = Meldr(
        make_config(tmp_path / "state"),
        registry=registry,
        llm=LLMGateway(provider),
    )
    calls: list[str] = []

    def resolve(link: str) -> bytes:
        calls.append(link)
        return b"Acme"

    result = mr.harmonize(
        rows,
        target="Parent",
        source_id="api",
        spec=spec,
        documents={"contract_url": "Note"},
        resolve=resolve,
    )

    assert calls == ["https://files.test/a.md"]
    assert result.children["Note"].height == 2
    assert result.children["Note"]["_meldr_parent_record_id"].to_list() == ["0", "1"]
    assert len(provider.calls) == 1


def test_text_columns_create_child_rows_and_resolve_failures_are_quarantined(
    tmp_path: Path,
) -> None:
    parent, child = _content_registry()
    from meldr.core.canonical import SchemaRegistry

    registry = SchemaRegistry()
    registry.register(parent)
    registry.register(child)
    rows = pl.DataFrame(
        {
            "id": ["P1", "P2"],
            "contract_url": ["https://files.test/missing.md", None],
            "notes": ["hello", None],
        }
    )
    spec = review.approve(
        propose_mapping(rows.select("id"), parent, source_id="api"),
        user="test",
        targets=["id"],
    )
    provider = FakeProvider(
        responses=[
            {
                "values": {"text": "hello"},
                "evidence": {
                    "text": {
                        "segment_id": "api:0:notes:0",
                        "quote": "hello",
                    }
                },
                "confidence": {"text": 0.99},
            }
        ]
    )
    mr = Meldr(
        make_config(tmp_path / "state"),
        registry=registry,
        llm=LLMGateway(provider),
    )

    result = mr.harmonize(
        rows,
        target="Parent",
        source_id="api",
        spec=spec,
        documents={"contract_url": "Note"},
        texts={"notes": "Note"},
        resolve=lambda _: (_ for _ in ()).throw(RuntimeError("no file")),
    )

    assert result.children["Note"].height == 1
    assert result.quarantine.height == 1
    assert "resolve_failed:RuntimeError" in str(result.quarantine["_meldr_errors"][0])


def test_many_extraction_returns_one_row_for_each_item(tmp_path: Path) -> None:
    _, child = _content_registry()
    many = child.model_copy(
        update={"name": "ManyNote", "extraction": EntityExtraction(cardinality="many")}
    )
    from meldr.core.canonical import SchemaRegistry

    registry = SchemaRegistry()
    registry.register(many)
    provider = FakeProvider(
        responses=[
            {
                "items": [
                    {
                        "values": {"text": "One"},
                        "evidence": {
                            "text": {
                                "segment_id": "docs:note.md:0",
                                "quote": "One",
                            }
                        },
                        "confidence": {"text": 0.99},
                    },
                    {
                        "values": {"text": "Two"},
                        "evidence": {
                            "text": {
                                "segment_id": "docs:note.md:0",
                                "quote": "Two",
                            }
                        },
                        "confidence": {"text": 0.98},
                    },
                ]
            }
        ]
    )
    mr = Meldr(
        make_config(tmp_path / "state"),
        registry=registry,
        llm=LLMGateway(provider),
    )

    result = mr.extract(
        RawBatch.from_contents([("note.md", b"One\n\nTwo")], "docs"),
        "ManyNote",
        write=False,
        index=False,
    )

    assert result.data.height == 2
    assert result.data["text"].to_list() == ["One", "Two"]
    assert result.data["_meldr_record_id"].to_list() == ["note.md#0", "note.md#1"]


def test_relations_are_returned_and_unknown_predicates_are_quarantined(tmp_path: Path) -> None:
    _, child = _content_registry()
    relation_entity = child.model_copy(update={"name": "RelationNote", "relations": ["mentions"]})
    from meldr.core.canonical import SchemaRegistry

    registry = SchemaRegistry()
    registry.register(relation_entity)
    provider = FakeProvider(
        responses=[
            {
                "values": {"text": "One"},
                "evidence": {"text": {"segment_id": "docs:note.md:0", "quote": "One"}},
                "confidence": {"text": 0.99},
                "relations": [
                    {
                        "subject_id": "P1",
                        "predicate": "mentions",
                        "object_id": "P2",
                        "evidence": {
                            "segment_id": "docs:note.md:0",
                            "quote": "One",
                        },
                        "confidence": 0.9,
                    },
                    {
                        "subject_id": "P1",
                        "predicate": "not_allowed",
                        "object_id": "P3",
                        "evidence": {
                            "segment_id": "docs:note.md:0",
                            "quote": "One",
                        },
                        "confidence": 0.9,
                    },
                ],
            }
        ]
    )
    registry = SchemaRegistry()
    registry.register(relation_entity)
    mr = Meldr(
        make_config(tmp_path / "state"),
        registry=registry,
        llm=LLMGateway(provider),
    )

    result = mr.harmonize([("note.md", b"One")], target="RelationNote", source_id="docs")

    assert result.relations["predicate"].to_list() == ["mentions"]
    assert result.quarantine.height == 1
    assert (
        "relation_predicate_not_allowed:not_allowed"
        in result.quarantine["_meldr_errors"].to_list()[0]
    )
