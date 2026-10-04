"""MeldrConfig and the YAML loader."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator

from meldr._yaml import read_mapping
from meldr.core.errors import ConfigError
from meldr.llm.policy import DataClass, SourcePolicy
from meldr.secrets import SecretRef, interpolate_str

EnvStr = Annotated[str, BeforeValidator(interpolate_str)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StorageConfig(_Model):
    root: EnvStr = "./.meldr"


class StoresConfig(_Model):
    type: Literal["file", "sql"] = "file"
    url: SecretRef | None = None

    @model_validator(mode="after")
    def _sql_needs_url(self) -> StoresConfig:
        if self.type == "sql" and self.url is None:
            raise ValueError("stores.url is necessary when stores.type is 'sql'.")
        return self


class ProviderConfig(_Model):
    type: Literal["anthropic", "openai", "openai_compatible", "litellm", "fake"]
    model: EnvStr = "fake-model"
    api_key: SecretRef | None = None
    base_url: EnvStr | None = None
    data_policy: list[DataClass] | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class RoutingConfig(_Model):
    mapping: str | None = None
    escalation: str | None = None
    extraction: str | None = None
    vision: str | None = None


class LLMConfig(_Model):
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    max_concurrency: int = 4
    max_retries: int = 4
    max_repairs: int = 2
    timeout_s: float = 120.0
    escalate_below: float = 0.9


class PriceConfig(_Model):
    input: float | None = None
    output: float | None = None


class EmbeddingsConfig(_Model):
    provider: Literal["fake", "openai", "openai_compatible", "sentence_transformers"] = "fake"
    model: EnvStr | None = None
    base_url: EnvStr | None = None
    api_key: SecretRef | None = None
    dim: int = 64


class ConfidenceWeights(_Model):
    llm: float = 0.6
    candidate: float = 0.2
    dryrun: float = 0.2


class MappingConfig(_Model):
    auto_approve_threshold: float = 0.90
    review_required_for_pii: bool = True
    always_review: list[str] = Field(default_factory=list)
    top_k_candidates: int = 5
    min_candidate_score: float = 0.25
    max_fields_per_llm_call: int = 40
    self_consistency: int = Field(default=1, ge=1, le=9)  # >1: majority vote of N samples
    learn_from_approved: bool = True  # approved mappings become candidates for new sources
    sample_size: int = 20
    sample_rows: int = 10_000
    dryrun_rows: int = 1000
    max_dryrun_failure: float = 0.02
    on_unapproved: Literal["block", "run_approved_only"] = "run_approved_only"
    on_breaking_drift: Literal["block", "quarantine"] = "block"
    confidence_weights: ConfidenceWeights = Field(default_factory=ConfidenceWeights)


class ExecutionConfig(_Model):
    batch_size: int = 50_000
    on_invalid: Literal["quarantine", "drop", "fail"] = "quarantine"


class ExtractionConfig(_Model):
    chunk_tokens: int = 800
    overlap_tokens: int = 100
    single_pass_max_tokens: int = 12_000
    max_documents_per_call: int = Field(default=100, ge=1)
    max_document_bytes: int = Field(default=10_000_000, ge=1)
    resolve_timeout_s: float = Field(default=30.0, gt=0)


class IndexingConfig(_Model):
    records: bool = False
    batch_size: int = 256


class MediaConfig(_Model):
    max_frames: int = 60
    min_frame_gap_s: float = 5.0
    frame_interval_s: float = 30.0
    window_s: float = 60.0
    frames_per_call: int = 8
    timeout_s: float = 600.0


class SchemasConfig(_Model):
    paths: list[EnvStr] = Field(default_factory=list)  # directories of entity YAML files
    linkml: list[EnvStr] = Field(default_factory=list)  # LinkML schema files


class AuthConfig(_Model):
    type: Literal["bearer", "api_key", "basic", "oauth2_client_credentials", "none"] = "none"
    token: SecretRef | None = None
    header: str | None = None
    query_param: str | None = None
    username: EnvStr | None = None
    password: SecretRef | None = None
    token_url: EnvStr | None = None
    client_id: EnvStr | None = None
    client_secret: SecretRef | None = None
    scope: str | None = None


class PaginationConfig(_Model):
    type: Literal["none", "page", "offset", "cursor", "link_header"] = "none"
    page_param: str = "page"
    start_page: int = 1
    size_param: str | None = None
    page_size: int = 100
    offset_param: str = "offset"
    limit_param: str = "limit"
    cursor_param: str = "cursor"
    next_path: str | None = None
    max_pages: int = 1000

    @model_validator(mode="after")
    def _cursor_needs_path(self) -> PaginationConfig:
        if self.type == "cursor" and not self.next_path:
            raise ValueError("cursor pagination must set next_path (where the next cursor is)")
        return self


class SourceConfig(_Model):
    connector: str
    data_policy: SourcePolicy | None = None
    entity_hint: str | None = None
    # sql
    url: SecretRef | None = None
    table: str | None = None
    query: str | None = None
    # files
    path: EnvStr | None = None
    format: str | None = None
    # rest
    base_url: EnvStr | None = None
    endpoint: str = ""
    auth: AuthConfig = Field(default_factory=AuthConfig)
    pagination: PaginationConfig = Field(default_factory=PaginationConfig)
    records_path: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    rate_limit_per_s: float | None = None
    arrays: Literal["json", "explode"] = "json"
    options: dict[str, Any] = Field(default_factory=dict)


class TableSinkConfig(_Model):
    type: str = "duckdb"  # duckdb | parquet | sql | delta | iceberg | <registered plugin>
    path: EnvStr | None = (
        None  # duckdb file, or parquet or delta root (local path or object-store URI)
    )
    url: SecretRef | None = None  # sql: SQLAlchemy URL of the database or warehouse
    namespace: str | None = None  # database schema for sql, catalog namespace for iceberg
    options: dict[str, Any] = Field(default_factory=dict)  # delta storage options, iceberg catalog


class VectorSinkConfig(_Model):
    type: str = "memory"  # memory | lancedb | qdrant | pgvector | chroma | <plugin>
    path: EnvStr | None = None
    url: SecretRef | None = None


class SinksConfig(_Model):
    tables: TableSinkConfig = Field(default_factory=TableSinkConfig)
    vectors: VectorSinkConfig = Field(default_factory=VectorSinkConfig)


class MeldrConfig(_Model):
    """Root configuration. Unknown keys fail with their path."""

    project: str = "meldr"
    storage: StorageConfig = Field(default_factory=StorageConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    pricing: dict[str, PriceConfig] = Field(default_factory=dict)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    mapping: MappingConfig = Field(default_factory=MappingConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    stores: StoresConfig = Field(default_factory=StoresConfig)
    schemas: SchemasConfig = Field(default_factory=SchemasConfig)
    sources: dict[str, SourceConfig] = Field(default_factory=dict)
    sinks: SinksConfig = Field(default_factory=SinksConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_dir: Path | None = None) -> MeldrConfig:
        """Validate a config mapping and resolve relative paths against `base_dir`."""
        try:
            cfg = cls.model_validate(data)
        except ValidationError as e:
            raise ConfigError(_format_errors(e)) from e
        if base_dir is not None:
            cfg = cfg._rebase(base_dir)
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> MeldrConfig:
        """Load and validate a YAML config file."""
        data = read_mapping(path, "config")
        return cls.from_dict(data, base_dir=Path(path).parent.resolve())

    def _rebase(self, base: Path) -> MeldrConfig:
        def fix(v: str) -> str:
            return v if Path(v).is_absolute() or urlparse(v).scheme else str((base / v).resolve())

        cfg = self.model_copy(deep=True)
        cfg.storage.root = fix(cfg.storage.root)
        cfg.schemas.paths = [fix(p) for p in cfg.schemas.paths]
        cfg.schemas.linkml = [fix(p) for p in cfg.schemas.linkml]
        for src in cfg.sources.values():
            if src.path:
                src.path = fix(src.path)
        if cfg.sinks.tables.path:
            cfg.sinks.tables.path = fix(cfg.sinks.tables.path)
        if cfg.sinks.vectors.path:
            cfg.sinks.vectors.path = fix(cfg.sinks.vectors.path)
        return cfg


def _format_errors(e: ValidationError) -> str:
    lines = ["Invalid Meldr config:"]
    for err in e.errors():
        loc = ".".join(str(p) for p in err["loc"])
        msg = err["msg"].removeprefix("Value error, ")
        if err["type"] == "extra_forbidden":
            msg = "unknown key (check spelling or remove it)"
        lines.append(f"  {loc}: {msg}")
    return "\n".join(lines)
