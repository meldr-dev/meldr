"""Meldr: AI-assisted data harmonization.

LLMs propose mappings at design time. Approved mappings run deterministically at run time.
"""

from __future__ import annotations

from meldr._version import __version__
from meldr.api import (
    Applied,
    aharmonize,
    apply_mapping,
    apropose_mapping,
    harmonize,
    propose_mapping,
)
from meldr.config import MeldrConfig, StoresConfig
from meldr.connectors.base import Connector, register_connector
from meldr.core.canonical import (
    CanonicalEntity,
    CanonicalField,
    EntityExtraction,
    SchemaRegistry,
)
from meldr.core.dtypes import DataType
from meldr.core.errors import (
    ConfigError,
    ConnectorError,
    DriftBlockedError,
    ExternalToolError,
    LLMError,
    LLMOutputError,
    LLMTransientError,
    MeldrError,
    MissingExtraError,
    PolicyViolation,
    SecretReferenceError,
    SinkError,
    SpecConflictError,
    SpecValidationError,
)
from meldr.core.mapping import (
    FieldMapping,
    FieldStatus,
    MappingSpec,
    Origin,
    ReviewCandidate,
    ReviewDecision,
    ReviewItem,
    SpecStatus,
    TransformStep,
)
from meldr.core.schema import FieldProfile, SourceSchema, shape_fingerprint
from meldr.core.segments import ContentSegment, Evidence, ExtractedRecord, Locator, Relation
from meldr.core.types import Modality, RawBatch, RawRecord
from meldr.domains import pack_path
from meldr.indexing.sinks import register_sink
from meldr.indexing.sinks.base import IndexItem, SearchHit, TableSink, VectorSink, WriteResult
from meldr.lineage.store import LineageStore, SQLLineageStore
from meldr.llm.base import Capability, Embedder, LLMProvider, Message
from meldr.llm.cache import SQLResponseCache, SQLVectorCache
from meldr.llm.gateway import LLMGateway
from meldr.llm.policy import DataClass, DataPolicy
from meldr.llm.providers.fake import FakeProvider
from meldr.mapping.store import FileMappingStore, MappingStore, SQLMappingStore
from meldr.observability import RunReport, configure_logging, request_id_context, safe_log
from meldr.pipeline.meldr import Meldr
from meldr.pipeline.results import ExtractResult, HarmonizeResult, MeldrResult
from meldr.testing import check_table_sink, check_vector_sink
from meldr.transforms.registry import OpArgs, OpDef, TransformRegistry, default_registry

__all__ = [
    "Applied",
    "CanonicalEntity",
    "CanonicalField",
    "Capability",
    "ConfigError",
    "Connector",
    "ConnectorError",
    "ContentSegment",
    "DataClass",
    "DataPolicy",
    "DataType",
    "DriftBlockedError",
    "Embedder",
    "EntityExtraction",
    "Evidence",
    "ExternalToolError",
    "ExtractResult",
    "ExtractedRecord",
    "FakeProvider",
    "FieldMapping",
    "FieldProfile",
    "FieldStatus",
    "FileMappingStore",
    "HarmonizeResult",
    "IndexItem",
    "LLMError",
    "LLMGateway",
    "LLMOutputError",
    "LLMProvider",
    "LLMTransientError",
    "LineageStore",
    "Locator",
    "MappingSpec",
    "MappingStore",
    "Meldr",
    "MeldrConfig",
    "MeldrError",
    "MeldrResult",
    "Message",
    "MissingExtraError",
    "Modality",
    "OpArgs",
    "OpDef",
    "Origin",
    "PolicyViolation",
    "RawBatch",
    "RawRecord",
    "Relation",
    "ReviewCandidate",
    "ReviewDecision",
    "ReviewItem",
    "RunReport",
    "SQLLineageStore",
    "SQLMappingStore",
    "SQLResponseCache",
    "SQLVectorCache",
    "SchemaRegistry",
    "SearchHit",
    "SecretReferenceError",
    "SinkError",
    "SourceSchema",
    "SpecConflictError",
    "SpecStatus",
    "SpecValidationError",
    "StoresConfig",
    "TableSink",
    "TransformRegistry",
    "TransformStep",
    "VectorSink",
    "WriteResult",
    "__version__",
    "aharmonize",
    "apply_mapping",
    "apropose_mapping",
    "check_table_sink",
    "check_vector_sink",
    "configure_logging",
    "default_registry",
    "harmonize",
    "pack_path",
    "propose_mapping",
    "register_connector",
    "register_sink",
    "request_id_context",
    "safe_log",
    "shape_fingerprint",
]
