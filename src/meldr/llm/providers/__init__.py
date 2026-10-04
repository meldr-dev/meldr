"""Provider construction from config."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from meldr.core.errors import ConfigError
from meldr.llm.base import LLMProvider

if TYPE_CHECKING:
    from meldr.config import EmbeddingsConfig, ProviderConfig
    from meldr.llm.base import Embedder
    from meldr.llm.providers.fake import Oracle
    from meldr.secrets import SecretRef


ProviderFactory = Callable[["ProviderConfig"], LLMProvider]
_CUSTOM: dict[str, ProviderFactory] = {}


def register_provider(type_name: str, factory: ProviderFactory) -> None:
    """Register a custom provider type (selected with `options.plugin: <type_name>`)."""
    _CUSTOM[type_name] = factory


def _secret(ref: SecretRef | None) -> str | None:
    return None if ref is None else ref.reveal()


def build_provider(
    name: str, cfg: ProviderConfig, *, fake_oracle: Oracle | None = None
) -> LLMProvider:
    """Instantiate the provider described by `cfg` (lazy SDK imports)."""
    plugin = cfg.options.get("plugin")
    if plugin is not None:
        if plugin not in _CUSTOM:
            raise ConfigError(f"llm.providers.{name}: unknown provider plugin '{plugin}'.")
        return _CUSTOM[plugin](cfg)
    if cfg.type == "fake":
        from meldr.llm.providers.fake import FakeProvider

        opts = dict(cfg.options)
        responses = opts.pop("responses", ())
        mode = opts.pop("mode", "scripted" if responses else "oracle")
        return FakeProvider(
            mode=mode,
            responses=responses,
            oracle=fake_oracle if mode == "oracle" else None,
            model=cfg.model,
            is_local=bool(opts.pop("is_local", True)),
            cassette_dir=opts.pop("cassette_dir", None),
        )
    if cfg.type == "anthropic":
        from meldr.llm.providers.anthropic import AnthropicProvider

        return AnthropicProvider(cfg.model, api_key=_secret(cfg.api_key), base_url=cfg.base_url)
    if cfg.type == "openai":
        from meldr.llm.providers.openai import OpenAIProvider

        return OpenAIProvider(cfg.model, api_key=_secret(cfg.api_key), base_url=cfg.base_url)
    if cfg.type == "openai_compatible":
        from meldr.llm.providers.openai_compatible import OpenAICompatibleProvider

        if not cfg.base_url:
            raise ConfigError(f"llm.providers.{name}: openai_compatible must have `base_url`.")
        return OpenAICompatibleProvider(
            cfg.model,
            base_url=cfg.base_url,
            api_key=_secret(cfg.api_key),
            is_local=bool(cfg.options.get("is_local", True)),
        )
    if cfg.type == "litellm":
        from meldr.llm.providers.litellm import LiteLLMProvider

        return LiteLLMProvider(
            cfg.model,
            api_key=_secret(cfg.api_key),
            base_url=cfg.base_url,
            is_local=bool(cfg.options.get("is_local", False)),
        )
    raise ConfigError(
        f"llm.providers.{name}: unknown provider type {cfg.type!r}"
    )  # pragma: no cover


def build_embedder(cfg: EmbeddingsConfig) -> Embedder:
    """Instantiate the configured embedder."""
    if cfg.provider == "fake":
        from meldr.llm.embedders.fake import FakeEmbedder

        return FakeEmbedder(dim=cfg.dim)
    if cfg.provider in ("openai", "openai_compatible"):
        from meldr.llm.embedders.openai import OpenAIEmbedder

        if not cfg.model:
            raise ConfigError("Set embeddings.model for OpenAI-style embeddings.")
        return OpenAIEmbedder(
            cfg.model, dim=cfg.dim, api_key=_secret(cfg.api_key), base_url=cfg.base_url
        )
    from meldr.llm.embedders.sentence_transformers import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(cfg.model or "all-MiniLM-L6-v2")
