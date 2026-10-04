"""Secret references: `${env:NAME}` and `${vault:path#key}`, resolved through a registry."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from meldr.core.errors import ConfigError, SecretReferenceError

Resolver = Callable[[str], str]

_REF = re.compile(r"\$\{(?P<scheme>[a-z_]+):(?P<body>[^}]*)\}")
_FULL_REF = re.compile(rf"^{_REF.pattern}$")

_resolvers: dict[str, Resolver] = {}


def register(scheme: str, resolver: Resolver) -> None:
    """Register a resolver for `${scheme:...}` references (for example, your vault client)."""
    _resolvers[scheme] = resolver


def _env_resolver(body: str) -> str:
    name, _, default = body.partition(":-")
    value = os.environ.get(name)
    if value is not None:
        return value
    if ":-" in body:
        return default
    raise SecretReferenceError(
        f"Environment variable {name} is not set. Export it or add it to your environment."
    )


def _vault_stub(body: str) -> str:
    raise SecretReferenceError(
        f"Cannot resolve ${{vault:{body}}}: there is no registered vault resolver. Register one "
        "with meldr.secrets.register('vault', resolver) before you load this config."
    )


register("env", _env_resolver)
register("vault", _vault_stub)


def is_reference(value: str) -> bool:
    """True when `value` is one full `${scheme:...}` reference with no other text."""
    return bool(_FULL_REF.match(value.strip()))


def resolve(ref: str) -> str:
    """Resolve one full reference string to its value."""
    m = _FULL_REF.match(ref.strip())
    if not m:
        raise SecretReferenceError(f"'{ref}' is not a ${{scheme:...}} reference.")
    scheme, body = m.group("scheme"), m.group("body")
    resolver = _resolvers.get(scheme)
    if resolver is None:
        raise SecretReferenceError(
            f"Unknown secret scheme '{scheme}'. Known: {sorted(_resolvers)}. "
            "Register one with meldr.secrets.register()."
        )
    return resolver(body)


def interpolate(value: str) -> str:
    """Replace every `${...}` reference inside a non-secret string."""
    return _REF.sub(lambda m: resolve(m.group(0)), value)


def interpolate_str(value: Any) -> Any:
    """Pydantic before-validator for non-secret strings."""
    return interpolate(value) if isinstance(value, str) and "${" in value else value


class SecretRef:
    """A reference to a secret. Only `reveal()` resolves the value."""

    __slots__ = ("ref",)

    def __init__(self, ref: str) -> None:
        ref = ref.strip()
        if not is_reference(ref):
            raise ConfigError("use ${env:…} or ${vault:…} instead of a literal secret")
        if ":-" in ref:
            raise ConfigError("secret fields cannot have defaults (${env:NAME:-default})")
        self.ref = ref

    def reveal(self) -> str:
        """The plain secret value, to give to a client library. Do not log it."""
        return resolve(self.ref)

    def __repr__(self) -> str:
        return f"SecretRef({self.ref!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SecretRef) and other.ref == self.ref

    def __hash__(self) -> int:
        return hash(self.ref)

    @classmethod
    def _validate(cls, value: Any) -> SecretRef:
        if isinstance(value, SecretRef):
            return value
        if not isinstance(value, str):
            raise ValueError("use ${env:…} or ${vault:…} instead of a literal secret")
        try:
            return cls(value)
        except ConfigError as e:
            raise ValueError(str(e)) from e

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(lambda v: v.ref),
            json_schema_input_schema=core_schema.str_schema(),
        )
