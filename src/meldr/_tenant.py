"""Helpers for isolated tenant state."""

from __future__ import annotations

import hashlib
import re

from meldr.core.errors import ConfigError

_TENANT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")


def validate_tenant(tenant: str | None) -> str | None:
    """Check a tenant name before it scopes storage."""
    if tenant is None:
        return None
    if not _TENANT_NAME.fullmatch(tenant):
        raise ConfigError(
            "tenant must start with a letter or digit and contain only letters, digits, "
            "'.', '_' or '-'."
        )
    return tenant


def tenant_token(tenant: str) -> str:
    """Return a stable safe token for a tenant namespace."""
    return "tenant_" + hashlib.sha256(tenant.encode("utf-8")).hexdigest()[:24]


def scoped_key(tenant: str | None, value: str) -> str:
    """Add a tenant token to a private storage key."""
    if tenant is None:
        return value
    return f"{tenant_token(tenant)}::{value}"


def scoped_identifier(tenant: str, value: str) -> str:
    """Add a tenant token to a file or schema identifier."""
    return f"{tenant_token(tenant)}__{value}"
