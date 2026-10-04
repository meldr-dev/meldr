"""REST connector: secret-referenced auth, pagination, rate limits, host allow-list."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlparse

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt

from meldr._async import iter_sync
from meldr._retry import RetryAfterWait, parse_retry_after
from meldr.config import AuthConfig, PaginationConfig
from meldr.core.errors import ConnectorError
from meldr.core.types import Modality, RawBatch, json_records
from meldr.observability import safe_log
from meldr.secrets import SecretRef, interpolate_str

if TYPE_CHECKING:
    from meldr.config import SourceConfig

MAX_REDIRECTS = 5


class _Retryable(Exception):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def dig(obj: Any, path: str | None) -> Any:
    """Follow a dot path into nested dicts ('meta.next_cursor')."""
    if not path:
        return obj
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _secret(ref: SecretRef | None) -> str:
    if ref is None:
        raise ConnectorError(
            "Auth is missing a necessary secret reference. Set it in the source `auth` config."
        )
    return ref.reveal()


def _next_link(header: str | None) -> str | None:
    if not header:
        return None
    for part in header.split(","):
        seg = part.split(";")
        if len(seg) >= 2 and any(s.strip() in ('rel="next"', "rel=next") for s in seg[1:]):
            return seg[0].strip().strip("<>")
    return None


class RestConnector:
    """Pulls JSON records from one API host."""

    modality = Modality.JSON

    def __init__(
        self,
        source_id: str,
        base_url: str,
        *,
        endpoint: str = "",
        auth: AuthConfig | None = None,
        pagination: PaginationConfig | None = None,
        records_path: str | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        rate_limit_per_s: float | None = None,
        max_retries: int = 4,
        retry_base_s: float = 1.0,
        timeout_s: float = 30.0,
        batch_size: int = 1,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.source_id = source_id
        self.base_url = base_url.rstrip("/") + "/"
        self.endpoint = endpoint
        self.auth = auth or AuthConfig()
        self.pagination = pagination or PaginationConfig()
        self.records_path = records_path
        self.params = dict(params or {})
        self.headers = dict(headers or {})
        self.min_interval = 1.0 / rate_limit_per_s if rate_limit_per_s else 0.0
        self.max_retries = max_retries
        self.retry_base_s = retry_base_s
        self.timeout_s = timeout_s
        self.batch_size = batch_size  # combine pages until a batch has this many records
        self._transport = transport
        self._last_request = 0.0
        self._token: str | None = None
        host = urlparse(self.base_url).netloc
        self.allowed_hosts = {host}
        if self.auth.token_url:
            self.allowed_hosts.add(urlparse(self.auth.token_url).netloc)

    def _check_host(self, url: str) -> None:
        host = urlparse(url).netloc
        if host not in self.allowed_hosts:
            raise ConnectorError(
                f"Meldr refused a request to host '{host}'. Source '{self.source_id}' can only "
                f"send requests to {sorted(self.allowed_hosts)}."
            )

    async def _auth(self, client: httpx.AsyncClient) -> tuple[dict[str, str], dict[str, str]]:
        a = self.auth
        if a.type == "bearer":
            return {"Authorization": f"Bearer {_secret(a.token)}"}, {}
        if a.type == "api_key":
            if a.query_param:
                return {}, {a.query_param: _secret(a.token)}
            return {a.header or "X-API-Key": _secret(a.token)}, {}
        if a.type == "basic":
            creds = httpx.BasicAuth(a.username or "", _secret(a.password))
            req = creds.auth_flow(httpx.Request("GET", self.base_url))
            return {"Authorization": next(req).headers["Authorization"]}, {}
        if a.type == "oauth2_client_credentials":
            if self._token is None:
                if not a.token_url:
                    raise ConnectorError("oauth2_client_credentials auth must have `token_url`.")
                self._check_host(a.token_url)
                data = {
                    "grant_type": "client_credentials",
                    "client_id": a.client_id or "",
                    "client_secret": _secret(a.client_secret),
                }
                if a.scope:
                    data["scope"] = a.scope
                resp = await client.post(a.token_url, data=data)
                if resp.status_code != 200:
                    raise ConnectorError(f"OAuth token request failed ({resp.status_code}).")
                self._token = str(resp.json()["access_token"])
            return {"Authorization": f"Bearer {self._token}"}, {}
        return {}, {}

    async def _throttle(self) -> None:
        if self.min_interval:
            wait = self._last_request + self.min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
        self._last_request = time.monotonic()

    async def _get(
        self, client: httpx.AsyncClient, url: str, params: dict[str, Any]
    ) -> httpx.Response:
        headers, auth_params = await self._auth(client)
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(_Retryable),
            stop=stop_after_attempt(self.max_retries + 1),
            wait=RetryAfterWait(self.retry_base_s, cap=60.0),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                current, current_params = url, {**params, **auth_params}
                for _ in range(MAX_REDIRECTS + 1):
                    self._check_host(current)
                    await self._throttle()
                    try:
                        resp = await client.get(
                            current,
                            params=current_params or None,
                            headers={**self.headers, **headers},
                        )
                    except (httpx.TimeoutException, httpx.TransportError) as e:
                        raise _Retryable(type(e).__name__) from e
                    if resp.is_redirect:
                        location = resp.headers.get("location", "")
                        current = urljoin(current, location)
                        current_params = {}
                        continue
                    break
                else:
                    raise ConnectorError("Too many redirects.")
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise _Retryable(
                        f"HTTP {resp.status_code}",
                        retry_after=parse_retry_after(resp.headers.get("retry-after")),
                    )
                if resp.status_code >= 400:
                    raise ConnectorError(
                        f"GET {urlparse(current).path} returned HTTP {resp.status_code}."
                    )
                return resp
        raise AssertionError("unreachable")  # pragma: no cover

    def _records(self, body: Any) -> tuple[list[dict[str, Any]], int]:
        """The JSON objects of the page, and its item count (for offset pagination)."""
        data = body if isinstance(body, list) else dig(body, self.records_path)
        if data is None:
            return [], 0
        if isinstance(data, dict):
            return [data], 1
        return [r for r in data if isinstance(r, dict)], len(data)

    def _batch(self, rows: list[dict[str, Any]], ids: list[str], meta: dict[str, Any]) -> RawBatch:
        records = json_records(rows, self.source_id, ids)
        return RawBatch(
            source_id=self.source_id, modality=Modality.JSON, records=records, source_metadata=meta
        )

    async def apull(self, **params: Any) -> AsyncIterator[RawBatch]:
        """Async pages -> batches. Params override query params, and `endpoint` the path."""
        p = self.pagination
        endpoint = str(params.pop("endpoint", self.endpoint))
        url = urljoin(self.base_url, endpoint.lstrip("/"))
        query = {**self.params, **params}
        meta = {"entity_hint": endpoint.strip("/").split("/")[-1] or self.source_id}
        async with httpx.AsyncClient(
            transport=self._transport, timeout=self.timeout_s, follow_redirects=False
        ) as client:
            page_no = p.start_page
            offset = 0
            cursor: Any = None
            next_url: str | None = url
            rows_buf: list[dict[str, Any]] = []
            ids_buf: list[str] = []
            for page in range(p.max_pages):
                q = dict(query)
                if p.type == "page":
                    q[p.page_param] = page_no
                    if p.size_param:
                        q[p.size_param] = p.page_size
                elif p.type == "offset":
                    q[p.offset_param] = offset
                    q[p.limit_param] = p.page_size
                elif p.type == "cursor" and cursor is not None:
                    q[p.cursor_param] = cursor
                target = url
                if p.type == "link_header":
                    if next_url is None:
                        break
                    # The next link already contains its query string.
                    target, q = next_url, (q if page == 0 else {})
                resp = await self._get(client, target, q)
                try:
                    body = resp.json()
                except ValueError as e:
                    raise ConnectorError("API response is not JSON.") from e
                rows, items = self._records(body)
                rows_buf.extend(rows)
                ids_buf.extend(f"{page}:{i}" for i in range(len(rows)))
                if len(rows_buf) >= self.batch_size:
                    yield self._batch(rows_buf, ids_buf, meta)
                    rows_buf, ids_buf = [], []
                if p.type == "none":
                    break
                if p.type == "page":
                    if not rows:
                        break
                    page_no += 1
                elif p.type == "offset":
                    if not items:  # servers can cap the limit, so only an empty page ends it
                        break
                    offset += items
                elif p.type == "cursor":
                    cursor = dig(body, p.next_path)
                    if not cursor:
                        break
                elif p.type == "link_header":
                    link = _next_link(resp.headers.get("link"))
                    next_url = urljoin(target, link) if link else None
            else:
                safe_log(
                    "rest.max_pages_reached",
                    level=logging.WARNING,
                    source_id=self.source_id,
                    max_pages=p.max_pages,
                )
            if rows_buf:
                yield self._batch(rows_buf, ids_buf, meta)

    def pull(self, **params: Any) -> Iterator[RawBatch]:
        """Yield batches as pages arrive (sync wrapper around `apull`)."""
        yield from iter_sync(self.apull(**params))


def from_config(source_id: str, cfg: SourceConfig, batch_size: int) -> RestConnector:
    if not cfg.base_url:
        raise ConnectorError(f"sources.{source_id}: the rest connector must have `base_url`.")
    return RestConnector(
        source_id,
        cfg.base_url,
        endpoint=cfg.endpoint,
        auth=cfg.auth,
        pagination=cfg.pagination,
        records_path=cfg.records_path,
        params={k: interpolate_str(v) for k, v in cfg.params.items()},
        headers={k: interpolate_str(v) for k, v in cfg.headers.items()},
        rate_limit_per_s=cfg.rate_limit_per_s,
        batch_size=batch_size,
    )
