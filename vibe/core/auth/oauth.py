from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
import types
from typing import Protocol
from urllib.parse import quote, urlsplit
import webbrowser

import httpx
import keyring
import keyring.errors
from pydantic import BaseModel

from vibe.core.config import (
    OAuthConfig,
    OAuthDeviceFlowStyle,
    ProviderAuthType,
    ProviderConfig,
)
from vibe.core.paths.global_paths import GLOBAL_AUTH_DIR

_SERVICE_NAME = "vibe"
_TOKEN_KEY_PREFIX = "oauth_token"


class OAuthAuthError(RuntimeError):
    pass


class AuthRequiredError(OAuthAuthError):
    pass


@dataclass
class OAuthDeviceFlowInfo:
    user_code: str
    verification_uri: str
    verification_uri_complete: str | None = None


@dataclass
class OAuthDeviceFlowHandle:
    device_code: str
    expires_in: int
    interval: int
    info: OAuthDeviceFlowInfo


class OAuthTokenSet(BaseModel):
    access_token: str
    refresh_token: str = ""
    token_type: str = "Bearer"
    expires_at: int | None = None
    scope: str = ""
    account_id: str = ""

    def is_expired(self, *, skew_seconds: int = 60) -> bool:
        if self.expires_at is None:
            return False
        return (self.expires_at - skew_seconds) <= int(time.time())


class OAuthTokenStore:
    def _keyring_username(self, provider_name: str) -> str:
        return f"{_TOKEN_KEY_PREFIX}::{provider_name}"

    def _file_path(self, provider_name: str) -> Path:
        return GLOBAL_AUTH_DIR.path / f"{provider_name}.json"

    def load(self, provider_name: str) -> OAuthTokenSet | None:
        key = self._load_from_keyring(provider_name)
        if key is not None:
            return key

        return self._load_from_file(provider_name)

    def save(self, provider_name: str, token_set: OAuthTokenSet) -> None:
        serialized = token_set.model_dump_json()
        if self._save_to_keyring(provider_name, serialized):
            return
        self._save_to_file(provider_name, serialized)

    def delete(self, provider_name: str) -> None:
        self._delete_from_keyring(provider_name)

        path = self._file_path(provider_name)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _load_from_keyring(self, provider_name: str) -> OAuthTokenSet | None:
        try:
            value = keyring.get_password(_SERVICE_NAME, self._keyring_username(provider_name))
        except Exception:
            return None

        if not value:
            return None

        try:
            return OAuthTokenSet.model_validate_json(value)
        except Exception:
            return None

    def _save_to_keyring(self, provider_name: str, serialized: str) -> bool:
        try:
            keyring.set_password(
                _SERVICE_NAME,
                self._keyring_username(provider_name),
                serialized,
            )
        except Exception:
            return False

        return True

    def _delete_from_keyring(self, provider_name: str) -> None:
        try:
            keyring.delete_password(_SERVICE_NAME, self._keyring_username(provider_name))
        except Exception:
            return

    def _load_from_file(self, provider_name: str) -> OAuthTokenSet | None:
        path = self._file_path(provider_name)
        try:
            data = path.read_text("utf-8")
        except FileNotFoundError:
            return None
        except OSError:
            return None

        try:
            return OAuthTokenSet.model_validate_json(data)
        except Exception:
            return None

    def _save_to_file(self, provider_name: str, serialized: str) -> None:
        auth_dir = GLOBAL_AUTH_DIR.path
        auth_dir.mkdir(parents=True, exist_ok=True)
        try:
            auth_dir.chmod(0o700)
        except OSError:
            pass

        path = self._file_path(provider_name)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(serialized, encoding="utf-8")
        try:
            tmp_path.chmod(0o600)
        except OSError:
            pass
        tmp_path.replace(path)


class OAuthDeviceClient:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self._timeout = timeout

    async def __aenter__(self) -> OAuthDeviceClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout))
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout))
            self._owns_client = True
        return self._client

    async def start_device_flow(
        self,
        oauth: OAuthConfig,
        *,
        open_browser: bool = True,
    ) -> OAuthDeviceFlowHandle:
        if oauth.device_flow_style == OAuthDeviceFlowStyle.CODEX:
            return await self._start_codex_device_flow(oauth, open_browser=open_browser)
        return await self._start_rfc8628_device_flow(oauth, open_browser=open_browser)

    async def _start_rfc8628_device_flow(
        self,
        oauth: OAuthConfig,
        *,
        open_browser: bool,
    ) -> OAuthDeviceFlowHandle:
        client_id = oauth.resolved_client_id()
        if not client_id:
            raise OAuthAuthError("Missing OAuth client id")
        if not oauth.device_authorization_endpoint:
            raise OAuthAuthError("Missing device authorization endpoint")

        client = self._get_client()
        response = await client.post(
            oauth.device_authorization_endpoint,
            data={
                "client_id": client_id,
                "scope": " ".join(oauth.scopes),
            },
            headers={"Accept": "application/json"},
        )
        payload = _response_json_or_none(response)
        if not response.is_success:
            if _looks_like_openai_auth0_mismatch(
                endpoint=oauth.device_authorization_endpoint,
                response=response,
            ):
                raise OAuthAuthError(
                    "Failed to initiate OAuth device flow: OpenAI account login requires Codex-style endpoints. "
                    "Use `https://auth.openai.com/api/accounts/deviceauth/usercode` for device authorization and "
                    "`https://auth.openai.com/api/accounts/deviceauth/token` for polling."
                )
            raise OAuthAuthError(
                _response_error_message(
                    response,
                    payload,
                    "Failed to initiate OAuth device flow",
                )
            )

        if payload is None:
            raise OAuthAuthError(
                "Invalid OAuth device flow response: expected JSON payload"
            )

        user_code = _payload_str(payload, "user_code")
        verification_uri = _payload_str(payload, "verification_uri")
        device_code = _payload_str(payload, "device_code")
        if not user_code or not verification_uri or not device_code:
            raise OAuthAuthError(
                "Invalid OAuth device flow response: missing one of user_code, verification_uri, or device_code"
            )

        info = OAuthDeviceFlowInfo(
            user_code=user_code,
            verification_uri=verification_uri,
            verification_uri_complete=_payload_optional_str(payload, "verification_uri_complete"),
        )

        if open_browser:
            webbrowser.open(info.verification_uri_complete or info.verification_uri)

        return OAuthDeviceFlowHandle(
            device_code=device_code,
            expires_in=_payload_int(payload, "expires_in", 900),
            interval=_payload_int(payload, "interval", 5),
            info=info,
        )

    async def _start_codex_device_flow(
        self,
        oauth: OAuthConfig,
        *,
        open_browser: bool,
    ) -> OAuthDeviceFlowHandle:
        client_id = oauth.resolved_client_id()
        if not client_id:
            raise OAuthAuthError("Missing OAuth client id")
        if not oauth.device_authorization_endpoint:
            raise OAuthAuthError("Missing device authorization endpoint")

        issuer = _oauth_issuer(oauth)
        verification_uri = f"{issuer}/codex/device" if issuer else ""

        client = self._get_client()
        response = await client.post(
            oauth.device_authorization_endpoint,
            json={"client_id": client_id},
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        payload = _response_json_or_none(response)
        if not response.is_success:
            raise OAuthAuthError(
                _response_error_message(
                    response,
                    payload,
                    "Failed to initiate OAuth device flow",
                )
            )
        if payload is None:
            raise OAuthAuthError(
                "Invalid OAuth device flow response: expected JSON payload"
            )

        user_code = _payload_str(payload, "user_code")
        device_auth_id = _payload_str(payload, "device_auth_id")
        if not user_code or not device_auth_id:
            raise OAuthAuthError(
                "Invalid OAuth device flow response: missing user_code or device_auth_id"
            )

        info = OAuthDeviceFlowInfo(
            user_code=user_code,
            verification_uri=verification_uri,
            verification_uri_complete=verification_uri,
        )

        if open_browser and verification_uri:
            webbrowser.open(verification_uri)

        return OAuthDeviceFlowHandle(
            device_code=device_auth_id,
            expires_in=_payload_int(payload, "expires_in", 900),
            interval=_payload_int(payload, "interval", 5),
            info=info,
        )

    async def wait_for_token(
        self,
        oauth: OAuthConfig,
        handle: OAuthDeviceFlowHandle,
    ) -> OAuthTokenSet:
        if oauth.device_flow_style == OAuthDeviceFlowStyle.CODEX:
            return await self._wait_for_codex_token(oauth, handle)
        return await self._wait_for_rfc8628_token(oauth, handle)

    async def _wait_for_rfc8628_token(
        self,
        oauth: OAuthConfig,
        handle: OAuthDeviceFlowHandle,
    ) -> OAuthTokenSet:
        client_id = oauth.resolved_client_id()
        if not client_id:
            raise OAuthAuthError("Missing OAuth client id")
        if not oauth.token_endpoint:
            raise OAuthAuthError("Missing OAuth token endpoint")

        client = self._get_client()
        elapsed = 0.0
        interval = float(max(handle.interval, 1))

        while elapsed < handle.expires_in:
            await asyncio.sleep(interval)
            elapsed += interval

            response = await client.post(
                oauth.token_endpoint,
                data={
                    "client_id": client_id,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": handle.device_code,
                },
                headers={"Accept": "application/json"},
            )

            payload = _response_json_or_none(response)
            if payload is None:
                raise OAuthAuthError(
                    _response_error_message(
                        response,
                        payload,
                        "OAuth token polling failed: expected JSON payload",
                    )
                )

            if "access_token" in payload:
                return _token_from_payload(payload)

            error = payload.get("error")
            if error == "authorization_pending":
                continue
            if error == "slow_down":
                interval = float(_payload_int(payload, "interval", int(interval) + 5))
                continue
            if error in {"expired_token", "access_denied"}:
                raise OAuthAuthError(f"OAuth device login failed: {error}")

            if not response.is_success:
                raise OAuthAuthError(
                    _response_error_message(response, payload, "OAuth token polling failed")
                )

        raise OAuthAuthError("OAuth device login timed out")

    async def _wait_for_codex_token(
        self,
        oauth: OAuthConfig,
        handle: OAuthDeviceFlowHandle,
    ) -> OAuthTokenSet:
        client_id = oauth.resolved_client_id()
        if not client_id:
            raise OAuthAuthError("Missing OAuth client id")
        if not oauth.device_poll_endpoint:
            raise OAuthAuthError("Missing OAuth device poll endpoint")
        if not oauth.token_endpoint:
            raise OAuthAuthError("Missing OAuth token endpoint")

        client = self._get_client()
        elapsed = 0.0
        interval = float(max(handle.interval, 1))

        while elapsed < handle.expires_in:
            await asyncio.sleep(interval)
            elapsed += interval

            response = await client.post(
                oauth.device_poll_endpoint,
                json={
                    "device_auth_id": handle.device_code,
                    "user_code": handle.info.user_code,
                },
                headers={"Accept": "application/json", "Content-Type": "application/json"},
            )
            payload = _response_json_or_none(response)

            if response.status_code in {403, 404}:
                continue

            if response.status_code == 200:
                if payload is None:
                    raise OAuthAuthError(
                        "OAuth device polling returned invalid response: expected JSON payload"
                    )
                authorization_code = _payload_str(payload, "authorization_code")
                code_verifier = _payload_str(payload, "code_verifier")
                if not authorization_code or not code_verifier:
                    raise OAuthAuthError(
                        "OAuth device polling response missing authorization_code or code_verifier"
                    )
                return await self._exchange_codex_authorization_code(
                    oauth,
                    client_id=client_id,
                    authorization_code=authorization_code,
                    code_verifier=code_verifier,
                )

            raise OAuthAuthError(
                _response_error_message(response, payload, "OAuth device polling failed")
            )

        raise OAuthAuthError("OAuth device login timed out")

    async def _exchange_codex_authorization_code(
        self,
        oauth: OAuthConfig,
        *,
        client_id: str,
        authorization_code: str,
        code_verifier: str,
    ) -> OAuthTokenSet:
        redirect_uri = oauth.device_redirect_uri or f"{_oauth_issuer(oauth)}/deviceauth/callback"

        body = (
            "grant_type=authorization_code"
            f"&code={quote(authorization_code, safe='')}"
            f"&redirect_uri={quote(redirect_uri, safe='')}"
            f"&client_id={quote(client_id, safe='')}"
            f"&code_verifier={quote(code_verifier, safe='')}"
        )

        client = self._get_client()
        response = await client.post(
            oauth.token_endpoint,
            content=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        payload = _response_json_or_none(response)
        if not response.is_success:
            raise OAuthAuthError(
                _response_error_message(
                    response,
                    payload,
                    "OAuth authorization code exchange failed",
                )
            )
        if payload is None:
            raise OAuthAuthError(
                "OAuth authorization code exchange returned invalid response"
            )

        return _token_from_payload(payload)

    async def refresh_access_token(
        self,
        oauth: OAuthConfig,
        refresh_token: str,
    ) -> OAuthTokenSet:
        client_id = oauth.resolved_client_id()
        if not client_id:
            raise OAuthAuthError("Missing OAuth client id")
        if not oauth.token_endpoint:
            raise OAuthAuthError("Missing OAuth token endpoint")

        client = self._get_client()
        response = await client.post(
            oauth.token_endpoint,
            data={
                "client_id": client_id,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            headers={"Accept": "application/json"},
        )

        payload = _response_json_or_none(response)
        if not response.is_success:
            msg = _response_error_message(response, payload, "OAuth refresh failed")
            raise OAuthAuthError(f"OAuth refresh failed: {msg}")
        if payload is None:
            raise OAuthAuthError("OAuth refresh failed: expected JSON payload")

        refreshed = _token_from_payload(payload)
        if not refreshed.refresh_token:
            refreshed.refresh_token = refresh_token
        return refreshed


class SupportsOAuthRefreshClient(Protocol):
    async def refresh_access_token(
        self,
        oauth: OAuthConfig,
        refresh_token: str,
    ) -> OAuthTokenSet: ...


@dataclass
class ResolvedProviderAuth:
    token: str | None
    extra_headers: dict[str, str]
    auth_type: ProviderAuthType


class ProviderAuthResolver:
    def __init__(
        self,
        *,
        token_store: OAuthTokenStore | None = None,
        oauth_client: SupportsOAuthRefreshClient | None = None,
    ) -> None:
        self._token_store = token_store or OAuthTokenStore()
        self._oauth_client = oauth_client or OAuthDeviceClient()

    async def resolve(self, provider: ProviderConfig) -> ResolvedProviderAuth:
        match provider.auth.type:
            case ProviderAuthType.API_KEY:
                token = (
                    os.getenv(provider.api_key_env_var) if provider.api_key_env_var else None
                )
                return ResolvedProviderAuth(
                    token=token,
                    extra_headers={},
                    auth_type=ProviderAuthType.API_KEY,
                )
            case ProviderAuthType.NONE:
                return ResolvedProviderAuth(
                    token=None,
                    extra_headers={},
                    auth_type=ProviderAuthType.NONE,
                )
            case ProviderAuthType.OAUTH:
                return await self._resolve_oauth(provider)

    async def force_refresh(self, provider: ProviderConfig) -> ResolvedProviderAuth:
        oauth = provider.auth.oauth
        if oauth is None:
            raise AuthRequiredError(f"Provider '{provider.name}' is missing OAuth config")

        token_set = self._token_store.load(provider.name)
        if token_set is None or not token_set.refresh_token:
            raise AuthRequiredError(
                f"No refresh token found for provider '{provider.name}'. Run `vibe --auth login --provider {provider.name}`."
            )

        refreshed = await self._oauth_client.refresh_access_token(
            oauth,
            token_set.refresh_token,
        )
        self._token_store.save(provider.name, refreshed)
        return _context_from_token(provider, refreshed)

    async def _resolve_oauth(self, provider: ProviderConfig) -> ResolvedProviderAuth:
        oauth = provider.auth.oauth
        if oauth is None:
            raise AuthRequiredError(f"Provider '{provider.name}' is missing OAuth config")

        token_set = self._token_store.load(provider.name)
        if token_set is None:
            raise AuthRequiredError(
                f"Missing OAuth token for provider '{provider.name}'. Run `vibe --auth login --provider {provider.name}`."
            )

        if token_set.is_expired():
            if not token_set.refresh_token:
                self._token_store.delete(provider.name)
                raise AuthRequiredError(
                    f"OAuth token expired for provider '{provider.name}'. Run `vibe --auth login --provider {provider.name}`."
                )

            try:
                token_set = await self._oauth_client.refresh_access_token(
                    oauth,
                    token_set.refresh_token,
                )
            except OAuthAuthError as exc:
                self._token_store.delete(provider.name)
                raise AuthRequiredError(
                    f"OAuth refresh failed for provider '{provider.name}'. Run `vibe --auth login --provider {provider.name}`."
                ) from exc

            self._token_store.save(provider.name, token_set)

        return _context_from_token(provider, token_set)


class SupportsProviderAuthResolver(Protocol):
    async def resolve(self, provider: ProviderConfig) -> ResolvedProviderAuth: ...

    async def force_refresh(self, provider: ProviderConfig) -> ResolvedProviderAuth: ...


def _response_json_or_none(response: httpx.Response) -> dict[str, object] | None:
    try:
        payload = response.json()
    except ValueError:
        return None

    if isinstance(payload, dict):
        return payload
    return None


def _payload_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _payload_optional_str(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _payload_int(payload: dict[str, object], key: str, default: int) -> int:
    value = payload.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _oauth_issuer(oauth: OAuthConfig) -> str:
    for candidate in (
        oauth.authorization_endpoint,
        oauth.device_authorization_endpoint,
        oauth.token_endpoint,
    ):
        if not candidate:
            continue
        parsed = urlsplit(candidate)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
    return ""


def _truncate(value: str, *, max_chars: int = 240) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars]}..."


def _response_error_message(
    response: httpx.Response,
    payload: dict[str, object] | None,
    prefix: str,
) -> str:
    if payload:
        description = _payload_str(payload, "error_description")
        error = _payload_str(payload, "error")
        if description:
            return f"{prefix}: {description}"
        if error:
            return f"{prefix}: {error}"

    status = response.status_code
    reason = response.reason_phrase or ""
    snippet = _truncate(response.text)
    return f"{prefix}: HTTP {status} {reason}. Response snippet: {snippet}"


def _looks_like_openai_auth0_mismatch(*, endpoint: str, response: httpx.Response) -> bool:
    if "auth0.openai.com/oauth/device/code" not in endpoint:
        return False
    if response.status_code != 403:
        return False
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type.lower():
        return True
    return "<html" in response.text.lower()


def _context_from_token(
    provider: ProviderConfig,
    token_set: OAuthTokenSet,
) -> ResolvedProviderAuth:
    oauth = provider.auth.oauth
    if oauth is None:
        return ResolvedProviderAuth(
            token=token_set.access_token,
            extra_headers={},
            auth_type=ProviderAuthType.OAUTH,
        )

    headers: dict[str, str] = {}
    if oauth.account_header_name:
        account_id = ""
        if oauth.account_id_env_var:
            account_id = os.getenv(oauth.account_id_env_var, "")
        if not account_id:
            account_id = token_set.account_id
        if account_id:
            headers[oauth.account_header_name] = account_id

    return ResolvedProviderAuth(
        token=token_set.access_token,
        extra_headers=headers,
        auth_type=ProviderAuthType.OAUTH,
    )


def _token_from_payload(payload: dict[str, object]) -> OAuthTokenSet:
    access_token = str(payload.get("access_token") or "")
    if not access_token:
        raise OAuthAuthError("Token response does not include access_token")

    refresh_token = str(payload.get("refresh_token") or "")
    token_type = str(payload.get("token_type") or "Bearer")
    scope = str(payload.get("scope") or "")
    expires_in_raw = payload.get("expires_in")

    expires_at: int | None = None
    if isinstance(expires_in_raw, (int, float, str)):
        try:
            expires_in = int(float(expires_in_raw))
            expires_at = int(time.time()) + expires_in
        except Exception:
            expires_at = None

    account_id = _extract_account_id_from_token_payload(payload)

    return OAuthTokenSet(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type=token_type,
        expires_at=expires_at,
        scope=scope,
        account_id=account_id,
    )


def _extract_account_id_from_token_payload(payload: dict[str, object]) -> str:
    account_id = _payload_str(payload, "account_id")
    if account_id:
        return account_id

    for token_key in ("id_token", "access_token"):
        token_value = _payload_str(payload, token_key)
        if not token_value:
            continue

        claims = _decode_jwt_claims(token_value)
        if claims is None:
            continue

        extracted = _extract_account_id_from_claims(claims)
        if extracted:
            return extracted

    return ""


def _decode_jwt_claims(token: str) -> dict[str, object] | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None

    payload_segment = parts[1]
    padding_len = (-len(payload_segment)) % 4
    if padding_len:
        payload_segment += "=" * padding_len

    try:
        decoded = base64.urlsafe_b64decode(payload_segment.encode("utf-8")).decode(
            "utf-8"
        )
        payload = json.loads(decoded)
    except Exception:
        return None

    if isinstance(payload, dict):
        return payload
    return None


def _extract_account_id_from_claims(claims: dict[str, object]) -> str:
    direct = _payload_str(claims, "chatgpt_account_id")
    if direct:
        return direct

    auth_claim = claims.get("https://api.openai.com/auth")
    if isinstance(auth_claim, dict):
        nested = _payload_str(auth_claim, "chatgpt_account_id")
        if nested:
            return nested

    organizations = claims.get("organizations")
    if isinstance(organizations, list) and organizations:
        first_org = organizations[0]
        if isinstance(first_org, dict):
            org_id = _payload_str(first_org, "id")
            if org_id:
                return org_id

    return ""
