from __future__ import annotations

import base64
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import keyring.errors
import pytest

from vibe.core.auth.oauth import (
    AuthRequiredError,
    OAuthAuthError,
    OAuthDeviceClient,
    OAuthDeviceFlowHandle,
    OAuthDeviceFlowInfo,
    OAuthTokenSet,
    OAuthTokenStore,
    ProviderAuthResolver,
)
from vibe.core.config import (
    OAuthConfig,
    OAuthDeviceFlowStyle,
    ProviderAuthConfig,
    ProviderAuthType,
    ProviderConfig,
)


class FakeOAuthClient:
    async def refresh_access_token(
        self,
        oauth: OAuthConfig,
        refresh_token: str,
    ) -> OAuthTokenSet:
        _ = oauth
        return OAuthTokenSet(
            access_token="new-access-token",
            refresh_token=refresh_token,
            expires_at=int(time.time()) + 3600,
        )


def _jwt_with_claims(claims: dict[str, object]) -> str:
    encoded_payload = base64.urlsafe_b64encode(
        json.dumps(claims).encode("utf-8")
    ).decode("utf-8").rstrip("=")
    return f"header.{encoded_payload}.signature"


class TestOAuthDeviceClient:
    @pytest.mark.asyncio
    async def test_codex_start_device_flow_success(self) -> None:
        mock_client = MagicMock()
        response = MagicMock()
        response.is_success = True
        response.json.return_value = {
            "device_auth_id": "device-auth-id",
            "user_code": "ABCD-EFGH",
            "interval": "5",
        }
        mock_client.post = AsyncMock(return_value=response)

        oauth_client = OAuthDeviceClient(client=mock_client)
        oauth = OAuthConfig(
            device_flow_style=OAuthDeviceFlowStyle.CODEX,
            client_id="client-id",
            device_authorization_endpoint="https://auth.openai.com/api/accounts/deviceauth/usercode",
            token_endpoint="https://auth.openai.com/oauth/token",
        )

        with patch("vibe.core.auth.oauth.webbrowser") as mock_browser:
            handle = await oauth_client.start_device_flow(oauth, open_browser=True)

        assert handle.device_code == "device-auth-id"
        assert handle.info.user_code == "ABCD-EFGH"
        mock_browser.open.assert_called_once_with("https://auth.openai.com/codex/device")

    @pytest.mark.asyncio
    async def test_device_flow_failure_shows_concise_html_snippet(self) -> None:
        mock_client = MagicMock()
        response = MagicMock()
        response.is_success = False
        response.status_code = 404
        response.reason_phrase = "Not Found"
        response.text = "<html>" + ("x" * 1200) + "</html>"
        response.json.side_effect = ValueError("not json")
        mock_client.post = AsyncMock(return_value=response)

        oauth_client = OAuthDeviceClient(client=mock_client)
        oauth = OAuthConfig(
            client_id="client-id",
            device_authorization_endpoint="https://example.com/oauth/device/code",
            token_endpoint="https://auth0.openai.com/oauth/token",
        )

        with pytest.raises(OAuthAuthError) as exc:
            await oauth_client.start_device_flow(oauth, open_browser=False)

        msg = str(exc.value)
        assert "HTTP 404" in msg
        assert "Response snippet:" in msg
        assert len(msg) < 420

    @pytest.mark.asyncio
    async def test_auth0_device_flow_returns_migration_hint(self) -> None:
        mock_client = MagicMock()
        response = MagicMock()
        response.is_success = False
        response.status_code = 403
        response.reason_phrase = "Forbidden"
        response.text = "<html><body>forbidden</body></html>"
        response.headers = {"content-type": "text/html"}
        response.json.side_effect = ValueError("not json")
        mock_client.post = AsyncMock(return_value=response)

        oauth_client = OAuthDeviceClient(client=mock_client)
        oauth = OAuthConfig(
            client_id="client-id",
            device_authorization_endpoint="https://auth0.openai.com/oauth/device/code",
            token_endpoint="https://auth0.openai.com/oauth/token",
        )

        with pytest.raises(OAuthAuthError) as exc:
            await oauth_client.start_device_flow(oauth, open_browser=False)

        msg = str(exc.value)
        assert "requires Codex-style endpoints" in msg
        assert "api/accounts/deviceauth/usercode" in msg

    @pytest.mark.asyncio
    async def test_codex_wait_for_token_success(self) -> None:
        mock_client = MagicMock()
        poll_response = MagicMock()
        poll_response.status_code = 200
        poll_response.is_success = True
        poll_response.json.return_value = {
            "authorization_code": "authz-code",
            "code_verifier": "code-verifier",
        }

        token_response = MagicMock()
        token_response.is_success = True
        token_response.status_code = 200
        token_response.json.return_value = {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "id_token": _jwt_with_claims(
                {
                    "https://api.openai.com/auth": {
                        "chatgpt_account_id": "acc_test",
                    }
                }
            ),
            "expires_in": 3600,
        }

        mock_client.post = AsyncMock(side_effect=[poll_response, token_response])

        oauth_client = OAuthDeviceClient(client=mock_client)
        oauth = OAuthConfig(
            device_flow_style=OAuthDeviceFlowStyle.CODEX,
            client_id="client-id",
            device_poll_endpoint="https://auth.openai.com/api/accounts/deviceauth/token",
            token_endpoint="https://auth.openai.com/oauth/token",
            device_redirect_uri="https://auth.openai.com/deviceauth/callback",
        )
        handle = OAuthDeviceFlowHandle(
            device_code="device-auth-id",
            expires_in=900,
            interval=1,
            info=OAuthDeviceFlowInfo(
                user_code="ABCD-EFGH",
                verification_uri="https://auth.openai.com/codex/device",
            ),
        )

        with patch("vibe.core.auth.oauth.asyncio.sleep", new_callable=AsyncMock):
            token = await oauth_client.wait_for_token(oauth, handle)

        assert token.access_token == "access-token"
        assert token.refresh_token == "refresh-token"
        assert token.account_id == "acc_test"


def test_oauth_token_store_file_fallback(config_dir: str) -> None:
    _ = config_dir
    store = OAuthTokenStore()
    token = OAuthTokenSet(
        access_token="access-token",
        refresh_token="refresh-token",
        expires_at=int(time.time()) + 3600,
    )

    with (
        patch(
            "vibe.core.auth.oauth.keyring.set_password",
            side_effect=keyring.errors.KeyringError("no keyring"),
        ),
        patch(
            "vibe.core.auth.oauth.keyring.get_password",
            side_effect=keyring.errors.KeyringError("no keyring"),
        ),
    ):
        store.save("openai", token)
        loaded = store.load("openai")

    assert loaded is not None
    assert loaded.access_token == "access-token"
    assert loaded.refresh_token == "refresh-token"


@pytest.mark.asyncio
async def test_provider_auth_resolver_for_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = ProviderConfig(
        name="openai",
        api_base="https://api.openai.com/v1",
        api_key_env_var="OPENAI_API_KEY",
        auth=ProviderAuthConfig(type=ProviderAuthType.API_KEY),
    )

    resolver = ProviderAuthResolver()
    context = await resolver.resolve(provider)

    assert context.token == "sk-test"
    assert context.auth_type == ProviderAuthType.API_KEY
    assert context.extra_headers == {}


@pytest.mark.asyncio
async def test_provider_auth_resolver_oauth_missing_token_raises() -> None:
    provider = ProviderConfig(
        name="openai-missing",
        api_base="https://api.openai.com/v1",
        api_key_env_var="",
        auth=ProviderAuthConfig(
            type=ProviderAuthType.OAUTH,
            oauth=OAuthConfig(
                client_id="client-id",
                device_authorization_endpoint="https://auth0.openai.com/oauth/device/code",
                token_endpoint="https://auth0.openai.com/oauth/token",
                scopes=["openid", "offline_access"],
            ),
        ),
    )

    resolver = ProviderAuthResolver(token_store=OAuthTokenStore(), oauth_client=FakeOAuthClient())

    with pytest.raises(AuthRequiredError):
        await resolver.resolve(provider)


@pytest.mark.asyncio
async def test_provider_auth_resolver_oauth_refreshes_expired_token(
    config_dir: str,
) -> None:
    _ = config_dir
    provider = ProviderConfig(
        name="openai",
        api_base="https://api.openai.com/v1",
        api_key_env_var="",
        auth=ProviderAuthConfig(
            type=ProviderAuthType.OAUTH,
            oauth=OAuthConfig(
                client_id="client-id",
                token_endpoint="https://auth0.openai.com/oauth/token",
                device_authorization_endpoint="https://auth0.openai.com/oauth/device/code",
                scopes=["openid", "offline_access"],
                account_header_name="ChatGPT-Account-Id",
            ),
        ),
    )

    store = OAuthTokenStore()
    with (
        patch(
            "vibe.core.auth.oauth.keyring.set_password",
            side_effect=keyring.errors.KeyringError("no keyring"),
        ),
        patch(
            "vibe.core.auth.oauth.keyring.get_password",
            side_effect=keyring.errors.KeyringError("no keyring"),
        ),
    ):
        store.save(
            provider.name,
            OAuthTokenSet(
                access_token="old-access-token",
                refresh_token="refresh-token",
                expires_at=int(time.time()) - 5,
            ),
        )

        resolver = ProviderAuthResolver(token_store=store, oauth_client=FakeOAuthClient())
        context = await resolver.resolve(provider)

    assert context.token == "new-access-token"
    assert context.auth_type == ProviderAuthType.OAUTH
