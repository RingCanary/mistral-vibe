from __future__ import annotations

import argparse

from vibe.cli.auth_flow import run_auth_flow
from vibe.core.auth.oauth import OAuthTokenSet, OAuthTokenStore
from vibe.core.config import (
    Backend,
    ModelConfig,
    OAuthConfig,
    ProviderAuthConfig,
    ProviderAuthType,
    ProviderConfig,
    VibeConfig,
)


def test_auth_flow_rejects_api_key_provider() -> None:
    config = VibeConfig(
        active_model="openai-mini",
        providers=[
            ProviderConfig(
                name="openai",
                api_base="https://api.openai.com/v1",
                api_key_env_var="OPENAI_API_KEY",
                backend=Backend.GENERIC,
                auth=ProviderAuthConfig(type=ProviderAuthType.API_KEY),
            )
        ],
        models=[ModelConfig(name="gpt-4o-mini", provider="openai", alias="openai-mini")],
        enable_auto_update=False,
        skip_credentials_validation=True,
    )
    args = argparse.Namespace(
        auth="status",
        auth_provider="openai",
        no_browser=True,
    )

    exit_code = run_auth_flow(args, config)
    assert exit_code == 1


def test_auth_flow_status_for_oauth_provider_without_token() -> None:
    config = VibeConfig(
        active_model="openai-mini",
        providers=[
            ProviderConfig(
                name="openai",
                api_base="https://api.openai.com/v1",
                api_key_env_var="",
                backend=Backend.GENERIC,
                auth=ProviderAuthConfig(
                    type=ProviderAuthType.OAUTH,
                    oauth=OAuthConfig(
                        client_id="client-id",
                        token_endpoint="https://auth0.openai.com/oauth/token",
                        device_authorization_endpoint="https://auth0.openai.com/oauth/device/code",
                        scopes=["openid", "offline_access"],
                    ),
                ),
            )
        ],
        models=[ModelConfig(name="gpt-4o-mini", provider="openai", alias="openai-mini")],
        enable_auto_update=False,
        skip_credentials_validation=True,
    )

    args = argparse.Namespace(
        auth="status",
        auth_provider="openai",
        no_browser=True,
    )
    exit_code = run_auth_flow(args, config)

    assert exit_code == 0


def test_auth_flow_logout_for_oauth_provider() -> None:
    token_store = OAuthTokenStore()
    token_store.save(
        "openai",
        OAuthTokenSet(access_token="access-token", refresh_token="refresh-token"),
    )

    config = VibeConfig(
        active_model="openai-mini",
        providers=[
            ProviderConfig(
                name="openai",
                api_base="https://api.openai.com/v1",
                api_key_env_var="",
                backend=Backend.GENERIC,
                auth=ProviderAuthConfig(
                    type=ProviderAuthType.OAUTH,
                    oauth=OAuthConfig(
                        client_id="client-id",
                        token_endpoint="https://auth0.openai.com/oauth/token",
                        device_authorization_endpoint="https://auth0.openai.com/oauth/device/code",
                        scopes=["openid", "offline_access"],
                    ),
                ),
            )
        ],
        models=[ModelConfig(name="gpt-4o-mini", provider="openai", alias="openai-mini")],
        enable_auto_update=False,
        skip_credentials_validation=True,
    )

    args = argparse.Namespace(
        auth="logout",
        auth_provider="openai",
        no_browser=True,
    )
    exit_code = run_auth_flow(args, config)

    assert exit_code == 0
    assert token_store.load("openai") is None
