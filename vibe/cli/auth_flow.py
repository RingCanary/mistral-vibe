from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import math

from rich import print as rprint

from vibe.core.auth.oauth import OAuthAuthError, OAuthDeviceClient, OAuthTokenStore
from vibe.core.config import ProviderAuthType, ProviderConfig, VibeConfig


def _resolve_provider(args: argparse.Namespace, config: VibeConfig) -> ProviderConfig:
    if args.auth_provider:
        for provider in config.providers:
            if provider.name == args.auth_provider:
                return provider
        raise ValueError(f"Provider '{args.auth_provider}' not found in config")

    active_model = config.get_active_model()
    return config.get_provider_for_model(active_model)


def _format_expiry(expires_at: int | None) -> str:
    if expires_at is None:
        return "unknown"

    dt = datetime.fromtimestamp(expires_at, tz=UTC)
    return dt.isoformat()


async def _login_oauth_provider(provider: ProviderConfig, no_browser: bool) -> None:
    oauth = provider.auth.oauth
    if oauth is None:
        raise RuntimeError(f"Provider '{provider.name}' is missing OAuth configuration")

    token_store = OAuthTokenStore()

    async with OAuthDeviceClient() as oauth_client:
        handle = await oauth_client.start_device_flow(
            oauth,
            open_browser=not no_browser,
        )

        rprint(
            "[cyan]Complete sign-in in your browser:[/] "
            f"{handle.info.verification_uri_complete or handle.info.verification_uri}"
        )
        rprint(f"[cyan]Code:[/] {handle.info.user_code}")
        if no_browser:
            rprint("[dim]Browser auto-open disabled (--no-browser).[/]")

        token = await oauth_client.wait_for_token(oauth, handle)
        token_store.save(provider.name, token)

    rprint(
        f"[green]OAuth login succeeded for provider '{provider.name}'.[/] "
        f"Token expiry: {_format_expiry(token.expires_at)}"
    )


def _status_oauth_provider(provider: ProviderConfig) -> None:
    token_store = OAuthTokenStore()
    token = token_store.load(provider.name)

    if token is None:
        rprint(
            f"[yellow]No OAuth token stored for provider '{provider.name}'.[/] "
            f"Run `vibe --auth login --provider {provider.name}`."
        )
        return

    if token.expires_at is None:
        rprint(
            f"[green]OAuth token present for provider '{provider.name}'.[/] "
            "Expiry: unknown"
        )
        return

    remaining = token.expires_at - int(datetime.now(tz=UTC).timestamp())
    if remaining <= 0:
        rprint(
            f"[yellow]OAuth token expired for provider '{provider.name}'.[/] "
            f"Expired at: {_format_expiry(token.expires_at)}"
        )
        return

    remaining_min = math.floor(remaining / 60)
    rprint(
        f"[green]OAuth token active for provider '{provider.name}'.[/] "
        f"Expires in ~{remaining_min} min at {_format_expiry(token.expires_at)}"
    )


def _logout_oauth_provider(provider: ProviderConfig) -> None:
    token_store = OAuthTokenStore()
    token_store.delete(provider.name)
    rprint(f"[green]OAuth token removed for provider '{provider.name}'.[/]")


def run_auth_flow(args: argparse.Namespace, config: VibeConfig) -> int:
    provider = _resolve_provider(args, config)

    if provider.auth.type != ProviderAuthType.OAUTH:
        if provider.auth.type == ProviderAuthType.API_KEY:
            rprint(
                f"[yellow]Provider '{provider.name}' uses API key auth.[/] "
                f"Set {provider.api_key_env_var} instead of OAuth."
            )
            return 1

        rprint(f"[yellow]Provider '{provider.name}' has auth type 'none'.[/]")
        return 1

    if args.auth == "status":
        _status_oauth_provider(provider)
        return 0

    if args.auth == "logout":
        _logout_oauth_provider(provider)
        return 0

    if args.auth == "login":
        try:
            asyncio.run(_login_oauth_provider(provider, args.no_browser))
            return 0
        except OAuthAuthError as exc:
            rprint(f"[red]OAuth login failed:[/] {exc}")
            return 1

    rprint(f"[red]Unknown auth command:[/] {args.auth}")
    return 1
