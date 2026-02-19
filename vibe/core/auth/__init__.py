from __future__ import annotations

from vibe.core.auth.crypto import EncryptedPayload, decrypt, encrypt
from vibe.core.auth.github import GitHubAuthProvider
from vibe.core.auth.oauth import (
    AuthRequiredError,
    OAuthAuthError,
    OAuthDeviceClient,
    OAuthTokenSet,
    OAuthTokenStore,
    ProviderAuthResolver,
    ResolvedProviderAuth,
    SupportsOAuthRefreshClient,
    SupportsProviderAuthResolver,
)

__all__ = [
    "AuthRequiredError",
    "EncryptedPayload",
    "GitHubAuthProvider",
    "OAuthAuthError",
    "OAuthDeviceClient",
    "OAuthTokenSet",
    "OAuthTokenStore",
    "ProviderAuthResolver",
    "ResolvedProviderAuth",
    "SupportsOAuthRefreshClient",
    "SupportsProviderAuthResolver",
    "decrypt",
    "encrypt",
]
