from __future__ import annotations

import httpx

from vibe.core.auth.oauth import SupportsProviderAuthResolver
from vibe.core.config import ProviderConfig
from vibe.core.llm.backend.generic import GenericBackend


class OpenAIChatGPTBackend(GenericBackend):
    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        provider: ProviderConfig,
        timeout: float = 720.0,
        auth_resolver: SupportsProviderAuthResolver | None = None,
    ) -> None:
        adjusted_provider = provider
        if adjusted_provider.api_style == "openai":
            adjusted_provider = adjusted_provider.model_copy(
                update={"api_style": "openai-chatgpt-codex"}
            )

        super().__init__(
            client=client,
            provider=adjusted_provider,
            timeout=timeout,
            auth_resolver=auth_resolver,
        )
