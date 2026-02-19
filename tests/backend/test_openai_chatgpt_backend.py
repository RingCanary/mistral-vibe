from __future__ import annotations

import json

import httpx
import pytest
import respx

from vibe.core.auth.oauth import ResolvedProviderAuth
from vibe.core.config import (
    Backend,
    ModelConfig,
    OAuthConfig,
    ProviderAuthConfig,
    ProviderAuthType,
    ProviderConfig,
)
from vibe.core.llm.backend.openai_chatgpt import OpenAIChatGPTBackend
from vibe.core.types import AvailableFunction, AvailableTool, LLMChunk, LLMMessage, Role


class StaticAuthResolver:
    def __init__(
        self,
        *,
        initial_token: str,
        refreshed_token: str | None = None,
        account_id: str = "",
    ) -> None:
        self._initial_token = initial_token
        self._refreshed_token = refreshed_token
        self._account_id = account_id
        self.force_refresh_calls = 0

    async def resolve(self, provider: ProviderConfig) -> ResolvedProviderAuth:
        _ = provider
        headers = {"ChatGPT-Account-Id": self._account_id} if self._account_id else {}
        return ResolvedProviderAuth(
            token=self._initial_token,
            extra_headers=headers,
            auth_type=ProviderAuthType.OAUTH,
        )

    async def force_refresh(self, provider: ProviderConfig) -> ResolvedProviderAuth:
        _ = provider
        self.force_refresh_calls += 1
        return ResolvedProviderAuth(
            token=self._refreshed_token or self._initial_token,
            extra_headers={"ChatGPT-Account-Id": self._account_id}
            if self._account_id
            else {},
            auth_type=ProviderAuthType.OAUTH,
        )


def _chatgpt_provider() -> ProviderConfig:
    return ProviderConfig(
        name="openai-chatgpt",
        api_base="https://chatgpt.com/backend-api/codex",
        api_style="openai-chatgpt-codex",
        api_key_env_var="",
        backend=Backend.OPENAI_CHATGPT,
        auth=ProviderAuthConfig(
            type=ProviderAuthType.OAUTH,
            oauth=OAuthConfig(account_header_name="ChatGPT-Account-Id"),
        ),
    )


@pytest.mark.asyncio
async def test_openai_chatgpt_complete_parses_response_and_sets_headers() -> None:
    with respx.mock(base_url="https://chatgpt.com") as mock_api:
        route = mock_api.post("/backend-api/codex/responses").mock(
            return_value=httpx.Response(
                status_code=200,
                json={
                    "id": "resp_123",
                    "object": "response",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "hi there"}],
                        },
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "run_bash",
                            "arguments": '{"command":"ls"}',
                        },
                    ],
                    "usage": {"input_tokens": 12, "output_tokens": 4},
                },
            )
        )

        provider = _chatgpt_provider()
        backend = OpenAIChatGPTBackend(
            provider=provider,
            auth_resolver=StaticAuthResolver(
                initial_token="oauth-access-token",
                account_id="acc_123",
            ),
        )
        model = ModelConfig(
            name="gpt-5.3-codex",
            provider="openai-chatgpt",
            alias="codex",
        )

        result = await backend.complete(
            model=model,
            messages=[LLMMessage(role=Role.user, content="Say hi")],
            temperature=0.2,
            tools=[
                AvailableTool(
                    function=AvailableFunction(
                        name="run_bash",
                        description="Run shell command",
                        parameters={"type": "object", "properties": {}},
                    )
                )
            ],
            max_tokens=128,
            tool_choice="auto",
            extra_headers={"user-agent": "mistral-vibe-test"},
        )

        assert route.called
        request = route.calls.last.request
        assert request.headers["authorization"] == "Bearer oauth-access-token"
        assert request.headers["chatgpt-account-id"] == "acc_123"
        assert request.headers["openai-beta"] == "responses=experimental"
        assert request.headers["originator"] == "vibe"
        assert request.headers["user-agent"] == "mistral-vibe-test"

        payload = json.loads(request.content)
        assert payload["model"] == "gpt-5.3-codex"
        assert payload["stream"] is False
        assert payload["max_output_tokens"] == 128
        assert "temperature" not in payload
        assert payload["tools"][0]["name"] == "run_bash"

        assert result.message.content == "hi there"
        assert result.message.tool_calls is not None
        assert result.message.tool_calls[0].id == "call_1"
        assert result.message.tool_calls[0].function.name == "run_bash"
        assert result.message.tool_calls[0].function.arguments == '{"command":"ls"}'
        assert result.usage is not None
        assert result.usage.prompt_tokens == 12
        assert result.usage.completion_tokens == 4


@pytest.mark.asyncio
async def test_openai_chatgpt_streaming_parses_text_and_usage() -> None:
    with respx.mock(base_url="https://chatgpt.com") as mock_api:
        chunks = [
            rb'data: {"type":"response.output_text.delta","delta":"Hello"}',
            rb'data: {"type":"response.output_text.delta","delta":" world"}',
            rb'data: {"type":"response.completed","response":{"usage":{"input_tokens":8,"output_tokens":3}}}',
            rb"data: [DONE]",
        ]
        route = mock_api.post("/backend-api/codex/responses").mock(
            return_value=httpx.Response(
                status_code=200,
                stream=httpx.ByteStream(stream=b"\n\n".join(chunks)),
                headers={"Content-Type": "text/event-stream"},
            )
        )

        provider = _chatgpt_provider()
        backend = OpenAIChatGPTBackend(
            provider=provider,
            auth_resolver=StaticAuthResolver(initial_token="oauth-access-token"),
        )
        model = ModelConfig(
            name="gpt-5.3-codex",
            provider="openai-chatgpt",
            alias="codex",
        )

        aggregated = LLMChunk(message=LLMMessage(role=Role.assistant), usage=None)
        async for chunk in backend.complete_streaming(
            model=model,
            messages=[LLMMessage(role=Role.user, content="hello")],
            temperature=0.2,
            tools=None,
            max_tokens=None,
            tool_choice="auto",
            extra_headers=None,
        ):
            aggregated = aggregated + chunk

        assert aggregated.message.content == "Hello world"
        assert aggregated.usage is not None
        assert aggregated.usage.prompt_tokens == 8
        assert aggregated.usage.completion_tokens == 3
        assert route.called
        payload = json.loads(route.calls.last.request.content)
        assert payload["stream"] is True
        assert "temperature" not in payload


@pytest.mark.asyncio
async def test_openai_chatgpt_refreshes_oauth_on_401() -> None:
    with respx.mock(base_url="https://chatgpt.com") as mock_api:
        route = mock_api.post("/backend-api/codex/responses").mock(
            side_effect=[
                httpx.Response(status_code=401, json={"error": "expired"}),
                httpx.Response(
                    status_code=200,
                    json={
                        "id": "resp_456",
                        "object": "response",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "refreshed",
                                    }
                                ],
                            }
                        ],
                        "usage": {"input_tokens": 2, "output_tokens": 1},
                    },
                ),
            ]
        )

        resolver = StaticAuthResolver(
            initial_token="expired-token",
            refreshed_token="fresh-token",
            account_id="acc_123",
        )
        provider = _chatgpt_provider()
        backend = OpenAIChatGPTBackend(provider=provider, auth_resolver=resolver)
        model = ModelConfig(
            name="gpt-5.3-codex",
            provider="openai-chatgpt",
            alias="codex",
        )

        result = await backend.complete(
            model=model,
            messages=[LLMMessage(role=Role.user, content="hello")],
            temperature=0.2,
            tools=None,
            max_tokens=None,
            tool_choice="auto",
            extra_headers=None,
        )

        assert result.message.content == "refreshed"
        assert resolver.force_refresh_calls == 1
        assert len(route.calls) == 2
        assert route.calls[0].request.headers["authorization"] == "Bearer expired-token"
        assert route.calls[1].request.headers["authorization"] == "Bearer fresh-token"
        first_payload = json.loads(route.calls[0].request.content)
        second_payload = json.loads(route.calls[1].request.content)
        assert "temperature" not in first_payload
        assert "temperature" not in second_payload


def test_openai_chatgpt_backend_sets_codex_api_style() -> None:
    provider = ProviderConfig(
        name="openai-chatgpt",
        api_base="https://chatgpt.com/backend-api/codex",
        backend=Backend.OPENAI_CHATGPT,
        api_style="openai",
        api_key_env_var="",
        auth=ProviderAuthConfig(type=ProviderAuthType.OAUTH),
    )

    backend = OpenAIChatGPTBackend(provider=provider)

    assert backend._provider.api_style == "openai-chatgpt-codex"
