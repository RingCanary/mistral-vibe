"""Test data for this module was generated using real LLM provider API responses,
with responses simplified and formatted to make them readable and maintainable.

To update or modify test parameters:
1. Make actual API calls to the target providers
2. Use the raw API responses as a base for updating test data
3. Simplify only where necessary for readability while preserving core structure

The closer test data remains to real API responses, the more reliable and accurate
the tests will be. Always prefer real API data over manually constructed examples.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from tests.backend.data import Chunk, JsonResponse, ResultData, Url
from tests.backend.data.fireworks import (
    SIMPLE_CONVERSATION_PARAMS as FIREWORKS_SIMPLE_CONVERSATION_PARAMS,
    STREAMED_SIMPLE_CONVERSATION_PARAMS as FIREWORKS_STREAMED_SIMPLE_CONVERSATION_PARAMS,
    STREAMED_TOOL_CONVERSATION_PARAMS as FIREWORKS_STREAMED_TOOL_CONVERSATION_PARAMS,
    TOOL_CONVERSATION_PARAMS as FIREWORKS_TOOL_CONVERSATION_PARAMS,
)
from tests.backend.data.mistral import (
    SIMPLE_CONVERSATION_PARAMS as MISTRAL_SIMPLE_CONVERSATION_PARAMS,
    STREAMED_SIMPLE_CONVERSATION_PARAMS as MISTRAL_STREAMED_SIMPLE_CONVERSATION_PARAMS,
    STREAMED_TOOL_CONVERSATION_PARAMS as MISTRAL_STREAMED_TOOL_CONVERSATION_PARAMS,
    TOOL_CONVERSATION_PARAMS as MISTRAL_TOOL_CONVERSATION_PARAMS,
)
from vibe.core.auth.oauth import ResolvedProviderAuth
from vibe.core.config import (
    Backend,
    ModelConfig,
    OAuthConfig,
    ProviderAuthConfig,
    ProviderAuthType,
    ProviderConfig,
)
from vibe.core.llm.backend.factory import BACKEND_FACTORY
from vibe.core.llm.backend.generic import GenericBackend
from vibe.core.llm.backend.mistral import MistralBackend
from vibe.core.llm.exceptions import BackendError
from vibe.core.llm.types import BackendLike
from vibe.core.types import LLMChunk, LLMMessage, Role, ToolCall
from vibe.core.utils import get_user_agent


OPENAI_COMPATIBLE_BASE_URLS = ("https://api.openai.com", "https://api.z.ai")


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
            extra_headers={},
            auth_type=ProviderAuthType.OAUTH,
        )


class TestBackend:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "base_url,json_response,result_data",
        [
            *FIREWORKS_SIMPLE_CONVERSATION_PARAMS,
            *FIREWORKS_TOOL_CONVERSATION_PARAMS,
            *MISTRAL_SIMPLE_CONVERSATION_PARAMS,
            *MISTRAL_TOOL_CONVERSATION_PARAMS,
            *[
                (base_url, entry[1], entry[2])
                for base_url in OPENAI_COMPATIBLE_BASE_URLS
                for entry in FIREWORKS_SIMPLE_CONVERSATION_PARAMS
            ],
            *[
                (base_url, entry[1], entry[2])
                for base_url in OPENAI_COMPATIBLE_BASE_URLS
                for entry in FIREWORKS_TOOL_CONVERSATION_PARAMS
            ],
        ],
    )
    async def test_backend_complete(
        self, base_url: Url, json_response: JsonResponse, result_data: ResultData
    ):
        with respx.mock(base_url=base_url) as mock_api:
            mock_api.post("/v1/chat/completions").mock(
                return_value=httpx.Response(status_code=200, json=json_response)
            )
            provider = ProviderConfig(
                name="provider_name",
                api_base=f"{base_url}/v1",
                api_key_env_var="API_KEY",
            )

            BackendClasses = [
                GenericBackend,
                *([MistralBackend] if base_url == "https://api.mistral.ai" else []),
            ]
            for BackendClass in BackendClasses:
                backend: BackendLike = BackendClass(provider=provider)
                model = ModelConfig(
                    name="model_name", provider="provider_name", alias="model_alias"
                )
                messages = [LLMMessage(role=Role.user, content="Just say hi")]

                result = await backend.complete(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    tools=None,
                    max_tokens=None,
                    tool_choice=None,
                    extra_headers=None,
                )

                assert result.message.content == result_data["message"]
                assert result.usage is not None
                assert (
                    result.usage.prompt_tokens == result_data["usage"]["prompt_tokens"]
                )
                assert (
                    result.usage.completion_tokens
                    == result_data["usage"]["completion_tokens"]
                )

                if result.message.tool_calls is None:
                    return

                assert len(result.message.tool_calls) == len(result_data["tool_calls"])
                for i, tool_call in enumerate[ToolCall](result.message.tool_calls):
                    assert (
                        tool_call.function.name == result_data["tool_calls"][i]["name"]
                    )
                    assert (
                        tool_call.function.arguments
                        == result_data["tool_calls"][i]["arguments"]
                    )
                    assert tool_call.index == result_data["tool_calls"][i]["index"]

    @pytest.mark.asyncio
    async def test_backend_complete_with_zai_api_base_path(self) -> None:
        with respx.mock(base_url="https://api.z.ai") as mock_api:
            route = mock_api.post("/api/paas/v4/chat/completions").mock(
                return_value=httpx.Response(
                    status_code=200,
                    json={
                        "id": "chatcmpl-zai",
                        "object": "chat.completion",
                        "created": 1730000000,
                        "model": "glm-4.5",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": "hi"},
                            }
                        ],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                    },
                )
            )
            provider = ProviderConfig(
                name="zai",
                api_base="https://api.z.ai/api/paas/v4",
                api_key_env_var="ZAI_API_KEY",
            )
            backend = GenericBackend(provider=provider)
            model = ModelConfig(name="glm-4.5", provider="zai", alias="zai-glm-45")
            messages = [LLMMessage(role=Role.user, content="Just say hi")]

            result = await backend.complete(
                model=model,
                messages=messages,
                temperature=0.2,
                tools=None,
                max_tokens=None,
                tool_choice=None,
                extra_headers=None,
            )

            assert route.called
            assert result.message.content == "hi"
            assert result.usage is not None
            assert result.usage.prompt_tokens == 5
            assert result.usage.completion_tokens == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "base_url,chunks,result_data",
        [
            *FIREWORKS_STREAMED_SIMPLE_CONVERSATION_PARAMS,
            *FIREWORKS_STREAMED_TOOL_CONVERSATION_PARAMS,
            *MISTRAL_STREAMED_SIMPLE_CONVERSATION_PARAMS,
            *MISTRAL_STREAMED_TOOL_CONVERSATION_PARAMS,
            *[
                (base_url, entry[1], entry[2])
                for base_url in OPENAI_COMPATIBLE_BASE_URLS
                for entry in FIREWORKS_STREAMED_SIMPLE_CONVERSATION_PARAMS
            ],
            *[
                (base_url, entry[1], entry[2])
                for base_url in OPENAI_COMPATIBLE_BASE_URLS
                for entry in FIREWORKS_STREAMED_TOOL_CONVERSATION_PARAMS
            ],
        ],
    )
    async def test_backend_complete_streaming(
        self, base_url: Url, chunks: list[Chunk], result_data: list[ResultData]
    ):
        with respx.mock(base_url=base_url) as mock_api:
            mock_api.post("/v1/chat/completions").mock(
                return_value=httpx.Response(
                    status_code=200,
                    stream=httpx.ByteStream(stream=b"\n\n".join(chunks)),
                    headers={"Content-Type": "text/event-stream"},
                )
            )
            provider = ProviderConfig(
                name="provider_name",
                api_base=f"{base_url}/v1",
                api_key_env_var="API_KEY",
            )
            BackendClasses = [
                GenericBackend,
                *([MistralBackend] if base_url == "https://api.mistral.ai" else []),
            ]
            for BackendClass in BackendClasses:
                backend: BackendLike = BackendClass(provider=provider)
                model = ModelConfig(
                    name="model_name", provider="provider_name", alias="model_alias"
                )

                messages = [
                    LLMMessage(role=Role.user, content="List files in current dir")
                ]

                results: list[LLMChunk] = []
                async for result in backend.complete_streaming(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    tools=None,
                    max_tokens=None,
                    tool_choice=None,
                    extra_headers=None,
                ):
                    results.append(result)

                for result, expected_result in zip(results, result_data, strict=True):
                    assert result.message.content == expected_result["message"]
                    assert result.usage is not None
                    assert (
                        result.usage.prompt_tokens
                        == expected_result["usage"]["prompt_tokens"]
                    )
                    assert (
                        result.usage.completion_tokens
                        == expected_result["usage"]["completion_tokens"]
                    )

                    if result.message.tool_calls is None:
                        continue

                    for i, tool_call in enumerate(result.message.tool_calls):
                        assert (
                            tool_call.function.name
                            == expected_result["tool_calls"][i]["name"]
                        )
                        assert (
                            tool_call.function.arguments
                            == expected_result["tool_calls"][i]["arguments"]
                        )
                        assert (
                            tool_call.index == expected_result["tool_calls"][i]["index"]
                        )

    @pytest.mark.asyncio
    async def test_backend_complete_streaming_sse_parsing_regressions(self) -> None:
        with respx.mock(base_url="https://api.example.com") as mock_api:
            stream_bytes = b"\n".join(
                [
                    b": keep-alive",
                    b"",
                    b'data:{"choices":[{"delta":{"role":"assistant","content":"Hello"}}],"usage":{"prompt_tokens":0,"completion_tokens":0}}',
                    b"",
                    b"event: message",
                    b'data: {"choices":',
                    b'data: [{"delta": {"content": " world"}}],',
                    b'data: "usage": {"prompt_tokens": 1, "completion_tokens": 1}}',
                    b"",
                    b"data: [DONE]",
                    b"",
                ]
            )
            mock_api.post("/v1/chat/completions").mock(
                return_value=httpx.Response(
                    status_code=200,
                    stream=httpx.ByteStream(stream=stream_bytes),
                    headers={"Content-Type": "text/event-stream"},
                )
            )
            provider = ProviderConfig(
                name="provider_name",
                api_base="https://api.example.com/v1",
                api_key_env_var="API_KEY",
            )
            backend = GenericBackend(provider=provider)
            model = ModelConfig(
                name="model_name", provider="provider_name", alias="model_alias"
            )

            results: list[LLMChunk] = []
            async for result in backend.complete_streaming(
                model=model,
                messages=[LLMMessage(role=Role.user, content="hello")],
                temperature=0.2,
                tools=None,
                max_tokens=None,
                tool_choice=None,
                extra_headers=None,
            ):
                results.append(result)

            assert [chunk.message.content for chunk in results] == ["Hello", " world"]
            assert [
                (chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
                for chunk in results
            ] == [(0, 0), (1, 1)]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "base_url,backend_class,response",
        [
            (
                "https://api.fireworks.ai",
                GenericBackend,
                httpx.Response(status_code=500, text="Internal Server Error"),
            ),
            (
                "https://api.fireworks.ai",
                GenericBackend,
                httpx.Response(status_code=429, text="Rate Limit Exceeded"),
            ),
            (
                "https://api.mistral.ai",
                MistralBackend,
                httpx.Response(status_code=500, text="Internal Server Error"),
            ),
            (
                "https://api.mistral.ai",
                MistralBackend,
                httpx.Response(status_code=429, text="Rate Limit Exceeded"),
            ),
            *[
                (base_url, GenericBackend, httpx.Response(status_code=500, text="Internal Server Error"))
                for base_url in OPENAI_COMPATIBLE_BASE_URLS
            ],
            *[
                (base_url, GenericBackend, httpx.Response(status_code=429, text="Rate Limit Exceeded"))
                for base_url in OPENAI_COMPATIBLE_BASE_URLS
            ],
        ],
    )
    async def test_backend_complete_streaming_error(
        self,
        base_url: Url,
        backend_class: type[MistralBackend | GenericBackend],
        response: httpx.Response,
    ):
        with respx.mock(base_url=base_url) as mock_api:
            mock_api.post("/v1/chat/completions").mock(return_value=response)
            provider = ProviderConfig(
                name="provider_name",
                api_base=f"{base_url}/v1",
                api_key_env_var="API_KEY",
            )
            backend = backend_class(provider=provider)
            model = ModelConfig(
                name="model_name", provider="provider_name", alias="model_alias"
            )
            messages = [LLMMessage(role=Role.user, content="Just say hi")]
            with pytest.raises(BackendError) as e:
                async for _ in backend.complete_streaming(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    tools=None,
                    max_tokens=None,
                    tool_choice=None,
                    extra_headers=None,
                ):
                    pass
            assert e.value.status == response.status_code
            assert e.value.reason == response.reason_phrase
            assert e.value.parsed_error is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "base_url,provider_name,expected_stream_options",
        [
            ("https://api.fireworks.ai", "fireworks", {"include_usage": True}),
            (
                "https://api.mistral.ai",
                "mistral",
                {"include_usage": True, "stream_tool_calls": True},
            ),
            ("https://api.openai.com", "openai", {"include_usage": True}),
            ("https://api.z.ai", "zai", {"include_usage": True}),
        ],
    )
    async def test_backend_streaming_payload_includes_stream_options(
        self, base_url: Url, provider_name: str, expected_stream_options: dict
    ):
        with respx.mock(base_url=base_url) as mock_api:
            route = mock_api.post("/v1/chat/completions").mock(
                return_value=httpx.Response(
                    status_code=200,
                    stream=httpx.ByteStream(
                        b'data: {"choices": [{"delta": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}\n\ndata: [DONE]\n\n'
                    ),
                    headers={"Content-Type": "text/event-stream"},
                )
            )
            provider = ProviderConfig(
                name=provider_name, api_base=f"{base_url}/v1", api_key_env_var="API_KEY"
            )
            backend = GenericBackend(provider=provider)
            model = ModelConfig(
                name="model_name", provider=provider_name, alias="model_alias"
            )
            messages = [LLMMessage(role=Role.user, content="hi")]

            async for _ in backend.complete_streaming(
                model=model,
                messages=messages,
                temperature=0.2,
                tools=None,
                max_tokens=None,
                tool_choice=None,
                extra_headers=None,
            ):
                pass

            assert route.called
            request = route.calls.last.request
            payload = json.loads(request.content)

            assert payload["stream"] is True
            assert payload["stream_options"] == expected_stream_options

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend_type", [Backend.MISTRAL, Backend.GENERIC])
    async def test_backend_user_agent(self, backend_type: Backend):
        user_agent = get_user_agent(backend_type)
        base_url = "https://api.example.com"
        json_response = {
            "id": "fake_id_1234",
            "created": 1234567890,
            "model": "devstral-latest",
            "usage": {
                "prompt_tokens": 100,
                "total_tokens": 300,
                "completion_tokens": 200,
            },
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "tool_calls": None,
                        "content": "Hey",
                    },
                }
            ],
        }
        with respx.mock(base_url=base_url) as mock_api:
            mock_api.post("/v1/chat/completions").mock(
                return_value=httpx.Response(status_code=200, json=json_response)
            )

            provider = ProviderConfig(
                name="provider_name",
                api_base=f"{base_url}/v1",
                api_key_env_var="API_KEY",
            )
            backend = BACKEND_FACTORY[backend_type](provider=provider)
            model = ModelConfig(
                name="model_name", provider="provider_name", alias="model_alias"
            )
            messages = [LLMMessage(role=Role.user, content="Just say hi")]

            await backend.complete(
                model=model,
                messages=messages,
                temperature=0.2,
                tools=None,
                max_tokens=None,
                tool_choice=None,
                extra_headers={"user-agent": user_agent},
            )

            assert mock_api.calls.last.request.headers["user-agent"] == user_agent

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend_type", [Backend.MISTRAL, Backend.GENERIC])
    async def test_backend_user_agent_when_streaming(self, backend_type: Backend):
        user_agent = get_user_agent(backend_type)

        base_url = "https://api.example.com"
        with respx.mock(base_url=base_url) as mock_api:
            chunks = [
                rb'data: {"id":"fake_id_1234","object":"chat.completion.chunk","created":1234567890,"model":"devstral-latest","choices":[{"index":0,"delta":{"role":"assistant","content":"Hey"},"finish_reason":"stop"}]}'
            ]
            mock_response = httpx.Response(
                status_code=200,
                stream=httpx.ByteStream(stream=b"\n\n".join(chunks)),
                headers={"Content-Type": "text/event-stream"},
            )
            mock_api.post("/v1/chat/completions").mock(return_value=mock_response)

            provider = ProviderConfig(
                name="provider_name",
                api_base=f"{base_url}/v1",
                api_key_env_var="API_KEY",
            )
            backend = BACKEND_FACTORY[backend_type](provider=provider)
            model = ModelConfig(
                name="model_name", provider="provider_name", alias="model_alias"
            )
            messages = [LLMMessage(role=Role.user, content="Just say hi")]

            async for _ in backend.complete_streaming(
                model=model,
                messages=messages,
                temperature=0.2,
                tools=None,
                max_tokens=None,
                tool_choice=None,
                extra_headers={"user-agent": user_agent},
            ):
                pass

            assert mock_api.calls.last.request.headers["user-agent"] == user_agent

    @pytest.mark.asyncio
    async def test_generic_backend_uses_oauth_headers(self) -> None:
        with respx.mock(base_url="https://api.openai.com") as mock_api:
            route = mock_api.post("/v1/chat/completions").mock(
                return_value=httpx.Response(
                    status_code=200,
                    json={
                        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                    },
                )
            )

            provider = ProviderConfig(
                name="openai",
                api_base="https://api.openai.com/v1",
                api_key_env_var="",
                auth=ProviderAuthConfig(
                    type=ProviderAuthType.OAUTH,
                    oauth=OAuthConfig(account_header_name="ChatGPT-Account-Id"),
                ),
            )
            resolver = StaticAuthResolver(
                initial_token="oauth-access-token",
                account_id="account-123",
            )
            backend = GenericBackend(provider=provider, auth_resolver=resolver)
            model = ModelConfig(name="gpt-4o-mini", provider="openai", alias="openai")

            result = await backend.complete(
                model=model,
                messages=[LLMMessage(role=Role.user, content="hello")],
                temperature=0.2,
                tools=None,
                max_tokens=None,
                tool_choice=None,
                extra_headers=None,
            )

            assert result.message.content == "ok"
            assert route.called
            assert route.calls.last.request.headers["authorization"] == "Bearer oauth-access-token"
            assert route.calls.last.request.headers["chatgpt-account-id"] == "account-123"

    @pytest.mark.asyncio
    async def test_generic_backend_refreshes_oauth_on_401(self) -> None:
        with respx.mock(base_url="https://api.openai.com") as mock_api:
            route = mock_api.post("/v1/chat/completions").mock(
                side_effect=[
                    httpx.Response(status_code=401, json={"error": "expired"}),
                    httpx.Response(
                        status_code=200,
                        json={
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": "refreshed",
                                    }
                                }
                            ],
                            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                        },
                    ),
                ]
            )

            provider = ProviderConfig(
                name="openai",
                api_base="https://api.openai.com/v1",
                api_key_env_var="",
                auth=ProviderAuthConfig(
                    type=ProviderAuthType.OAUTH,
                    oauth=OAuthConfig(token_endpoint="https://auth0.openai.com/oauth/token"),
                ),
            )
            resolver = StaticAuthResolver(
                initial_token="expired-access-token",
                refreshed_token="fresh-access-token",
            )
            backend = GenericBackend(provider=provider, auth_resolver=resolver)
            model = ModelConfig(name="gpt-4o-mini", provider="openai", alias="openai")

            result = await backend.complete(
                model=model,
                messages=[LLMMessage(role=Role.user, content="hello")],
                temperature=0.2,
                tools=None,
                max_tokens=None,
                tool_choice=None,
                extra_headers=None,
            )

            assert result.message.content == "refreshed"
            assert resolver.force_refresh_calls == 1
            assert len(route.calls) == 2
            assert route.calls[0].request.headers["authorization"] == "Bearer expired-access-token"
            assert route.calls[1].request.headers["authorization"] == "Bearer fresh-access-token"
