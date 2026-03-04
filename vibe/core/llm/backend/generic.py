from __future__ import annotations

from collections.abc import AsyncGenerator
import json
import types
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import httpx

from vibe.core.auth.oauth import (
    AuthRequiredError,
    ProviderAuthResolver,
    ResolvedProviderAuth,
    SupportsProviderAuthResolver,
)
from vibe.core.config import ProviderAuthType
from vibe.core.llm.backend.anthropic import AnthropicAdapter
from vibe.core.llm.backend.base import APIAdapter, PreparedRequest
from vibe.core.llm.backend.vertex import VertexAnthropicAdapter
from vibe.core.llm.exceptions import BackendErrorBuilder
from vibe.core.llm.message_utils import merge_consecutive_user_messages
from vibe.core.types import (
    AvailableTool,
    FunctionCall,
    LLMChunk,
    LLMMessage,
    LLMUsage,
    Role,
    StrToolChoice,
    ToolCall,
)
from vibe.core.utils import async_generator_retry, async_retry

if TYPE_CHECKING:
    from vibe.core.config import ModelConfig, ProviderConfig


class OpenAIAdapter(APIAdapter):
    endpoint: ClassVar[str] = "/chat/completions"

    def build_payload(
        self,
        model_name: str,
        converted_messages: list[dict[str, Any]],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
    ) -> dict[str, Any]:
        payload = {
            "model": model_name,
            "messages": converted_messages,
            "temperature": temperature,
        }

        if tools:
            payload["tools"] = [tool.model_dump(exclude_none=True) for tool in tools]
        if tool_choice:
            payload["tool_choice"] = (
                tool_choice
                if isinstance(tool_choice, str)
                else tool_choice.model_dump()
            )
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        return payload

    def build_headers(self, api_key: str | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _reasoning_to_api(
        self, msg_dict: dict[str, Any], field_name: str
    ) -> dict[str, Any]:
        if field_name != "reasoning_content" and "reasoning_content" in msg_dict:
            msg_dict[field_name] = msg_dict.pop("reasoning_content")
        return msg_dict

    def _reasoning_from_api(
        self, msg_dict: dict[str, Any], field_name: str
    ) -> dict[str, Any]:
        if field_name != "reasoning_content" and field_name in msg_dict:
            msg_dict["reasoning_content"] = msg_dict.pop(field_name)
        return msg_dict

    def prepare_request(  # noqa: PLR0913
        self,
        *,
        model_name: str,
        messages: list[LLMMessage],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        enable_streaming: bool,
        provider: ProviderConfig,
        api_key: str | None = None,
        thinking: str = "off",
    ) -> PreparedRequest:
        merged_messages = merge_consecutive_user_messages(messages)
        field_name = provider.reasoning_field_name
        converted_messages = [
            self._reasoning_to_api(
                msg.model_dump(exclude_none=True, exclude={"message_id"}), field_name
            )
            for msg in merged_messages
        ]

        payload = self.build_payload(
            model_name, converted_messages, temperature, tools, max_tokens, tool_choice
        )

        if enable_streaming:
            payload["stream"] = True
            stream_options = {"include_usage": True}
            if provider.name == "mistral":
                stream_options["stream_tool_calls"] = True
            payload["stream_options"] = stream_options

        headers = self.build_headers(api_key)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        return PreparedRequest(self.endpoint, headers, body)

    def _parse_message(
        self, data: dict[str, Any], field_name: str
    ) -> LLMMessage | None:
        if data.get("choices"):
            choice = data["choices"][0]
            if "message" in choice:
                msg_dict = self._reasoning_from_api(choice["message"], field_name)
                return LLMMessage.model_validate(msg_dict)
            if "delta" in choice:
                msg_dict = self._reasoning_from_api(choice["delta"], field_name)
                return LLMMessage.model_validate(msg_dict)
            raise ValueError("Invalid response data: missing message or delta")

        if "message" in data:
            msg_dict = self._reasoning_from_api(data["message"], field_name)
            return LLMMessage.model_validate(msg_dict)
        if "delta" in data:
            msg_dict = self._reasoning_from_api(data["delta"], field_name)
            return LLMMessage.model_validate(msg_dict)

        return None

    def parse_response(
        self, data: dict[str, Any], provider: ProviderConfig
    ) -> LLMChunk:
        message = self._parse_message(data, provider.reasoning_field_name)
        if message is None:
            message = LLMMessage(role=Role.assistant, content="")

        usage_data = data.get("usage") or {}
        usage = LLMUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
        )

        return LLMChunk(message=message, usage=usage)


class OpenAIChatGPTCodexAdapter(APIAdapter):
    endpoint: ClassVar[str] = "/responses"

    def build_headers(
        self, api_key: str | None = None, *, enable_streaming: bool
    ) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "OpenAI-Beta": "responses=experimental",
            "originator": "vibe",
        }
        if enable_streaming:
            headers["Accept"] = "text/event-stream"
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def prepare_request(  # noqa: PLR0913
        self,
        *,
        model_name: str,
        messages: list[LLMMessage],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        enable_streaming: bool,
        provider: ProviderConfig,
        api_key: str | None = None,
        thinking: str = "off",
    ) -> PreparedRequest:
        _ = provider
        merged_messages = merge_consecutive_user_messages(messages)
        instructions, input_items = self._to_responses_input(merged_messages)

        payload: dict[str, Any] = {
            "model": model_name,
            "store": False,
            "stream": enable_streaming,
            "input": input_items,
            "parallel_tool_calls": True,
            "text": {"verbosity": "medium"},
        }
        if instructions:
            payload["instructions"] = instructions

        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters,
                    "strict": False,
                }
                for tool in tools
            ]

        if tool_choice:
            if isinstance(tool_choice, str):
                payload["tool_choice"] = tool_choice
            else:
                payload["tool_choice"] = {
                    "type": "function",
                    "name": tool_choice.function.name,
                }

        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens

        _ = temperature

        if thinking != "off":
            payload["reasoning"] = {
                "effort": thinking,
                "summary": "auto",
            }
            payload["include"] = ["reasoning.encrypted_content"]

        headers = self.build_headers(api_key, enable_streaming=enable_streaming)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return PreparedRequest(self.endpoint, headers, body)

    def parse_response(
        self, data: dict[str, Any], provider: ProviderConfig
    ) -> LLMChunk:
        _ = provider
        event_type = self._as_str(data.get("type"))
        if event_type:
            return self._parse_stream_event(data, event_type)
        return self._parse_non_stream_response(data)

    def _to_responses_input(
        self, messages: list[LLMMessage]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        instruction_parts: list[str] = []
        input_items: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.system:
                if msg.content:
                    instruction_parts.append(msg.content)
                continue

            if msg.role == Role.user:
                input_items.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": msg.content or "",
                            }
                        ],
                    }
                )
                continue

            if msg.role == Role.assistant:
                if msg.content:
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": msg.content,
                                }
                            ],
                            "status": "completed",
                        }
                    )

                for idx, tool_call in enumerate(msg.tool_calls or []):
                    call_id = tool_call.id or f"call_{idx}"
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": tool_call.function.name or "",
                            "arguments": tool_call.function.arguments or "{}",
                        }
                    )
                continue

            if msg.role == Role.tool:
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id or "",
                        "output": msg.content or "",
                    }
                )

        instructions = "\n\n".join(instruction_parts) if instruction_parts else None
        return instructions, input_items

    def _parse_non_stream_response(self, data: dict[str, Any]) -> LLMChunk:
        output_items = self._as_list(data.get("output"))
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        tool_index = 0
        for output in output_items:
            item = self._as_dict(output)
            item_type = self._as_str(item.get("type"))

            if item_type == "message":
                role = self._as_str(item.get("role"))
                if role and role != "assistant":
                    continue
                for content in self._as_list(item.get("content")):
                    content_item = self._as_dict(content)
                    content_type = self._as_str(content_item.get("type"))
                    if content_type in {"output_text", "text"}:
                        text = self._as_str(content_item.get("text"))
                        if text:
                            content_parts.append(text)
                    elif content_type == "refusal":
                        refusal = self._as_str(content_item.get("refusal"))
                        if refusal:
                            content_parts.append(refusal)
                continue

            if item_type == "reasoning":
                summaries = self._as_list(item.get("summary"))
                summary_texts = [
                    self._as_str(self._as_dict(summary).get("text"))
                    for summary in summaries
                ]
                joined_summary = "\n\n".join(text for text in summary_texts if text)
                if joined_summary:
                    reasoning_parts.append(joined_summary)
                continue

            if item_type == "function_call":
                call = self._tool_call_from_item(item, tool_index, include_arguments=True)
                if call:
                    tool_calls.append(call)
                    tool_index += 1

        usage = self._usage_from_usage_dict(self._as_dict(data.get("usage")))
        message = LLMMessage(
            role=Role.assistant,
            content="".join(content_parts) or None,
            reasoning_content="\n\n".join(reasoning_parts) or None,
            tool_calls=tool_calls or None,
        )
        return LLMChunk(message=message, usage=usage)

    def _parse_stream_event(self, data: dict[str, Any], event_type: str) -> LLMChunk:
        if event_type in {"response.output_text.delta", "response.refusal.delta"}:
            delta = self._as_str(data.get("delta"))
            if not delta:
                return self._empty_chunk()
            return LLMChunk(
                message=LLMMessage(role=Role.assistant, content=delta),
                usage=None,
            )

        if event_type == "response.reasoning_summary_text.delta":
            delta = self._as_str(data.get("delta"))
            if not delta:
                return self._empty_chunk()
            return LLMChunk(
                message=LLMMessage(role=Role.assistant, reasoning_content=delta),
                usage=None,
            )

        if event_type == "response.output_item.added":
            item = self._as_dict(data.get("item"))
            if self._as_str(item.get("type")) == "function_call":
                tool_call = self._tool_call_from_item(
                    item,
                    index=self._as_int(data.get("output_index"), 0),
                    include_arguments=False,
                )
                if tool_call:
                    return LLMChunk(
                        message=LLMMessage(role=Role.assistant, tool_calls=[tool_call]),
                        usage=None,
                    )
            return self._empty_chunk()

        if event_type == "response.output_item.done":
            item = self._as_dict(data.get("item"))
            if self._as_str(item.get("type")) == "function_call":
                tool_call = self._tool_call_from_item(
                    item,
                    index=self._as_int(data.get("output_index"), 0),
                    include_arguments=False,
                )
                if tool_call:
                    return LLMChunk(
                        message=LLMMessage(role=Role.assistant, tool_calls=[tool_call]),
                        usage=None,
                    )
            return self._empty_chunk()

        if event_type == "response.function_call_arguments.delta":
            call_id = self._as_str(data.get("call_id")) or self._as_str(
                data.get("item_id")
            )
            delta = self._as_str(data.get("delta"))
            if not call_id or not delta:
                return self._empty_chunk()

            tool_call = ToolCall(
                id=call_id,
                index=self._as_int(data.get("output_index"), 0),
                function=FunctionCall(
                    name=self._as_str(data.get("name")) or None,
                    arguments=delta,
                ),
            )
            return LLMChunk(
                message=LLMMessage(role=Role.assistant, tool_calls=[tool_call]),
                usage=None,
            )

        if event_type in {"response.completed", "response.done"}:
            usage = self._usage_from_usage_dict(
                self._as_dict(self._as_dict(data.get("response")).get("usage"))
            )
            return LLMChunk(
                message=LLMMessage(role=Role.assistant, content=""),
                usage=usage,
            )

        return self._empty_chunk()

    def _tool_call_from_item(
        self,
        item: dict[str, Any],
        index: int,
        *,
        include_arguments: bool,
    ) -> ToolCall | None:
        call_id = self._as_str(item.get("call_id")) or self._as_str(item.get("id"))
        if not call_id:
            return None

        name = self._as_str(item.get("name")) or None
        arguments = self._as_str(item.get("arguments")) if include_arguments else ""

        return ToolCall(
            id=call_id,
            index=index,
            function=FunctionCall(name=name, arguments=arguments or None),
        )

    def _usage_from_usage_dict(self, usage_data: dict[str, Any]) -> LLMUsage:
        prompt_tokens = self._as_int(
            usage_data.get("input_tokens", usage_data.get("prompt_tokens", 0)),
            0,
        )
        completion_tokens = self._as_int(
            usage_data.get("output_tokens", usage_data.get("completion_tokens", 0)),
            0,
        )
        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _empty_chunk(self) -> LLMChunk:
        return LLMChunk(message=LLMMessage(role=Role.assistant, content=""), usage=None)

    def _as_dict(self, value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def _as_list(self, value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    def _as_str(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        return str(value)

    def _as_int(self, value: Any, default: int) -> int:
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value))
            except ValueError:
                return default
        return default


ADAPTERS: dict[str, APIAdapter] = {
    "openai": OpenAIAdapter(),
    "openai-chatgpt-codex": OpenAIChatGPTCodexAdapter(),
    "anthropic": AnthropicAdapter(),
    "vertex-anthropic": VertexAnthropicAdapter(),
}


class GenericBackend:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        provider: ProviderConfig,
        timeout: float = 720.0,
        auth_resolver: SupportsProviderAuthResolver | None = None,
    ) -> None:
        """Initialize the backend.

        Args:
            client: Optional httpx client to use. If not provided, one will be created.
        """
        self._client = client
        self._owns_client = client is None
        self._provider = provider
        self._timeout = timeout
        self._auth_resolver = auth_resolver or ProviderAuthResolver()

    async def _build_prepared_request(  # noqa: PLR0913
        self,
        *,
        adapter: APIAdapter,
        model: ModelConfig,
        messages: list[LLMMessage],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        enable_streaming: bool,
        extra_headers: dict[str, str] | None,
        auth_context: ResolvedProviderAuth,
    ) -> tuple[PreparedRequest, dict[str, str], str]:
        req = adapter.prepare_request(
            model_name=model.name,
            messages=messages,
            temperature=temperature,
            tools=tools,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            enable_streaming=enable_streaming,
            provider=self._provider,
            api_key=auth_context.token,
            thinking=model.thinking,
        )

        headers = dict(req.headers)
        headers.update(auth_context.extra_headers)
        if extra_headers:
            headers.update(extra_headers)

        base = req.base_url or self._provider.api_base
        url = f"{base}{req.endpoint}"
        return req, headers, url

    async def _resolve_auth(self) -> ResolvedProviderAuth:
        try:
            return await self._auth_resolver.resolve(self._provider)
        except AuthRequiredError as exc:
            raise RuntimeError(str(exc)) from exc

    async def __aenter__(self) -> GenericBackend:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
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
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
            self._owns_client = True
        return self._client

    async def complete(
        self,
        *,
        model: ModelConfig,
        messages: list[LLMMessage],
        temperature: float = 0.2,
        tools: list[AvailableTool] | None = None,
        max_tokens: int | None = None,
        tool_choice: StrToolChoice | AvailableTool | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMChunk:
        api_style = getattr(self._provider, "api_style", "openai")
        adapter = ADAPTERS[api_style]
        auth_context = await self._resolve_auth()
        req, headers, url = await self._build_prepared_request(
            adapter=adapter,
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            enable_streaming=False,
            extra_headers=extra_headers,
            auth_context=auth_context,
        )

        try:
            res_data, _ = await self._make_request(url, req.body, headers)
            return adapter.parse_response(res_data, self._provider)

        except httpx.HTTPStatusError as e:
            if (
                e.response.status_code == 401
                and auth_context.auth_type == ProviderAuthType.OAUTH
            ):
                refreshed_auth_context = await self._auth_resolver.force_refresh(
                    self._provider
                )
                req, headers, url = await self._build_prepared_request(
                    adapter=adapter,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    tools=tools,
                    max_tokens=max_tokens,
                    tool_choice=tool_choice,
                    enable_streaming=False,
                    extra_headers=extra_headers,
                    auth_context=refreshed_auth_context,
                )
                try:
                    res_data, _ = await self._make_request(url, req.body, headers)
                    return adapter.parse_response(res_data, self._provider)
                except httpx.HTTPStatusError as final_err:
                    raise BackendErrorBuilder.build_http_error(
                        provider=self._provider.name,
                        endpoint=url,
                        response=final_err.response,
                        headers=final_err.response.headers,
                        model=model.name,
                        messages=messages,
                        temperature=temperature,
                        has_tools=bool(tools),
                        tool_choice=tool_choice,
                    ) from final_err

            raise BackendErrorBuilder.build_http_error(
                provider=self._provider.name,
                endpoint=url,
                response=e.response,
                headers=e.response.headers,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e
        except httpx.RequestError as e:
            raise BackendErrorBuilder.build_request_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e

    async def complete_streaming(
        self,
        *,
        model: ModelConfig,
        messages: list[LLMMessage],
        temperature: float = 0.2,
        tools: list[AvailableTool] | None = None,
        max_tokens: int | None = None,
        tool_choice: StrToolChoice | AvailableTool | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        api_style = getattr(self._provider, "api_style", "openai")
        adapter = ADAPTERS[api_style]
        auth_context = await self._resolve_auth()
        req, headers, url = await self._build_prepared_request(
            adapter=adapter,
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            enable_streaming=True,
            extra_headers=extra_headers,
            auth_context=auth_context,
        )

        try:
            async for res_data in self._make_streaming_request(url, req.body, headers):
                yield adapter.parse_response(res_data, self._provider)

        except httpx.HTTPStatusError as e:
            if (
                e.response.status_code == 401
                and auth_context.auth_type == ProviderAuthType.OAUTH
            ):
                refreshed_auth_context = await self._auth_resolver.force_refresh(
                    self._provider
                )
                req, headers, url = await self._build_prepared_request(
                    adapter=adapter,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    tools=tools,
                    max_tokens=max_tokens,
                    tool_choice=tool_choice,
                    enable_streaming=True,
                    extra_headers=extra_headers,
                    auth_context=refreshed_auth_context,
                )
                try:
                    async for res_data in self._make_streaming_request(
                        url, req.body, headers
                    ):
                        yield adapter.parse_response(res_data, self._provider)
                    return
                except httpx.HTTPStatusError as final_err:
                    raise BackendErrorBuilder.build_http_error(
                        provider=self._provider.name,
                        endpoint=url,
                        response=final_err.response,
                        headers=final_err.response.headers,
                        model=model.name,
                        messages=messages,
                        temperature=temperature,
                        has_tools=bool(tools),
                        tool_choice=tool_choice,
                    ) from final_err

            raise BackendErrorBuilder.build_http_error(
                provider=self._provider.name,
                endpoint=url,
                response=e.response,
                headers=e.response.headers,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e
        except httpx.RequestError as e:
            raise BackendErrorBuilder.build_request_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e

    class HTTPResponse(NamedTuple):
        data: dict[str, Any]
        headers: dict[str, str]

    @async_retry(tries=3)
    async def _make_request(
        self, url: str, data: bytes, headers: dict[str, str]
    ) -> HTTPResponse:
        client = self._get_client()
        response = await client.post(url, content=data, headers=headers)
        response.raise_for_status()

        response_headers = dict(response.headers.items())
        response_body = response.json()
        return self.HTTPResponse(response_body, response_headers)

    @async_generator_retry(tries=3)
    async def _make_streaming_request(
        self, url: str, data: bytes, headers: dict[str, str]
    ) -> AsyncGenerator[dict[str, Any]]:
        def decode_data_event(data_lines: list[str]) -> dict[str, Any] | None:
            if not data_lines:
                return None

            payload = "\n".join(data_lines).strip()
            if payload == "[DONE]":
                return None
            return json.loads(payload)

        client = self._get_client()
        async with client.stream(
            method="POST", url=url, content=data, headers=headers
        ) as response:
            if not response.is_success:
                await response.aread()
            response.raise_for_status()
            data_lines: list[str] = []
            async for line in response.aiter_lines():
                if line.strip() == "":
                    parsed_data = decode_data_event(data_lines)
                    if parsed_data is None:
                        if data_lines and "\n".join(data_lines).strip() == "[DONE]":
                            return
                        data_lines.clear()
                        continue

                    data_lines.clear()
                    yield parsed_data
                    continue

                if line.startswith(":"):
                    continue

                DELIM_CHAR = ":"
                if DELIM_CHAR not in line:
                    raise ValueError(
                        f"Stream chunk improperly formatted. "
                        f"Expected `key{DELIM_CHAR}value`, received `{line}`"
                    )
                key, value = line.split(DELIM_CHAR, 1)
                if value.startswith(" "):
                    value = value[1:]

                if key != "data":
                    # This might be the case with openrouter, so we just ignore it
                    continue

                data_lines.append(value)

            parsed_data = decode_data_event(data_lines)
            if parsed_data is not None:
                yield parsed_data

    async def count_tokens(
        self,
        *,
        model: ModelConfig,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        tools: list[AvailableTool] | None = None,
        tool_choice: StrToolChoice | AvailableTool | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> int:
        probe_messages = list(messages)
        if not probe_messages or probe_messages[-1].role != Role.user:
            probe_messages.append(LLMMessage(role=Role.user, content=""))

        result = await self.complete(
            model=model,
            messages=probe_messages,
            temperature=temperature,
            tools=tools,
            max_tokens=16,  # Minimal amount for openrouter with openai models
            tool_choice=tool_choice,
            extra_headers=extra_headers,
        )
        if result.usage is None:
            raise ValueError("Missing usage in non streaming completion")

        return result.usage.prompt_tokens

    async def close(self) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()
            self._client = None
