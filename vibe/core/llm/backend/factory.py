from __future__ import annotations

from vibe.core.config import Backend
from vibe.core.llm.backend.generic import GenericBackend
from vibe.core.llm.backend.mistral import MistralBackend
from vibe.core.llm.backend.openai_chatgpt import OpenAIChatGPTBackend

BACKEND_FACTORY = {
    Backend.MISTRAL: MistralBackend,
    Backend.GENERIC: GenericBackend,
    Backend.OPENAI_CHATGPT: OpenAIChatGPTBackend,
}
