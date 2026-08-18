# PHASE 5 — LLM Providers

> **Build guide for the from-scratch reconstruction of the arXiv Paper Curator (Agentic RAG).**
>
> This phase implements the LLM provider abstraction layer. It defines a shared `LLMClientProtocol` interface and three concrete implementations — **OpenAI**, **AWS Bedrock**, and **Ollama** — each exposing the same RAG answer generation, streaming, and health-check capabilities. The phase also wires the provider-selection mechanism (via the `PROVIDER` environment variable) into the application startup in [`src/main.py`](../src/main.py).

---

## 1. Phase Objective

By the end of this phase you will have:

- A runtime-checkable [`LLMClientProtocol`](../src/services/llm_client_protocol.py) that all LLM clients implement.
- An [`OpenAILLMClient`](../src/services/openai_llm/client.py) backed by the OpenAI Chat Completions API (default model `gpt-4o-mini`).
- A [`BedrockLLMClient`](../src/services/bedrock_llm/client.py) backed by the AWS Bedrock `converse` API via `boto3`.
- An [`OllamaClient`](../src/services/ollama/client.py) for local models, plus a reusable [`RAGPromptBuilder`](../src/services/ollama/prompts.py) and [`ResponseParser`](../src/services/ollama/prompts.py).
- A provider-selection mechanism in [`src/main.py`](../src/main.py) that chooses Bedrock or OpenAI based on the `PROVIDER` env var.

> **Note on scope:** The Ollama client is implemented and fully functional, but it is **not** wired into the provider-selection switch in `src/main.py` (only `bedrock` and the default `openai` are selectable). This is intentional and documented as a pitfall in Section 8.

---

## 2. Prerequisites

- Completion of **PHASE 1 (Project Setup)** and **PHASE 2 (Configuration)** so that [`src/config.py`](../src/config.py) and the `Settings`/`get_settings()` helpers exist.
- Completion of **PHASE 4 (Schemas)** so that the `RAGResponse` schema (used by the structured prompt builder) exists in [`src/schemas/ollama.py`](../src/schemas/ollama.py).
- A valid API key for at least one provider:
  - **OpenAI:** `OPENAI_API_KEY`
  - **AWS Bedrock:** AWS credentials with Bedrock access (region + optional access key/secret)
  - **Ollama:** a running local Ollama server (default `http://localhost:11434`)
- Python 3.11+ and the project's virtual environment active.

---

## 3. Dependencies to Install

Add the following to `pyproject.toml` (or install via `uv add` / `pip install`):

```bash
# Core LLM provider SDKs
openai
boto3
langchain-openai
langchain-aws
langchain-ollama
langchain-core
httpx
```

> These are in addition to the base dependencies installed in PHASE 1 (e.g. `pydantic`, `pydantic-settings`, `fastapi`).

---

## 4. Directory Structure to Create

```
src/services/
├── llm_client_protocol.py          # Shared Protocol interface
├── openai_llm/
│   ├── __init__.py                 # Re-exports OpenAILLMClient
│   ├── client.py                   # OpenAILLMClient
│   └── factory.py                  # make_openai_llm_client
├── bedrock_llm/
│   ├── __init__.py                 # (empty)
│   ├── client.py                   # BedrockLLMClient
│   └── factory.py                  # make_bedrock_llm_client
└── ollama/
    ├── __init__.py                 # Re-exports OllamaClient
    ├── client.py                   # OllamaClient
    ├── factory.py                  # make_ollama_client
    ├── prompts.py                  # RAGPromptBuilder, ResponseParser
    └── prompts/
        └── rag_system.txt          # System prompt text
```

---

## 5. Step-by-Step Implementation

### Step 1 — Define the shared `LLMClientProtocol`

**Full file path:** `src/services/llm_client_protocol.py`

```python
"""Shared protocol interface for all LLM client implementations."""

from typing import Any, AsyncIterator, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Shared interface for all LLM client implementations.

    All LLM providers (OpenAI, Bedrock, Ollama) implement this protocol so
    they can be used interchangeably throughout the application.
    """

    def get_langchain_model(self, model: str, temperature: float = 0.0) -> Any:
        """Return a LangChain chat model instance for use in agent nodes."""
        ...

    async def generate_rag_answer(
        self, query: str, chunks: List[Dict[str, Any]], model: str = "", **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate a RAG answer from retrieved chunks."""
        ...

    async def generate_rag_answer_stream(
        self, query: str, chunks: List[Dict[str, Any]], model: str = ""
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream a RAG answer from retrieved chunks."""
        ...

    async def health_check(self) -> Dict[str, Any]:
        """Check provider API connectivity."""
        ...
```

**Explanation:** This is a `typing.Protocol` marked `@runtime_checkable`, so `isinstance(client, LLMClientProtocol)` works at runtime. It defines the four methods every provider must implement: returning a LangChain model, generating a RAG answer, streaming a RAG answer, and checking health. The `...` (ellipsis) bodies are placeholders — protocols only declare the interface.

---

### Step 2 — Implement the OpenAI client

**Full file path:** `src/services/openai_llm/client.py`

```python
"""Client for OpenAI API — drop-in replacement for OllamaClient."""

import logging
from typing import Any, AsyncIterator, Dict, List

from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI

from src.config import Settings
from src.services.ollama.prompts import RAGPromptBuilder, ResponseParser

logger = logging.getLogger(__name__)


class OpenAILLMClient:
    """Client for OpenAI API — drop-in replacement for OllamaClient."""

    def __init__(self, settings: Settings):
        self.api_key = settings.openai_api_key
        self.timeout = settings.openai_timeout
        self.prompt_builder = RAGPromptBuilder()
        self.response_parser = ResponseParser()

    def _get_async_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=self.api_key, timeout=self.timeout)

    def get_langchain_model(self, model: str, temperature: float = 0.0):
        return ChatOpenAI(model=model, temperature=temperature, api_key=self.api_key)

    async def health_check(self) -> Dict[str, Any]:
        """Check OpenAI API connectivity."""
        try:
            client = self._get_async_client()
            await client.models.list()
            return {"status": "ok", "provider": "openai"}
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return {"status": "error", "provider": "openai", "error": str(e)}

    async def generate_rag_answer(
        self, query: str, chunks: List[Dict[str, Any]], model: str = "", **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate a RAG answer using retrieved chunks via OpenAI chat completions."""
        try:
            model = model or "gpt-4o-mini"
            prompt = self.prompt_builder.create_rag_prompt(query, chunks)

            client = self._get_async_client()
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful research assistant. Answer questions based only on the provided context from academic papers. Be concise and accurate.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )

            answer = response.choices[0].message.content

            # Build sources list from chunks
            sources = []
            for chunk in chunks:
                arxiv_id = chunk.get("arxiv_id", "")
                arxiv_id_clean = arxiv_id.replace("arXiv:", "").strip()
                sources.append(
                    {
                        "arxiv_id": arxiv_id,
                        "url": f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf",
                        "title": chunk.get("title", ""),
                        "authors": chunk.get("authors", ""),
                        "abstract": chunk.get("abstract", ""),
                    }
                )

            return {
                "answer": answer,
                "sources": sources,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }

        except Exception as e:
            logger.error(f"OpenAI RAG answer generation failed: {e}")
            raise

    async def generate_rag_answer_stream(
        self, query: str, chunks: List[Dict[str, Any]], model: str = ""
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream a RAG answer using OpenAI streaming chat completions."""
        try:
            model = model or "gpt-4o-mini"
            prompt = self.prompt_builder.create_rag_prompt(query, chunks)

            client = self._get_async_client()
            stream = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful research assistant. Answer questions based only on the provided context from academic papers. Be concise and accurate.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield {"content": chunk.choices[0].delta.content}

        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise
```

**Explanation:** The OpenAI client reuses the `RAGPromptBuilder` and `ResponseParser` from the Ollama package (they are provider-agnostic). It uses `AsyncOpenAI` for async calls, defaults to `gpt-4o-mini`, uses `temperature=0.0` for deterministic answers, and builds a `sources` list with arXiv PDF URLs for citation. The streaming variant yields `{"content": ...}` dicts.

---

### Step 3 — Add the OpenAI factory

**Full file path:** `src/services/openai_llm/factory.py`

```python
from functools import lru_cache

from src.config import get_settings

from .client import OpenAILLMClient


@lru_cache(maxsize=1)
def make_openai_llm_client() -> OpenAILLMClient:
    """Create a cached OpenAI LLM client singleton."""
    settings = get_settings()
    return OpenAILLMClient(settings)
```

**Explanation:** The factory is decorated with `@lru_cache(maxsize=1)` so the client is created once and reused (a singleton). It pulls settings via `get_settings()` and constructs the client.

---

### Step 4 — Add the OpenAI package `__init__.py`

**Full file path:** `src/services/openai_llm/__init__.py`

```python
from .client import OpenAILLMClient

__all__ = ["OpenAILLMClient"]
```

**Explanation:** Re-exports the client so callers can do `from src.services.openai_llm import OpenAILLMClient`.

---

### Step 5 — Implement the Bedrock client

**Full file path:** `src/services/bedrock_llm/client.py`

```python
"""LLM client backed by AWS Bedrock — drop-in replacement for OpenAILLMClient."""

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import boto3
from langchain_aws import ChatBedrock

from src.config import Settings

logger = logging.getLogger(__name__)


class BedrockLLMClient:
    """LLM client backed by AWS Bedrock — drop-in replacement for OpenAILLMClient."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.region = settings.bedrock.region
        self.model_id = settings.bedrock.model_id

    def _get_client(self) -> Any:
        """Create a boto3 bedrock-runtime client."""
        kwargs: Dict[str, Any] = {"region_name": self.region}

        if self.settings.bedrock.aws_access_key_id:
            kwargs["aws_access_key_id"] = self.settings.bedrock.aws_access_key_id
        if self.settings.bedrock.aws_secret_access_key:
            kwargs["aws_secret_access_key"] = self.settings.bedrock.aws_secret_access_key

        return boto3.client("bedrock-runtime", **kwargs)

    @staticmethod
    def _infer_provider(model_id: str) -> Optional[str]:
        """Infer the model provider from the model ID prefix."""
        model_id_lower = model_id.lower()
        if model_id_lower.startswith("meta") or "llama" in model_id_lower:
            return "meta"
        if model_id_lower.startswith("anthropic") or "claude" in model_id_lower:
            return "anthropic"
        if model_id_lower.startswith("amazon") or "titan" in model_id_lower or "nova" in model_id_lower:
            return "amazon"
        if "mistral" in model_id_lower:
            return "mistral"
        if "cohere" in model_id_lower:
            return "cohere"
        return None

    def get_langchain_model(self, model: str = "", temperature: float = 0.0) -> Any:
        """Return a LangChain ChatBedrock instance for use in agent nodes."""
        kwargs: Dict[str, Any] = {
            "model_id": model or self.model_id,
            "region_name": self.region,
            "temperature": temperature,
        }
        if self.settings.bedrock.aws_access_key_id:
            kwargs["aws_access_key_id"] = self.settings.bedrock.aws_access_key_id
        if self.settings.bedrock.aws_secret_access_key:
            kwargs["aws_secret_access_key"] = self.settings.bedrock.aws_secret_access_key
        return ChatBedrock(**kwargs)

    async def health_check(self) -> Dict[str, Any]:
        """Check Bedrock API connectivity by listing foundation models."""
        try:
            client = self._get_client()
            await asyncio.to_thread(client.list_foundation_models)
            return {"status": "ok", "provider": "bedrock", "model_id": self.model_id}
        except Exception as e:
            logger.error(f"Bedrock health check failed: {e}")
            return {"status": "error", "provider": "bedrock", "error": str(e)}

    async def generate_rag_answer(
        self, query: str, chunks: List[Dict[str, Any]], model: str = "", **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate a RAG answer using Bedrock converse API."""
        try:
            model = model or self.model_id
            prompt = self._build_prompt(query, chunks)

            client = self._get_client()

            def _call_converse() -> Dict[str, Any]:
                return client.converse(
                    modelId=model,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    system=[{"text": "You are a helpful research assistant. Answer questions based only on the provided context from academic papers. Be concise and accurate."}],
                    inferenceConfig={"temperature": 0.0},
                )

            response = await asyncio.to_thread(_call_converse)

            answer = response["output"]["message"]["content"][0]["text"]

            sources = []
            for chunk in chunks:
                arxiv_id = chunk.get("arxiv_id", "")
                arxiv_id_clean = arxiv_id.replace("arXiv:", "").strip()
                sources.append(
                    {
                        "arxiv_id": arxiv_id,
                        "url": f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf",
                        "title": chunk.get("title", ""),
                        "authors": chunk.get("authors", ""),
                        "abstract": chunk.get("abstract", ""),
                    }
                )

            usage = response.get("usage", {})
            return {
                "answer": answer,
                "sources": sources,
                "usage": {
                    "prompt_tokens": usage.get("inputTokens", 0),
                    "completion_tokens": usage.get("outputTokens", 0),
                    "total_tokens": usage.get("totalTokens", 0),
                },
            }

        except Exception as e:
            logger.error(f"Bedrock RAG answer generation failed: {e}")
            raise

    async def generate_rag_answer_stream(
        self, query: str, chunks: List[Dict[str, Any]], model: str = ""
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream a RAG answer using Bedrock converse_stream API."""
        try:
            model = model or self.model_id
            prompt = self._build_prompt(query, chunks)

            client = self._get_client()

            def _collect_stream() -> List[str]:
                response = client.converse_stream(
                    modelId=model,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    system=[{"text": "You are a helpful research assistant. Answer questions based only on the provided context from academic papers. Be concise and accurate."}],
                    inferenceConfig={"temperature": 0.0},
                )
                stream = response.get("stream", [])
                parts = []
                for event in stream:
                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"].get("delta", {})
                        if "text" in delta:
                            parts.append(delta["text"])
                return parts

            parts = await asyncio.to_thread(_collect_stream)
            for part in parts:
                yield {"content": part}

        except Exception as e:
            logger.error(f"Bedrock streaming failed: {e}")
            raise

    def _build_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Build a RAG prompt from query and chunks."""
        context = "\n\n".join(
            f"[{i}. arXiv:{chunk.get('arxiv_id', '')}] {chunk.get('chunk_text', '')}"
            for i, chunk in enumerate(chunks)
        )
        return f"### Context from Papers:\n{context}\n\n### Question:\n{query}\n\n### Answer:\n"
```

**Explanation:** The Bedrock client uses `boto3`'s `bedrock-runtime` client and the `converse` / `converse_stream` APIs. Because `boto3` is synchronous, blocking calls are wrapped in `asyncio.to_thread`. The `_infer_provider` static method maps model ID prefixes to providers (used for guardrail/formatting decisions). It builds its own prompt (rather than reusing `RAGPromptBuilder`) and returns the same `answer`/`sources`/`usage` shape as the OpenAI client, making it a true drop-in replacement.

---

### Step 6 — Add the Bedrock factory

**Full file path:** `src/services/bedrock_llm/factory.py`

```python
from src.config import Settings, get_settings

from .client import BedrockLLMClient


def make_bedrock_llm_client(settings: Settings | None = None) -> BedrockLLMClient:
    """Create a Bedrock LLM client.

    :param settings: Optional settings instance. If None, loads from config.
    """
    if settings is None:
        settings = get_settings()
    return BedrockLLMClient(settings)
```

**Explanation:** Unlike the OpenAI factory, this one is **not** cached with `lru_cache` — it accepts an optional `settings` parameter (so the caller can pass the already-loaded settings from `src/main.py`). It returns a fresh client each call.

---

### Step 7 — Add the Bedrock package `__init__.py`

**Full file path:** `src/services/bedrock_llm/__init__.py`

```python
```

**Explanation:** This file is intentionally empty in the reference project. It exists to mark the directory as a Python package. (You may optionally re-export `BedrockLLMClient` here for convenience.)

---

### Step 8 — Create the Ollama system prompt file

**Full file path:** `src/services/ollama/prompts/rag_system.txt`

```text
You are an AI assistant specialized in answering questions about academic papers from arXiv. Your task is to provide accurate, well-reasoned answers based strictly on the provided excerpts from research papers.

Guidelines:
1. Base your answer STRICTLY on the provided context excerpts. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say so clearly.
3. Cite the specific papers you use by their arXiv ID, e.g. [1], [2].
4. Be concise and to the point. Aim for answers under 300 words.
5. Use markdown formatting for readability where appropriate.
6. If multiple papers provide conflicting information, note the disagreement.
7. Do not mention that you are an AI or that you have limitations.
```

**Explanation:** This is the system prompt loaded by `RAGPromptBuilder`. It instructs the model to answer strictly from the provided excerpts, cite papers by arXiv ID, and stay under 300 words.

---

### Step 9 — Implement the prompt builder and response parser

**Full file path:** `src/services/ollama/prompts.py`

```python
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.schemas.ollama import RAGResponse

logger = logging.getLogger(__name__)


class RAGPromptBuilder:
    """Builder class for creating RAG prompts."""

    def __init__(self):
        self.prompts_dir = Path(__file__).parent / "prompts"
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load the system prompt from the prompts directory."""
        try:
            prompt_file = self.prompts_dir / "rag_system.txt"
            if prompt_file.exists():
                return prompt_file.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning(f"Failed to load system prompt: {e}")

        # Fallback default
        return (
            "You are an AI assistant specialized in answering questions about academic papers from arXiv. "
            "Base your answer strictly on the provided context excerpts."
        )

    def create_rag_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Create a RAG prompt from a query and retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks):
            arxiv_id = chunk.get("arxiv_id", "")
            text = chunk.get("chunk_text", chunk.get("text", ""))
            context_parts.append(f"[{i}. arXiv:{arxiv_id}] {text}")

        context = "\n\n".join(context_parts)

        prompt = (
            f"### Context from Papers:\n{context}\n\n"
            f"### Question:\n{query}\n\n"
            f"### Answer:\n"
        )
        return prompt

    def create_structured_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a structured prompt that requests JSON output."""
        prompt = self.create_rag_prompt(query, chunks)
        return {
            "prompt": prompt,
            "format": RAGResponse.model_json_schema(),
        }


class ResponseParser:
    """Parser for LLM responses."""

    @staticmethod
    def parse_structured_response(response: str) -> Dict[str, Any]:
        """Parse a structured response from Ollama.

        :param response: Raw response string
        :returns: Parsed dictionary
        """
        try:
            # Try direct JSON parse first
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback to extracting JSON from the response
            return ResponseParser._extract_json_fallback(response)

    @staticmethod
    def _extract_json_fallback(response: str) -> Dict[str, Any]:
        """Extract JSON from a response that may contain extra text."""
        # Try to find JSON object in the response
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Return empty dict if parsing fails
        return {}
```

**Explanation:** `RAGPromptBuilder` loads the system prompt from `rag_system.txt` (with a fallback), builds a text prompt with a `### Context from Papers:` / `### Question:` / `### Answer:` structure, and can produce a structured prompt with the `RAGResponse` JSON schema for JSON-mode generation. `ResponseParser` parses structured responses, with a fallback that extracts the JSON object from a response that may contain surrounding text.

---

### Step 10 — Implement the Ollama client

**Full file path:** `src/services/ollama/client.py`

```python
"""Client for interacting with Ollama local LLM service."""

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from langchain_ollama import ChatOllama

from src.config import Settings
from src.services.ollama.prompts import RAGPromptBuilder, ResponseParser

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama local LLM service."""

    def __init__(self, settings: Settings):
        self.base_url = settings.ollama_host
        self.timeout = httpx.Timeout(float(settings.ollama_timeout))
        self.prompt_builder = RAGPromptBuilder()
        self.response_parser = ResponseParser()

    def get_langchain_model(self, model: str, temperature: float = 0.0):
        """Return a LangChain ChatOllama instance."""
        kwargs: Dict[str, Any] = {
            "model": model,
            "base_url": self.base_url,
            "temperature": temperature,
        }
        # qwen3 models require reasoning=False to avoid thinking tokens
        if "qwen3" in model.lower():
            kwargs["reasoning"] = False
        return ChatOllama(**kwargs)

    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama server connectivity."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/version")
                response.raise_for_status()
                version = response.json().get("version", "unknown")
                return {"status": "ok", "provider": "ollama", "version": version}
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return {"status": "error", "provider": "ollama", "error": str(e)}

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models on the Ollama server."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                return response.json().get("models", [])
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    async def generate(self, model: str, prompt: str, stream: bool = False, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate a completion using the Ollama generate API."""
        try:
            data = {"model": model, "prompt": prompt, "stream": stream, **kwargs}

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/generate", json=data)
                response.raise_for_status()

                if stream:
                    # For streaming, return the raw response for the caller to iterate
                    return response

                result = response.json()
                usage_metadata = result.get("usage_metadata", {})
                return {
                    "response": result.get("response", ""),
                    "usage": {
                        "prompt_tokens": usage_metadata.get("prompt_eval_count", 0),
                        "completion_tokens": usage_metadata.get("eval_count", 0),
                        "total_tokens": usage_metadata.get("prompt_eval_count", 0) + usage_metadata.get("eval_count", 0),
                    },
                }
        except Exception as e:
            logger.error(f"Ollama generate failed: {e}")
            return None

    async def generate_stream(self, model: str, prompt: str, **kwargs):
        """Stream a completion from Ollama."""
        try:
            data = {"model": model, "prompt": prompt, "stream": True, **kwargs}

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", f"{self.base_url}/api/generate", json=data) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                chunk = json.loads(line)
                                if chunk.get("response"):
                                    yield {"content": chunk["response"]}
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            logger.error(f"Ollama stream failed: {e}")
            raise

    async def generate_rag_answer(
        self, query: str, chunks: List[Dict[str, Any]], model: str = "", **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate a RAG answer using Ollama."""
        try:
            model = model or "llama3.2"
            use_structured_output = kwargs.pop("use_structured_output", False)

            if use_structured_output:
                prompt_data = self.prompt_builder.create_structured_prompt(query, chunks)
                result = await self.generate(model, prompt_data["prompt"], format=prompt_data["format"])
            else:
                prompt = self.prompt_builder.create_rag_prompt(query, chunks)
                result = await self.generate(model, prompt, temperature=0.7, top_p=0.9)

            if not result:
                return {"answer": "", "sources": [], "usage": {}}

            answer = result.get("response", "")

            if use_structured_output:
                parsed = self.response_parser.parse_structured_response(answer)
                answer = parsed.get("answer", answer)

            sources = []
            for chunk in chunks:
                arxiv_id = chunk.get("arxiv_id", "")
                arxiv_id_clean = arxiv_id.replace("arXiv:", "").strip()
                sources.append(
                    {
                        "arxiv_id": arxiv_id,
                        "url": f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf",
                        "title": chunk.get("title", ""),
                        "authors": chunk.get("authors", ""),
                        "abstract": chunk.get("abstract", ""),
                    }
                )

            return {"answer": answer, "sources": sources, "usage": result.get("usage", {})}

        except Exception as e:
            logger.error(f"Ollama RAG answer generation failed: {e}")
            raise

    async def generate_rag_answer_stream(
        self, query: str, chunks: List[Dict[str, Any]], model: str = ""
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream a RAG answer using Ollama."""
        try:
            model = model or "llama3.2"
            prompt = self.prompt_builder.create_rag_prompt(query, chunks)
            async for chunk in self.generate_stream(model, prompt, temperature=0.7, top_p=0.9):
                yield chunk
        except Exception as e:
            logger.error(f"Ollama RAG streaming failed: {e}")
            raise
```

**Explanation:** The Ollama client talks to a local Ollama server over HTTP (`httpx`). It uses `settings.ollama_host` and `settings.ollama_timeout` for the base URL and timeout. It supports health checks (`/api/version`), model listing (`/api/tags`), raw generation (`/api/generate`), streaming, and the full RAG answer interface. For `qwen3` models it disables reasoning tokens. The default RAG model is `llama3.2` with `temperature=0.7` and `top_p=0.9`.

> **⚠️ Important:** `OllamaClient.__init__` references `settings.ollama_host` and `settings.ollama_timeout`. In the current reference [`src/config.py`](../src/config.py), these fields do **not** exist (see Section 8). If you instantiate `OllamaClient` with the current `Settings`, it will raise `AttributeError`. Add these fields to `Settings` if you intend to use Ollama.

---

### Step 11 — Add the Ollama factory

**Full file path:** `src/services/ollama/factory.py`

```python
from functools import lru_cache

from src.config import get_settings

from .client import OllamaClient


@lru_cache(maxsize=1)
def make_ollama_client() -> OllamaClient:
    """Create a cached Ollama client singleton."""
    settings = get_settings()
    return OllamaClient(settings)
```

**Explanation:** Like the OpenAI factory, this is cached with `@lru_cache(maxsize=1)` to create a singleton.

---

### Step 12 — Add the Ollama package `__init__.py`

**Full file path:** `src/services/ollama/__init__.py`

```python
from .client import OllamaClient

__all__ = ["OllamaClient"]
```

**Explanation:** Re-exports the client so callers can do `from src.services.ollama import OllamaClient`.

---

### Step 13 — Wire the provider selection into `src/main.py`

**Full file path:** `src/main.py` (lifespan section)

Add the following inside the `lifespan` async context manager, after the embeddings service is created:

```python
# Select LLM provider based on configuration
if settings.provider == "bedrock":
    app.state.llm_client = make_bedrock_llm_client(settings)
    logger.info(f"LLM provider: AWS Bedrock (model={settings.bedrock.model_id})")
else:
    app.state.llm_client = make_openai_llm_client()
    logger.info(f"LLM provider: OpenAI (model={settings.openai_model})")
```

Add the corresponding imports at the top of `src/main.py`:

```python
from src.services.bedrock_llm.factory import make_bedrock_llm_client
from src.services.openai_llm.factory import make_openai_llm_client
```

**Explanation:** This is the provider-selection mechanism. When `PROVIDER=bedrock`, the app builds a `BedrockLLMClient` (passing the already-loaded settings). Otherwise it defaults to the cached `OpenAILLMClient`. The chosen client is stored on `app.state.llm_client` for use by API routes and agent nodes. **Note:** Ollama is intentionally not part of this switch — selecting it requires extending this `if/else` (see Section 8).

---

## 6. Configuration

Add the following environment variables to your `.env` file (see PHASE 2 for the full config setup):

```bash
# Provider selection: "openai" (default) or "bedrock"
PROVIDER=openai

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT=300

# AWS Bedrock (only needed if PROVIDER=bedrock)
BEDROCK__REGION=us-east-1
BEDROCK__MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
# Optional explicit AWS credentials (otherwise use default AWS credential chain)
BEDROCK__AWS_ACCESS_KEY_ID=
BEDROCK__AWS_SECRET_ACCESS_KEY=

# Ollama (optional — client implemented but NOT wired into provider selection)
# NOTE: settings.ollama_host / settings.ollama_timeout are NOT defined in src/config.py
# Add them to Settings if you intend to use Ollama.
OLLAMA_HOST=http://localhost:11434
OLLAMA_TIMEOUT=300
```

**Expected results:** The `Settings` object loads these values. With `PROVIDER=openai`, the app builds an `OpenAILLMClient`; with `PROVIDER=bedrock`, it builds a `BedrockLLMClient`.

---

## 7. Verification

Run the following checks to confirm the phase is complete.

### 7.1 Import and protocol check

```bash
python -c "
from src.services.llm_client_protocol import LLMClientProtocol
from src.services.openai_llm.client import OpenAILLMClient
from src.services.bedrock_llm.client import BedrockLLMClient
from src.services.ollama.client import OllamaClient
from src.config import get_settings

settings = get_settings()
print('OpenAI is protocol:', isinstance(OpenAILLMClient(settings), LLMClientProtocol))
print('Bedrock is protocol:', isinstance(BedrockLLMClient(settings), LLMClientProtocol))
"
```

**Expected results:** Both `OpenAILLMClient` and `BedrockLLMClient` report `True` for `isinstance(..., LLMClientProtocol)`. (Ollama will fail to instantiate until `ollama_host`/`ollama_timeout` are added to `Settings`.)

### 7.2 Factory smoke test

```bash
python -c "
from src.services.openai_llm.factory import make_openai_llm_client
from src.services.bedrock_llm.factory import make_bedrock_llm_client
from src.config import get_settings

a = make_openai_llm_client()
b = make_openai_llm_client()
print('OpenAI singleton:', a is b)  # True due to lru_cache

settings = get_settings()
c = make_bedrock_llm_client(settings)
print('Bedrock client created:', type(c).__name__)
"
```

**Expected results:** The two OpenAI clients are the same object (`True`), confirming the `lru_cache` singleton. The Bedrock client is created successfully.

### 7.3 Health check (requires live provider credentials)

```bash
python -c "
import asyncio
from src.config import get_settings
from src.services.openai_llm.factory import make_openai_llm_client

async def main():
    client = make_openai_llm_client()
    print(await client.health_check())

asyncio.run(main())
"
```

**Expected results:** With a valid `OPENAI_API_KEY`, this prints `{'status': 'ok', 'provider': 'openai'}`. Without credentials it prints a `status: error` dict — which is also correct behavior (graceful failure).

### 7.4 Provider selection in `src/main.py`

Start the API and confirm the startup log line:

```bash
uvicorn src.main:app --reload
```

**Expected results:** The logs show either `LLM provider: OpenAI (model=gpt-4o-mini)` (default) or `LLM provider: AWS Bedrock (model=...)` when `PROVIDER=bedrock`.

---

## 8. Common Pitfalls

- **Ollama is not wired into provider selection.** The `if/else` in [`src/main.py`](../src/main.py:88) only handles `bedrock` and the default `openai`. To use Ollama, extend the switch, e.g. `elif settings.provider == "ollama": app.state.llm_client = make_ollama_client()`.
- **`ollama_host` / `ollama_timeout` are missing from `Settings`.** [`OllamaClient.__init__`](../src/services/ollama/client.py:17) reads `settings.ollama_host` and `settings.ollama_timeout`, but [`src/config.py`](../src/config.py) does not define them. Instantiating `OllamaClient` with the current `Settings` raises `AttributeError`. Add these fields to `Settings` (e.g. `ollama_host: str = "http://localhost:11434"` and `ollama_timeout: int = 300`) before using Ollama.
- **Bedrock requires `asyncio.to_thread`.** `boto3` is synchronous. All blocking calls (`converse`, `converse_stream`, `list_foundation_models`) must be wrapped in `asyncio.to_thread` to avoid blocking the event loop.
- **Model ID prefixes matter.** `_infer_provider` relies on the model ID prefix (`anthropic.`, `meta.`, `amazon.`, `mistral.`, `cohere.`). Using an unexpected model ID format returns `None`, which may break provider-specific formatting downstream.
- **`qwen3` models emit thinking tokens.** If using Ollama with a `qwen3` model, pass `reasoning=False` (as the client does) or the answer will be prefixed with reasoning text.
- **`lru_cache` singletons hold state.** `make_openai_llm_client` and `make_ollama_client` are cached. If you change settings at runtime, the cached client will not pick up the change — restart the process or use a fresh factory.
- **Streaming shapes differ per provider.** OpenAI yields `{"content": ...}` per delta; Bedrock collects the full stream then yields; Ollama yields per line. Downstream consumers must handle all three shapes uniformly via the protocol.

---

## 9. Definition of Done

- [ ] [`src/services/llm_client_protocol.py`](../src/services/llm_client_protocol.py) defines a `@runtime_checkable` `LLMClientProtocol` with `get_langchain_model`, `generate_rag_answer`, `generate_rag_answer_stream`, and `health_check`.
- [ ] [`src/services/openai_llm/client.py`](../src/services/openai_llm/client.py) implements `OpenAILLMClient` (default `gpt-4o-mini`) with RAG answer, streaming, and health-check methods.
- [ ] [`src/services/openai_llm/factory.py`](../src/services/openai_llm/factory.py) provides a cached `make_openai_llm_client()` singleton.
- [ ] [`src/services/bedrock_llm/client.py`](../src/services/bedrock_llm/client.py) implements `BedrockLLMClient` using the boto3 `converse` API with `asyncio.to_thread`.
- [ ] [`src/services/bedrock_llm/factory.py`](../src/services/bedrock_llm/factory.py) provides `make_bedrock_llm_client(settings)`.
- [ ] [`src/services/ollama/client.py`](../src/services/ollama/client.py) implements `OllamaClient` with health check, model listing, generate, and streaming.
- [ ] [`src/services/ollama/prompts.py`](../src/services/ollama/prompts.py) provides `RAGPromptBuilder` and `ResponseParser`.
- [ ] [`src/services/ollama/prompts/rag_system.txt`](../src/services/ollama/prompts/rag_system.txt) contains the system prompt.
- [ ] Provider selection in [`src/main.py`](../src/main.py) chooses Bedrock or OpenAI based on `PROVIDER`.
- [ ] Both `OpenAILLMClient` and `BedrockLLMClient` satisfy `LLMClientProtocol` at runtime.
- [ ] Health checks and RAG answer generation verified against at least one live provider.