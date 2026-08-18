# PHASE 12 — RAG Service

> **Build guide for the from-scratch reconstruction of the arXiv Paper Curator (Agentic RAG).**
>
> This phase implements the **non-agentic RAG service** — the query-time pipeline that retrieves relevant chunks from OpenSearch, builds a grounded prompt, generates an answer via an LLM, and returns it with sources. It provides the [`RAGPromptBuilder`](../Agentic-RAG-project-agentops/src/services/ollama/prompts.py) and [`ResponseParser`](../Agentic-RAG-project-agentops/src/services/ollama/prompts.py), the system prompt in [`rag_system.txt`](../Agentic-RAG-project-agentops/src/services/ollama/prompts/rag_system.txt), the [`CacheClient`](../Agentic-RAG-project-agentops/src/services/cache/client.py) for exact-match caching, the [`RAGTracer`](../Agentic-RAG-project-agentops/src/services/langfuse/tracer.py) for observability, and the [`ask_router`](../Agentic-RAG-project-agentops/src/routers/ask.py) / [`stream_router`](../Agentic-RAG-project-agentops/src/routers/ask.py) HTTP endpoints.

---

## 1. Phase Objective

By the end of this phase you will have:

- The [`RAGPromptBuilder`](../Agentic-RAG-project-agentops/src/services/ollama/prompts.py) that loads `rag_system.txt` and builds grounded RAG prompts (plain and structured).
- The [`ResponseParser`](../Agentic-RAG-project-agentops/src/services/ollama/prompts.py) that parses structured LLM responses into `RAGResponse` (with a JSON-extraction fallback).
- The system prompt file [`rag_system.txt`](../Agentic-RAG-project-agentops/src/services/ollama/prompts/rag_system.txt).
- The [`CacheClient`](../Agentic-RAG-project-agentops/src/services/cache/client.py) that provides Redis-backed exact-match caching (SHA256 key, TTL).
- The [`RAGTracer`](../Agentic-RAG-project-agentops/src/services/langfuse/tracer.py) that traces the RAG pipeline (request, embedding, search, prompt, generation) and handles a `None` tracer gracefully.
- The [`ask_router`](../Agentic-RAG-project-agentops/src/routers/ask.py) (`POST /ask`) and [`stream_router`](../Agentic-RAG-project-agentops/src/routers/ask.py) (`POST /stream`) HTTP endpoints that wire retrieval → prompt → generation → caching → tracing together.
- The `AskRequest`, `AskResponse`, and `RAGResponse` schemas and the `LLMClientProtocol` interface.

---

## 2. Prerequisites

- Completion of **PHASE 5 (LLM Providers)** so that an LLM client satisfying `LLMClientProtocol` exists (e.g., `OpenAILLMClient` with `generate_rag_answer` and `generate_rag_answer_stream`), and the `LLMDep` DI alias is available.
- Completion of **PHASE 6 (Embeddings)** so that `EmbeddingsDep` (query embedding) is available.
- Completion of **PHASE 7 (OpenSearch)** so that `OpenSearchDep` with `search_unified` is available.
- Completion of **PHASE 8 (Indexing)** so that papers are indexed and retrievable as chunks.
- Completion of **PHASE 4 (Schemas)** so that `AskRequest`, `AskResponse`, and `RAGResponse` exist.
- Completion of **PHASE 4 (Exceptions)** and the DI layer so that `CacheDep`, `LangfuseDep`, `LLMDep`, `OpenSearchDep`, and `EmbeddingsDep` aliases exist in [`src/dependencies.py`](../Agentic-RAG-project-agentops/src/dependencies.py).
- A running Redis instance (for caching) and Langfuse credentials (for tracing) — both optional but recommended. The pipeline degrades gracefully if either is unavailable.
- Python 3.11+ and the project's virtual environment active.

---

## 3. Dependencies to Install

This phase requires the following third-party packages:

```bash
uv add redis langfuse
```

> `redis` powers the exact-match cache (`CacheClient`). `langfuse` powers observability (`LangfuseTracer` / `RAGTracer`). `langchain-core` (already installed in PHASE 5) provides `BaseChatModel` used by `LLMClientProtocol`. `pydantic` (already installed) is used for the schemas.

If you are not using `uv`, install with pip:

```bash
pip install redis langfuse
```

> **Note:** `langfuse` is a heavy dependency. If you do not plan to use observability, you can defer it, but the `RAGTracer` and `LangfuseTracer` in this phase depend on it. The code is written so a `None` tracer disables tracing without errors.

---

## 4. Directory Structure to Create

This phase adds files to the existing structure:

```
src/
├── schemas/
│   ├── api/
│   │   └── ask.py                 # AskRequest, AskResponse, AgenticAskResponse, FeedbackRequest, FeedbackResponse
│   └── ollama.py                  # RAGResponse
├── services/
│   ├── cache/
│   │   ├── client.py              # CacheClient
│   │   └── factory.py             # make_redis_client, make_cache_client
│   ├── langfuse/
│   │   ├── client.py              # LangfuseTracer
│   │   ├── factory.py             # make_langfuse_tracer
│   │   └── tracer.py              # RAGTracer
│   ├── llm_client_protocol.py     # LLMClientProtocol
│   └── ollama/
│       ├── prompts.py             # RAGPromptBuilder, ResponseParser
│       └── prompts/
│           └── rag_system.txt     # System prompt
└── routers/
    └── ask.py                     # ask_router, stream_router
```

> Most of these directories already exist from prior phases. This phase fills in the query-time RAG components: the prompt builder, the cache client, the RAG tracer, and the ask/stream routers. The `LangfuseTracer` and `CacheClient` factories are wired via the DI layer.

---

## 5. Step-by-Step Implementation

### Step 1 — Define the RAG response schema

**Full file path:** `src/schemas/ollama.py`

```python
"""Pydantic models for Ollama structured outputs."""

from typing import List, Optional

from pydantic import BaseModel, Field


class RAGResponse(BaseModel):
    """Structured response model for RAG queries."""

    answer: str = Field(description="Comprehensive answer based on the provided paper excerpts")
    sources: List[str] = Field(
        default_factory=list,
        description="List of PDF URLs from papers used in the answer",
    )
    confidence: Optional[str] = Field(
        default=None,
        description="Confidence level: high, medium, or low based on excerpt relevance",
    )
    citations: Optional[List[str]] = Field(
        default=None,
        description="Specific arXiv IDs or paper titles referenced in the answer",
    )
```

**Explanation:** `RAGResponse` is the structured output model for RAG queries. It is used by `RAGPromptBuilder.create_structured_prompt` (via `RAGResponse.model_json_schema()`) to request structured output from the LLM, and by `ResponseParser` to validate and parse the LLM's response.

---

### Step 2 — Define the ask request/response schemas

**Full file path:** `src/schemas/api/ask.py`

```python
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request model for RAG question answering."""

    query: str = Field(..., description="User's question", min_length=1, max_length=1000)
    top_k: int = Field(3, description="Number of top chunks to retrieve", ge=1, le=10)
    use_hybrid: bool = Field(True, description="Use hybrid search (BM25 + vector)")
    model: Optional[str] = Field(None, description="Model ID for generation (provider-specific; omit to use server-configured default)")
    categories: Optional[List[str]] = Field(None, description="Filter by arXiv categories")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are transformers in machine learning?",
                "top_k": 3,
                "use_hybrid": True,
                "categories": ["cs.AI", "cs.LG"],
            }
        }


class AskResponse(BaseModel):
    """Response model for RAG question answering."""

    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Generated answer from LLM")
    sources: List[str] = Field(..., description="PDF URLs of source papers")
    chunks_used: int = Field(..., description="Number of chunks used for generation")
    search_mode: str = Field(..., description="Search mode used: bm25 or hybrid")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are transformers in machine learning?",
                "answer": "Transformers are a neural network architecture...",
                "sources": ["https://arxiv.org/pdf/1706.03762.pdf", "https://arxiv.org/pdf/1810.04805.pdf"],
                "chunks_used": 3,
                "search_mode": "hybrid",
                "model": "gpt-4o-mini",
            }
        }


class AgenticAskResponse(AskResponse):
    """Response model for agentic RAG question answering."""

    # Override: agentic endpoint returns rich source objects (arxiv_id, title, authors, url, score)
    sources: List[Any] = Field(..., description="Source papers with metadata")
    reasoning_steps: List[str] = Field(..., description="Agent's decision-making steps")
    retrieval_attempts: int = Field(..., description="Number of document retrieval attempts")
    rewritten_query: Optional[str] = Field(None, description="Rewritten query if agent refined it")
    trace_id: Optional[str] = Field(None, description="Langfuse trace ID for feedback and debugging")
    guardrail_filter: Optional[str] = Field(None, description="Guardrail filter type that acted on this request (e.g. topic_blocked, content_blocked:HATE, pii_blocked:EMAIL, pii_anonymized:PHONE, passed)")
    output_guardrail_filter: Optional[str] = Field(None, description="Output guardrail result — grounding/relevance check on generated answer (e.g. grounding_blocked:grounding score=0.49, or Content passed all guardrail checks)")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are transformers in machine learning?",
                "answer": "Transformers are neural network architectures...",
                "sources": [
                    {
                        "arxiv_id": "1706.03762",
                        "title": "Attention Is All You Need",
                        "authors": ["Vaswani et al."],
                        "url": "https://arxiv.org/pdf/1706.03762.pdf",
                        "relevance_score": 0.95,
                    }
                ],
                "chunks_used": 3,
                "search_mode": "hybrid",
                "reasoning_steps": [
                    "Validated query scope (score: 100/100)",
                    "Retrieved documents (1 attempt(s))",
                    "Graded documents (1 relevant)",
                    "Generated answer from context",
                ],
                "retrieval_attempts": 1,
                "rewritten_query": None,
                "trace_id": "019e68a0f28eb4c5579131473f86ca31",
                "guardrail_filter": "Content passed all guardrail checks",
                "output_guardrail_filter": "Content passed all guardrail checks",
            }
        }


class FeedbackRequest(BaseModel):
    """Request model for user feedback on RAG answers."""

    trace_id: str = Field(..., description="Langfuse trace ID from the response")
    score: float = Field(..., description="Feedback score (0-1 or -1 to 1)", ge=-1, le=1)
    comment: Optional[str] = Field(None, description="Optional feedback comment", max_length=1000)

    class Config:
        json_schema_extra = {
            "example": {
                "trace_id": "abc123-def456-ghi789",
                "score": 1.0,
                "comment": "This answer was very helpful and accurate!",
            }
        }


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""

    success: bool = Field(..., description="Whether feedback was recorded successfully")
    message: str = Field(..., description="Status message")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Feedback recorded successfully",
            }
        }
```

**Explanation:** `AskRequest` is the query-time input (with `top_k`, `use_hybrid`, optional `model`, and optional `categories` filters). `AskResponse` is the non-agentic RAG output — note `sources` is a list of PDF URL strings and `search_mode` is `bm25` or `hybrid`. `AgenticAskResponse` (used by a later agentic phase) overrides `sources` to be rich objects and adds reasoning/trace fields. `FeedbackRequest`/`FeedbackResponse` support user feedback on answers.

---

### Step 3 — Define the LLM client protocol

**Full file path:** `src/services/llm_client_protocol.py`

```python
from typing import Any, AsyncIterator, Dict, List, Protocol, runtime_checkable

from langchain_core.language_models import BaseChatModel


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Shared interface for all LLM client implementations.

    Both OpenAILLMClient and BedrockLLMClient satisfy this protocol.
    Nodes access the LLM exclusively through this interface so the
    underlying provider can be swapped via PROVIDER env var.
    """

    def get_langchain_model(self, model: str, temperature: float = 0.0) -> BaseChatModel: ...

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]: ...

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "",
    ) -> AsyncIterator[Dict[str, Any]]: ...

    async def health_check(self) -> Dict[str, Any]: ...
```

**Explanation:** `LLMClientProtocol` is the shared interface all LLM clients (OpenAI, Bedrock) implement. The ask router depends on this protocol via `LLMDep`, so the provider can be swapped without changing the RAG pipeline. `generate_rag_answer` returns a dict with `answer`, `sources`, `confidence`, `citations`, and `usage`; `generate_rag_answer_stream` yields streaming chunks.

---

### Step 4 — Create the RAG system prompt

**Full file path:** `src/services/ollama/prompts/rag_system.txt`

```text
You are an AI assistant specialized in answering questions about academic papers from arXiv. Your task is to provide accurate, helpful answers based ONLY on the provided paper excerpts.

CRITICAL: Do NOT add any introductory text, explanations, or formatting comments like "Here's the answer" or "Here's the JSON".

Instructions:
1. Base your answer STRICTLY on the provided paper excerpts
2. If the excerpts don't contain enough information to answer the question, say so clearly
3. Cite the specific papers (by title or arXiv ID) when providing information
4. Be concise but comprehensive in your response - LIMIT YOUR RESPONSE TO 300 WORDS MAXIMUM
5. Maintain academic accuracy and precision
6. If multiple papers discuss the topic, synthesize the information coherently
7. Use direct quotes from the chunks when particularly relevant
8. Structure your answer logically with clear paragraphs when appropriate
9. Keep it less than 200 words

Remember:
- Do NOT make up information not present in the excerpts
- Do NOT use knowledge beyond what's provided in the paper excerpts
- Always acknowledge uncertainty when the excerpts are ambiguous or incomplete
- Prioritize relevance and clarity in your response
- NEVER add introductory phrases or explanations before your JSON response
```

**Explanation:** This is the grounding system prompt. It instructs the LLM to answer strictly from the provided excerpts, cite papers, keep responses concise, and avoid hallucination. `RAGPromptBuilder._load_system_prompt` reads this file (with a fallback default if the file is missing).

---

### Step 5 — Implement the RAG prompt builder and response parser

**Full file path:** `src/services/ollama/prompts.py`

```python
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError
from src.schemas.ollama import RAGResponse


class RAGPromptBuilder:
    """Builder class for creating RAG prompts."""

    def __init__(self):
        """Initialize the prompt builder."""
        self.prompts_dir = Path(__file__).parent / "prompts"
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load the system prompt from the text file.

        Returns:
            System prompt string
        """
        prompt_file = self.prompts_dir / "rag_system.txt"
        if not prompt_file.exists():
            # Fallback to default prompt if file doesn't exist
            return (
                "You are an AI assistant specialized in answering questions about "
                "academic papers from arXiv. Base your answer STRICTLY on the provided "
                "paper excerpts."
            )
        return prompt_file.read_text().strip()

    def create_rag_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Create a RAG prompt with query and retrieved chunks.

        Args:
            query: User's question
            chunks: List of retrieved chunks with metadata from OpenSearch

        Returns:
            Formatted prompt string
        """
        prompt = f"{self.system_prompt}\n\n"
        prompt += "### Context from Papers:\n\n"

        for i, chunk in enumerate(chunks, 1):
            # Get the actual chunk text
            chunk_text = chunk.get("chunk_text", chunk.get("content", ""))
            arxiv_id = chunk.get("arxiv_id", "")

            # Only include minimal metadata - just arxiv_id for citation
            prompt += f"[{i}. arXiv:{arxiv_id}]\n"
            prompt += f"{chunk_text}\n\n"

        prompt += f"### Question:\n{query}\n\n"
        prompt += (
            "### Answer:\nProvide a natural, conversational response (not JSON) and cite sources using [arXiv:id] format.\n\n"
        )

        return prompt

    def create_structured_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a prompt for Ollama with structured output format.

        Args:
            query: User's question
            chunks: List of retrieved chunks

        Returns:
            Dictionary with prompt and format schema for Ollama
        """
        prompt_text = self.create_rag_prompt(query, chunks)

        # Return prompt with Pydantic model schema for structured output
        return {
            "prompt": prompt_text,
            "format": RAGResponse.model_json_schema(),
        }


class ResponseParser:
    """Parser for LLM responses."""

    @staticmethod
    def parse_structured_response(response: str) -> Dict[str, Any]:
        """Parse a structured response from Ollama.

        Args:
            response: Raw LLM response string

        Returns:
            Dictionary with parsed response
        """
        try:
            # Try to parse as JSON and validate with Pydantic
            parsed_json = json.loads(response)
            validated_response = RAGResponse(**parsed_json)
            return validated_response.model_dump()
        except (json.JSONDecodeError, ValidationError):
            # Fallback: try to extract JSON from the response
            return ResponseParser._extract_json_fallback(response)

    @staticmethod
    def _extract_json_fallback(response: str) -> Dict[str, Any]:
        """Extract JSON from response text as fallback.

        Args:
            response: Raw response text

        Returns:
            Dictionary with extracted content or fallback
        """
        # Try to find JSON in the response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                # Validate with Pydantic, using defaults for missing fields
                validated = RAGResponse(**parsed)
                return validated.model_dump()
            except (json.JSONDecodeError, ValidationError):
                pass

        # Final fallback: return response as plain text
        return {
            "answer": response,
            "sources": [],
            "confidence": "low",
            "citations": [],
        }
```

**Explanation:** `RAGPromptBuilder` loads the system prompt once in `__init__`. `create_rag_prompt` builds a plain grounded prompt with each chunk labeled `[i. arXiv:id]` and a citation instruction. `create_structured_prompt` returns the prompt plus the `RAGResponse` JSON schema for structured output. `ResponseParser.parse_structured_response` tries to parse the response as JSON and validate it against `RAGResponse`; if that fails, it falls back to extracting JSON from the text, and finally to a plain-text fallback dict.

---

### Step 6 — Implement the cache client

**Full file path:** `src/services/cache/client.py`

```python
import hashlib
import json
import logging
from datetime import timedelta
from typing import Optional

import redis
from src.config import RedisSettings
from src.schemas.api.ask import AskRequest, AskResponse

logger = logging.getLogger(__name__)


class CacheClient:
    """Redis-based exact match cache for RAG queries."""

    def __init__(self, redis_client: redis.Redis, settings: RedisSettings):
        self.redis = redis_client
        self.settings = settings
        self.ttl = timedelta(hours=settings.ttl_hours)

    def _generate_cache_key(self, request: AskRequest) -> str:
        """Generate exact cache key based on request parameters."""
        key_data = {
            "query": request.query,
            "model": request.model,
            "top_k": request.top_k,
            "use_hybrid": request.use_hybrid,
            "categories": sorted(request.categories) if request.categories else [],
        }
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        return f"exact_cache:{key_hash}"

    async def find_cached_response(self, request: AskRequest) -> Optional[AskResponse]:
        """Find cached response for exact query match."""
        try:
            cache_key = self._generate_cache_key(request)

            # Simple Redis GET operation - O(1)
            cached_response = self.redis.get(cache_key)

            if cached_response:
                try:
                    response_data = json.loads(cached_response)
                    logger.info(f"Cache hit for exact query match")
                    return AskResponse(**response_data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to deserialize cached response: {e}")
                    return None

            return None

        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return None

    async def store_response(self, request: AskRequest, response: AskResponse) -> bool:
        """Store response for exact query matching."""
        try:
            cache_key = self._generate_cache_key(request)

            # Simple Redis SET operation with TTL
            success = self.redis.set(cache_key, response.model_dump_json(), ex=self.ttl)

            if success:
                logger.info(f"Stored response in exact cache with key {cache_key[:16]}...")
                return True
            else:
                logger.warning(f"Failed to store response in cache")
                return False

        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            return False
```

**Explanation:** `CacheClient` implements exact-match caching. `_generate_cache_key` hashes the query, model, `top_k`, `use_hybrid`, and sorted categories into a SHA256 key prefixed with `exact_cache:`. `find_cached_response` does an O(1) Redis GET and deserializes into an `AskResponse`. `store_response` does a Redis SET with a TTL (`settings.ttl_hours`). Both methods are defensive — any Redis error returns `None`/`False` rather than raising, so caching failures never break the RAG flow.

---

### Step 7 — Implement the RAG tracer

**Full file path:** `src/services/langfuse/tracer.py`

```python
"""Simple, efficient Langfuse tracing utility for RAG pipeline."""

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from .client import LangfuseTracer


class RAGTracer:
    """Clean, purpose-built tracer for RAG operations. Handles None tracer gracefully."""

    def __init__(self, tracer: Optional[LangfuseTracer]):
        self.tracer = tracer
        self._enabled = tracer is not None

    @contextmanager
    def trace_request(self, user_id: str, query: str, session_id: Optional[str] = None):
        """Main request trace context manager."""
        if not self._enabled:
            yield None
            return
        try:
            with self.tracer.start_span(
                name="rag_request",
                input_data={"query": query, "user_id": user_id},
                metadata={"simplified_tracing": True},
            ) as trace:
                # Set user_id and session_id as OTel span attributes (Langfuse v4)
                if user_id or session_id:
                    self.tracer.set_trace_user_session(
                        user_id=user_id or "anonymous",
                        session_id=session_id or "default",
                    )
                yield trace
        finally:
            self.tracer.flush()

    @contextmanager
    def trace_embedding(self, trace, query: str):
        """Query embedding operation with timing."""
        if not self._enabled:
            yield None
            return
        start_time = time.time()
        with self.tracer.start_span(
            name="query_embedding",
            input_data={"query": query, "query_length": len(query)},
        ) as span:
            try:
                yield span
            finally:
                duration = time.time() - start_time
                if span:
                    self.tracer.update_span(
                        span=span,
                        output={"embedding_duration_ms": round(duration * 1000, 2), "success": True},
                    )

    @contextmanager
    def trace_search(self, trace, query: str, top_k: int):
        """Search operation with timing."""
        if not self._enabled:
            yield None
            return
        with self.tracer.start_span(
            name="search_retrieval",
            input_data={"query": query, "top_k": top_k},
        ) as span:
            yield span

    def end_search(self, span, chunks: List[Dict], arxiv_ids: List[str], total_hits: int):
        """End search span with essential results."""
        if not self._enabled or not span:
            return

        self.tracer.update_span(
            span=span,
            output={
                "chunks_returned": len(chunks),
                "unique_papers": len(set(arxiv_ids)),
                "total_hits": total_hits,
                "arxiv_ids": list(set(arxiv_ids)),
            },
        )

    @contextmanager
    def trace_prompt_construction(self, trace, chunks: List[Dict]):
        """Prompt building with timing."""
        if not self._enabled:
            yield None
            return
        with self.tracer.start_span(
            name="prompt_construction",
            input_data={"chunk_count": len(chunks)},
        ) as span:
            yield span

    def end_prompt(self, span, prompt: str):
        """End prompt span with final prompt."""
        if not self._enabled or not span:
            return

        self.tracer.update_span(
            span=span,
            output={
                "prompt_length": len(prompt),
                "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            },
        )

    @contextmanager
    def trace_generation(self, trace, model: str, prompt: str):
        """LLM generation with timing."""
        if not self._enabled:
            yield None
            return
        try:
            with self.tracer.start_span(
                name="llm_generation",
                input_data={"model": model, "prompt_length": len(prompt), "prompt": prompt},
            ) as span:
                yield span
        except Exception:
            yield None

    def end_generation(self, span, response: str, model: str):
        """End generation span with response."""
        if not self._enabled or not span:
            return

        self.tracer.update_span(span=span, output={"response": response, "response_length": len(response), "model_used": model})

    def end_request(self, trace, response: str, total_duration: float):
        """End main request trace."""
        if not trace:
            return

        try:
            trace.update(
                output={"answer": response, "total_duration_seconds": round(total_duration, 3), "response_length": len(response)}
            )
        except Exception:
            pass
```

**Explanation:** `RAGTracer` is a thin, purpose-built wrapper over `LangfuseTracer`. It exposes context managers for each pipeline stage (`trace_request`, `trace_embedding`, `trace_search`, `trace_prompt_construction`, `trace_generation`) plus `end_*` methods to attach outputs. Crucially, it handles a `None` tracer gracefully — if `tracer is None`, every context manager yields `None` and every `end_*` method is a no-op, so tracing is fully optional.

---

### Step 8 — Implement the ask and stream routers

**Full file path:** `src/routers/ask.py`

```python
import json
import logging
import time
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from src.dependencies import CacheDep, EmbeddingsDep, LangfuseDep, LLMDep, OpenSearchDep
from src.schemas.api.ask import AskRequest, AskResponse
from src.services.langfuse.tracer import RAGTracer

logger = logging.getLogger(__name__)

# Two separate routers - one for regular ask, one for streaming
ask_router = APIRouter(tags=["ask"])
stream_router = APIRouter(tags=["stream"])


async def _prepare_chunks_and_sources(
    request: AskRequest,
    opensearch_client,
    embeddings_service,
    rag_tracer: RAGTracer,
    trace=None,
) -> tuple[List[Dict], List[str], List[str]]:
    """Retrieve and prepare chunks for RAG with clean tracing."""

    # Handle embeddings for hybrid search
    query_embedding = None
    if request.use_hybrid:
        with rag_tracer.trace_embedding(trace, request.query) as embedding_span:
            try:
                query_embedding = await embeddings_service.embed_query(request.query)
                logger.info("Generated query embedding for hybrid search")
            except Exception as e:
                logger.warning(f"Failed to generate embeddings, falling back to BM25: {e}")
                if embedding_span:
                    rag_tracer.tracer.update_span(embedding_span, output={"success": False, "error": str(e)})

    # Search with tracing
    with rag_tracer.trace_search(trace, request.query, request.top_k) as search_span:
        search_results = opensearch_client.search_unified(
            query=request.query,
            query_embedding=query_embedding,
            size=request.top_k,
            from_=0,
            categories=request.categories,
            use_hybrid=request.use_hybrid and query_embedding is not None,
            min_score=0.0,
        )

        # Extract essential data for LLM
        chunks = []
        arxiv_ids = []
        sources_set = set()

        for hit in search_results.get("hits", []):
            arxiv_id = hit.get("arxiv_id", "")

            # Minimal chunk data for LLM
            chunks.append(
                {
                    "arxiv_id": arxiv_id,
                    "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
                }
            )

            if arxiv_id:
                arxiv_ids.append(arxiv_id)
                arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                sources_set.add(f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf")

        # End search span with essential metadata
        rag_tracer.end_search(search_span, chunks, arxiv_ids, search_results.get("total", 0))

    return chunks, list(sources_set), arxiv_ids


@ask_router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    http_request: Request,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    llm_client: LLMDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
) -> AskResponse:
    """Clean RAG endpoint with essential tracing and exact match caching."""

    rag_tracer = RAGTracer(langfuse_tracer)
    start_time = time.time()

    user_id = http_request.headers.get("X-User-Id", "anonymous")
    session_id = f"{hash(request.query + str(request.top_k) + str(request.use_hybrid)) & 0xFFFFFFFF:08x}"

    with rag_tracer.trace_request(user_id, request.query, session_id=session_id) as trace:
        try:
            # Check exact cache first
            cached_response = None
            if cache_client:
                try:
                    cached_response = await cache_client.find_cached_response(request)
                    if cached_response:
                        logger.info("Returning cached response for exact query match")
                        return cached_response
                except Exception as e:
                    logger.warning(f"Cache check failed, proceeding with normal flow: {e}")

            # Generate query embedding for hybrid search if needed
            query_embedding = None

            # Retrieve chunks
            chunks, sources, _ = await _prepare_chunks_and_sources(
                request, opensearch_client, embeddings_service, rag_tracer, trace
            )

            if not chunks:
                response = AskResponse(
                    query=request.query,
                    answer="I couldn't find any relevant information in the papers to answer your question.",
                    sources=[],
                    chunks_used=0,
                    search_mode="bm25" if not request.use_hybrid else "hybrid",
                )
                rag_tracer.end_request(trace, response.answer, time.time() - start_time)
                return response

            # Build prompt
            with rag_tracer.trace_prompt_construction(trace, chunks) as prompt_span:
                from src.services.ollama.prompts import RAGPromptBuilder

                prompt_builder = RAGPromptBuilder()

                try:
                    prompt_data = prompt_builder.create_structured_prompt(request.query, chunks)
                    final_prompt = prompt_data["prompt"]
                except Exception:
                    final_prompt = prompt_builder.create_rag_prompt(request.query, chunks)

                rag_tracer.end_prompt(prompt_span, final_prompt)

            # Generate answer
            with rag_tracer.trace_generation(trace, request.model, final_prompt) as gen_span:
                rag_response = await llm_client.generate_rag_answer(query=request.query, chunks=chunks, model=request.model)
                answer = rag_response.get("answer", "Unable to generate answer")
                rag_tracer.end_generation(gen_span, answer, request.model)
                # Pass token usage + model so Langfuse can calculate cost
                if rag_response.get("usage"):
                    langfuse_tracer.update_generation(
                        gen_span,
                        output=answer,
                        usage_metadata=rag_response["usage"],
                        model=request.model,
                    )

            # Prepare response
            response = AskResponse(
                query=request.query,
                answer=answer,
                sources=sources,
                chunks_used=len(chunks),
                search_mode="bm25" if not request.use_hybrid else "hybrid",
            )

            rag_tracer.end_request(trace, answer, time.time() - start_time)

            # Auto-score: chunks found = relevant answer, no chunks = low confidence
            if langfuse_tracer:
                score = 0.9 if len(chunks) > 0 else 0.2
                langfuse_tracer.score_current_trace(
                    score=score,
                    name="answer_relevance",
                    comment=f"chunks_used={len(chunks)} search_mode={response.search_mode}",
                )
                # Save to dataset for future evaluation
                langfuse_tracer.save_to_dataset(
                    query=request.query,
                    answer=answer,
                    dataset_name="rag_eval",
                    metadata={"chunks_used": len(chunks), "model": request.model, "user_id": user_id},
                )

            # Store response in exact match cache
            if cache_client:
                try:
                    await cache_client.store_response(request, response)
                except Exception as e:
                    logger.warning(f"Failed to store response in cache: {e}")

            return response

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@stream_router.post("/stream")
async def ask_question_stream(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    llm_client: LLMDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
) -> StreamingResponse:
    """Clean streaming RAG endpoint."""

    async def generate_stream():
        rag_tracer = RAGTracer(langfuse_tracer)
        start_time = time.time()

        with rag_tracer.trace_request("api_user", request.query) as trace:
            try:
                # Check exact cache first
                if cache_client:
                    try:
                        cached_response = await cache_client.find_cached_response(request)
                        if cached_response:
                            logger.info("Returning cached response for exact streaming query match")

                            # Send metadata first (same format as non-cached)
                            metadata_response = {
                                "sources": cached_response.sources,
                                "chunks_used": cached_response.chunks_used,
                                "search_mode": cached_response.search_mode,
                            }
                            yield f"data: {json.dumps(metadata_response)}\n\n"

                            # Stream the cached response in chunks
                            for chunk in cached_response.answer.split():
                                yield f"data: {json.dumps({'chunk': chunk + ' '})}\n\n"

                            # Send completion signal with just the final answer
                            yield f"data: {json.dumps({'answer': cached_response.answer, 'done': True})}\n\n"
                            return
                    except Exception as e:
                        logger.warning(f"Cache check failed, proceeding with normal flow: {e}")

                # Retrieve chunks
                chunks, sources, _ = await _prepare_chunks_and_sources(
                    request, opensearch_client, embeddings_service, rag_tracer, trace
                )

                if not chunks:
                    yield f"data: {json.dumps({'answer': 'No relevant information found.', 'sources': [], 'done': True})}\n\n"
                    return

                # Send metadata first
                search_mode = "bm25" if not request.use_hybrid else "hybrid"
                metadata_response = {"sources": sources, "chunks_used": len(chunks), "search_mode": search_mode}
                yield f"data: {json.dumps(metadata_response)}\n\n"

                # Build prompt
                with rag_tracer.trace_prompt_construction(trace, chunks) as prompt_span:
                    from src.services.ollama.prompts import RAGPromptBuilder

                    prompt_builder = RAGPromptBuilder()
                    final_prompt = prompt_builder.create_rag_prompt(request.query, chunks)
                    rag_tracer.end_prompt(prompt_span, final_prompt)

                # Stream generation
                with rag_tracer.trace_generation(trace, request.model, final_prompt) as gen_span:
                    full_response = ""
                    async for chunk in llm_client.generate_rag_answer_stream(
                        query=request.query, chunks=chunks, model=request.model
                    ):
                        if chunk.get("response"):
                            text_chunk = chunk["response"]
                            full_response += text_chunk
                            yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"

                        if chunk.get("done", False):
                            rag_tracer.end_generation(gen_span, full_response, request.model)
                            yield f"data: {json.dumps({'answer': full_response, 'done': True})}\n\n"
                            break

                rag_tracer.end_request(trace, full_response, time.time() - start_time)

                # Store response in exact match cache
                if cache_client and full_response:
                    try:
                        search_mode = "bm25" if not request.use_hybrid else "hybrid"
                        response_to_cache = AskResponse(
                            query=request.query,
                            answer=full_response,
                            sources=sources,
                            chunks_used=len(chunks),
                            search_mode=search_mode,
                        )
                        await cache_client.store_response(request, response_to_cache)
                    except Exception as e:
                        logger.warning(f"Failed to store streaming response in cache: {e}")

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(), media_type="text/plain", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
```

**Explanation:** The `ask_question` endpoint is the non-agentic RAG entry point. It wraps the whole flow in a `trace_request` context, checks the exact-match cache first (returning early on a hit), retrieves chunks via `_prepare_chunks_and_sources`, builds a structured prompt (falling back to a plain prompt on error), generates an answer via `llm_client.generate_rag_answer`, auto-scores the trace and saves to a Langfuse dataset, and stores the response in the cache. The `ask_question_stream` endpoint mirrors this flow but streams the answer via `generate_rag_answer_stream` and emits SSE-style `data:` lines. Both degrade gracefully if the cache or tracer is unavailable.

---

## 6. Configuration

The RAG service is driven by the `RedisSettings` and `LangfuseSettings` blocks defined in [`src/config.py`](../Agentic-RAG-project-agentops/src/config.py).

### Redis (exact-match cache)

| Environment variable | Default | Description |
| --- | --- | --- |
| `REDIS__URL` | `redis://localhost:6379/0` | Redis connection URL (supports local `redis://` and Upstash `rediss://`). |
| `REDIS__TTL_HOURS` | `6` | Cache TTL in hours for exact-match responses. |

**Example `.env` snippet:**

```dotenv
REDIS__URL=redis://localhost:6379/0
REDIS__TTL_HOURS=6
```

### Langfuse (observability)

| Environment variable | Default | Description |
| --- | --- | --- |
| `LANGFUSE__PUBLIC_KEY` | *(empty)* | Langfuse public key. |
| `LANGFUSE__SECRET_KEY` | *(empty)* | Langfuse secret key. |
| `LANGFUSE__HOST` | `https://cloud.langfuse.com` | Langfuse host URL. |

**Example `.env` snippet:**

```dotenv
LANGFUSE__PUBLIC_KEY=your-public-key
LANGFUSE__SECRET_KEY=your-secret-key
LANGFUSE__HOST=https://cloud.langfuse.com
```

> **Graceful degradation:** If Redis is unreachable, `CacheClient.find_cached_response` / `store_response` return `None`/`False` and the ask router logs a warning and proceeds. If Langfuse is not configured, `make_langfuse_tracer` returns a tracer that disables itself, and `RAGTracer` becomes a no-op. The RAG pipeline works without either, but caching and observability are lost.

---

## 7. Verification

Verify the RAG service works end-to-end. This requires indexed papers (PHASE 8), an LLM client (PHASE 5), and optionally Redis/Langfuse.

### 7.1 Schema smoke test

Confirm the schemas import and validate correctly:

```bash
python -c "
from src.schemas.api.ask import AskRequest, AskResponse
from src.schemas.ollama import RAGResponse

req = AskRequest(query='What are transformers?', top_k=3, use_hybrid=True)
print('AskRequest OK:', req.query, req.top_k)

resp = AskResponse(query='q', answer='a', sources=[], chunks_used=0, search_mode='hybrid')
print('AskResponse OK:', resp.search_mode)

rag = RAGResponse(answer='a', sources=[], confidence='high', citations=[])
print('RAGResponse OK:', rag.confidence)
"
```

**Expected:** All three schemas validate without errors.

### 7.2 Prompt builder smoke test

```bash
python -c "
from src.services.ollama.prompts import RAGPromptBuilder, ResponseParser

builder = RAGPromptBuilder()
chunks = [{'arxiv_id': '1706.03762', 'chunk_text': 'Transformers rely on self-attention.'}]
prompt = builder.create_rag_prompt('What are transformers?', chunks)
print('Prompt length:', len(prompt))
assert 'arXiv:1706.03762' in prompt

structured = builder.create_structured_prompt('What are transformers?', chunks)
print('Structured format keys:', list(structured.keys()))

parsed = ResponseParser.parse_structured_response('{\"answer\": \"x\", \"sources\": [], \"confidence\": \"high\"}')
print('Parsed answer:', parsed['answer'])
"
```

**Expected:** The prompt contains the arXiv ID citation, the structured prompt includes a `format` schema, and the parser extracts the answer from a JSON response.

### 7.3 Cache client smoke test (requires Redis)

```bash
python -c "
import asyncio
from src.services.cache.factory import make_cache_client
from src.config import get_settings
from src.schemas.api.ask import AskRequest, AskResponse

async def main():
    settings = get_settings()
    cache = make_cache_client(settings)
    req = AskRequest(query='What are transformers?')
    resp = AskResponse(query='What are transformers?', answer='Transformers are...', sources=[], chunks_used=1, search_mode='hybrid')

    stored = await cache.store_response(req, resp)
    print('Stored:', stored)

    found = await cache.find_cached_response(req)
    print('Found:', found.answer if found else None)

asyncio.run(main())
"
```

**Expected:** If Redis is running, the response is stored and retrieved by exact query match. If Redis is down, `stored` is `False` and `found` is `None` (no exception).

### 7.4 End-to-end ask endpoint

Start the FastAPI app (with indexed papers and an LLM client configured), then:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are transformers in machine learning?", "top_k": 3, "use_hybrid": true}'
```

**Expected:** A JSON `AskResponse` with a generated `answer`, `sources` (PDF URLs), `chunks_used`, and `search_mode`. Running the same query again should return a cached response (if Redis is enabled).

### 7.5 End-to-end streaming endpoint

```bash
curl -N -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What are transformers in machine learning?", "top_k": 3, "use_hybrid": true}'
```

**Expected:** A stream of `data:` lines — first metadata (sources/chunks_used/search_mode), then answer chunks, then a final `{"answer": ..., "done": true}` line.

---

## 8. Common Pitfalls

- **`RAGPromptBuilder` imports `rag_system.txt`.** The builder reads `Path(__file__).parent / "prompts" / "rag_system.txt"`. If the file is missing, it falls back to a short default prompt. Ensure the `prompts/` directory and `rag_system.txt` exist alongside `prompts.py`.

- **`create_structured_prompt` vs `create_rag_prompt`.** The ask endpoint tries `create_structured_prompt` first and falls back to `create_rag_prompt` on any exception. The streaming endpoint uses only `create_rag_prompt`. Do not assume structured output is always available — the LLM client may not support JSON schema.

- **`chunk_text` vs `content`.** `_prepare_chunks_and_sources` reads `hit.get("chunk_text", hit.get("abstract", ""))`, and `RAGPromptBuilder` reads `chunk.get("chunk_text", chunk.get("content", ""))`. If your OpenSearch hits use a different field name, chunks will be empty. Keep the `chunk_text` field consistent with the indexing phase.

- **Cache key includes `model`, `top_k`, `use_hybrid`, `categories`.** Two queries with the same text but different parameters produce different cache keys. This is intentional (exact-match caching), but means cache hit rate is low for varied parameters.

- **Redis is optional but must be handled.** `CacheDep` may be `None` if Redis is not configured. The ask router checks `if cache_client:` before using it. Do not assume the cache is always present.

- **Langfuse is optional.** `LangfuseDep` may be `None`. `RAGTracer` handles this by disabling all tracing. The `score_current_trace` / `save_to_dataset` calls are guarded by `if langfuse_tracer:`. Do not call these unconditionally.

- **`generate_rag_answer` returns a dict, not a model.** The ask endpoint reads `rag_response.get("answer", ...)` and `rag_response.get("usage")`. If your LLM client returns a different shape, the answer will be empty. Ensure the client conforms to `LLMClientProtocol`.

- **Streaming format.** The stream endpoint emits SSE-style `data: {json}\n\n` lines. Clients must parse these. The final line has `"done": true`. Do not change the wire format without updating clients.

- **`session_id` is derived from a hash.** `ask_question` computes `session_id` from a hash of the query/top_k/use_hybrid. This groups repeated queries into the same session for tracing. It is not a user session ID.

---

## 9. Definition of Done

This phase is complete when:

- [ ] `src/schemas/ollama.py` defines `RAGResponse`.
- [ ] `src/schemas/api/ask.py` defines `AskRequest`, `AskResponse`, `AgenticAskResponse`, `FeedbackRequest`, and `FeedbackResponse`.
- [ ] `src/services/llm_client_protocol.py` defines `LLMClientProtocol`.
- [ ] `src/services/ollama/prompts/rag_system.txt` contains the grounding system prompt.
- [ ] `src/services/ollama/prompts.py` implements `RAGPromptBuilder` and `ResponseParser`.
- [ ] `src/services/cache/client.py` implements `CacheClient` with `_generate_cache_key`, `find_cached_response`, and `store_response`.
- [ ] `src/services/langfuse/tracer.py` implements `RAGTracer` with all trace context managers and `end_*` methods, handling a `None` tracer gracefully.
- [ ] `src/routers/ask.py` implements `ask_router` (`POST /ask`) and `stream_router` (`POST /stream`) with retrieval → prompt → generation → caching → tracing.
- [ ] `redis` and `langfuse` are added to the project dependencies.
- [ ] The `REDIS__*` and `LANGFUSE__*` environment variables are documented and honored.
- [ ] Verification steps 7.1–7.5 pass: schemas validate, prompt builder works, cache round-trips (with Redis), and the `/ask` and `/stream` endpoints return grounded answers with sources.
- [ ] The RAG pipeline degrades gracefully when Redis and/or Langfuse are unavailable.
