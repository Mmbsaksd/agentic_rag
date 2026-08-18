# PHASE 16 — API Routers

> Build guide for exposing the agentic RAG service (PHASE 15) and the existing RAG/search infrastructure over HTTP. This phase creates the FastAPI dependency provider, the agentic/supervisor/A2A routers, the request/response schemas, and wires everything into `main.py`.

---

## 1. Phase Objective

In PHASE 15 you built the `AgenticRAGService`, `SupervisorAgent`, and `SummarizerAgent`. In this phase you will expose them (and the existing search/health infrastructure) over HTTP by:

1. Creating the **API schemas** (`ask.py`, `health.py`, `search.py`) that define request/response models, including the rich `AgenticAskResponse`.
2. Creating the **dependency provider** (`dependencies.py`) with `Annotated`/`Depends` aliases and the `AgenticRAGDep` provider.
3. Creating the **agentic RAG router** (`agentic_ask.py`) with `POST /api/v1/ask-agentic` and `POST /api/v1/feedback`.
4. Creating the **supervisor router** (`supervisor_ask.py`) with `POST /api/v1/ask-supervisor`.
5. Creating the **A2A router** (`a2a.py`) with `GET /.well-known/agent.json` and `POST /a2a/tasks/send`.
6. Creating the **hybrid search router** (`hybrid_search.py`) and the **health router** (`ping.py`).
7. Creating the **A2A models** (`services/a2a/models.py`).
8. Wiring all routers and `app.state` services into **`main.py`**.

By the end of this phase the full agentic RAG system is reachable via HTTP.

---

## 2. Prerequisites

- PHASE 15 completed: `AgenticRAGService`, `make_agentic_rag_service`, `SupervisorAgent`, and `SummarizerAgent` exist in [`src/services/agents/`](Agentic-RAG-project-agentops/src/services/agents/).
- PHASE 13 completed: `Context` dataclass is available for building the supervisor context.
- PHASE 7 completed: `OpenSearchClient` with `search_unified()` and `health_check()`.
- PHASE 6 completed: `JinaEmbeddingsClient` with `embed_query()`.
- PHASE 5 completed: `LLMClientProtocol` and `OpenAILLMClient`.
- PHASE 12 completed: `LangfuseTracer` with `submit_feedback()` and `flush()`.
- Existing routers from earlier phases (`ask.py`, `stream_router`) are present.

---

## 3. Dependencies to Install

No new dependencies are required for this phase. All packages (FastAPI, Pydantic, SQLAlchemy, Langfuse, LangGraph) were installed in earlier phases. Confirm FastAPI and Pydantic are available:

```bash
pip install fastapi pydantic
```

---

## 4. Directory Structure to Create

All files live in existing packages. No new directories are required:

```
src/
├── dependencies.py                 # dependency provider + Annotated aliases (NEW)
├── main.py                         # wire routers + app.state (UPDATE)
├── schemas/api/
│   ├── ask.py                      # AskRequest, AskResponse, AgenticAskResponse, Feedback* (NEW)
│   ├── health.py                   # ServiceStatus, HealthResponse (NEW)
│   └── search.py                   # SearchRequest, HybridSearchRequest, SearchHit, SearchResponse (NEW)
├── services/a2a/
│   └── models.py                   # A2A protocol models (NEW)
└── routers/
    ├── agentic_ask.py              # /ask-agentic, /feedback (NEW)
    ├── supervisor_ask.py           # /ask-supervisor (NEW)
    ├── a2a.py                      # /.well-known/agent.json, /a2a/tasks/send (NEW)
    ├── hybrid_search.py            # /hybrid-search/ (NEW)
    └── ping.py                     # /health (NEW)
```

---

## 5. Step-by-Step Implementation

### Step 1 — `schemas/api/ask.py`

Define the request/response models for the RAG and agentic RAG endpoints. `AgenticAskResponse` extends `AskResponse` with rich source objects, reasoning steps, retrieval metadata, and guardrail filters.

Create [`ask.py`](Agentic-RAG-project-agentops/src/schemas/api/ask.py):

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

**Explanation:** `AskRequest` is the shared input for both standard and agentic endpoints. `AskResponse` is the base output with string `sources` (PDF URLs). `AgenticAskResponse` overrides `sources` to `List[Any]` (rich source objects) and adds `reasoning_steps`, `retrieval_attempts`, `rewritten_query`, `trace_id`, `guardrail_filter`, and `output_guardrail_filter`. `FeedbackRequest`/`FeedbackResponse` support the feedback loop.

---

### Step 2 — `schemas/api/health.py`

Define the health check response models.

Create [`health.py`](Agentic-RAG-project-agentops/src/schemas/api/health.py):

```python
from typing import Dict, Optional

from pydantic import BaseModel, Field


class ServiceStatus(BaseModel):
    """Individual service status."""

    status: str = Field(..., description="Service status", example="healthy")
    message: Optional[str] = Field(None, description="Status message", example="Connected successfully")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Overall health status", example="ok")
    version: str = Field(..., description="Application version", example="0.1.0")
    environment: str = Field(..., description="Deployment environment", example="development")
    service_name: str = Field(..., description="Service identifier", example="rag-api")
    services: Optional[Dict[str, ServiceStatus]] = Field(None, description="Individual service statuses")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "status": "ok",
                "version": "0.1.0",
                "environment": "development",
                "service_name": "rag-api",
                "services": {
                    "database": {"status": "healthy", "message": "Connected successfully"},
                    "pdf_parser": {"status": "healthy", "message": "Docling parser ready"},
                },
            }
        }
```

**Explanation:** `ServiceStatus` captures the status and optional message of a single dependency. `HealthResponse` aggregates an overall status, app version/environment/service name, and a dict of per-service statuses.

---

### Step 3 — `schemas/api/search.py`

Define the search request/response models used by the hybrid search endpoint.

Create [`search.py`](Agentic-RAG-project-agentops/src/schemas/api/search.py):

```python
from typing import List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(..., min_length=1, max_length=500, description="Search query across title, abstract, and authors")
    size: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    from_: int = Field(default=0, ge=0, alias="from", description="Offset for pagination")
    categories: Optional[List[str]] = Field(default=None, description="Filter by categories")
    latest_papers: bool = Field(default=False, description="Sort by publication date (newest first) instead of relevance")


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search supporting all search modes."""

    query: str = Field(..., description="Search query text", min_length=1, max_length=500)
    size: int = Field(10, description="Number of results to return", ge=1, le=100)
    from_: int = Field(0, description="Offset for pagination", ge=0, alias="from")
    categories: Optional[List[str]] = Field(None, description="Filter by arXiv categories (e.g., ['cs.AI', 'cs.LG'])")
    latest_papers: bool = Field(False, description="Sort by publication date instead of relevance")
    use_hybrid: bool = Field(True, description="Enable hybrid search (BM25 + vector) with automatic embedding generation")
    min_score: float = Field(0.0, description="Minimum score threshold for results", ge=0.0)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "query": "machine learning neural networks",
                "size": 10,
                "categories": ["cs.AI", "cs.LG"],
                "latest_papers": False,
                "use_hybrid": True,
            }
        }


class SearchHit(BaseModel):
    """Individual search result."""

    arxiv_id: str
    title: str
    authors: Optional[str]
    abstract: Optional[str]
    published_date: Optional[str]
    pdf_url: Optional[str]
    score: float
    highlights: Optional[dict] = None

    # Chunk-specific fields (for unified search)
    chunk_text: Optional[str] = Field(None, description="Text content of the matching chunk")
    chunk_id: Optional[str] = Field(None, description="Unique identifier of the chunk")
    section_name: Optional[str] = Field(None, description="Section name where the chunk was found")


class SearchResponse(BaseModel):
    """Search response model."""

    query: str
    total: int
    hits: List[SearchHit]
    size: int = Field(description="Number of results requested")
    from_: int = Field(alias="from", description="Offset used for pagination")
    search_mode: Optional[str] = Field(None, description="Search mode used: bm25, vector, or hybrid")
    error: Optional[str] = None

    class Config:
        populate_by_name = True
```

**Explanation:** `HybridSearchRequest` supports all search modes with `use_hybrid`, `min_score`, pagination (`from_` with alias `from`), category filters, and latest-first sorting. `SearchHit` models a single result including chunk-level fields. `SearchResponse` wraps the hits with total count and search mode.

---

### Step 4 — `services/a2a/models.py`

Define the Agent-to-Agent (A2A) protocol models used by the A2A router.

Create [`models.py`](Agentic-RAG-project-agentops/src/services/a2a/models.py):

```python
from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class Part(BaseModel):
    type: Literal["text"] = "text"
    text: str


class Message(BaseModel):
    role: Literal["user", "agent"]
    parts: list[Part]


class AgentSkill(BaseModel):
    id: str
    name: str
    description: str


class AgentCapabilities(BaseModel):
    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = False


class AgentCard(BaseModel):
    name: str
    description: str
    url: str
    version: str
    capabilities: AgentCapabilities
    skills: list[AgentSkill]
    defaultInputModes: list[str] = ["text"]
    defaultOutputModes: list[str] = ["text"]


class TaskSendParams(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    message: Message


class TaskStatus(BaseModel):
    state: Literal["submitted", "working", "completed", "failed"]


class Artifact(BaseModel):
    index: int = 0
    parts: list[Part]


class Task(BaseModel):
    id: str
    status: TaskStatus
    artifacts: list[Artifact] = []
```

**Explanation:** These models implement the A2A protocol surface: `Part`/`Message` for the payload, `AgentSkill`/`AgentCapabilities`/`AgentCard` for agent discovery, and `TaskSendParams`/`TaskStatus`/`Artifact`/`Task` for task exchange. `TaskSendParams.id` defaults to a generated UUID.

---

### Step 5 — `dependencies.py`

Create the dependency provider with `Annotated`/`Depends` aliases and the `AgenticRAGDep` provider. Most providers read from `request.app.state`, which is populated in `main.py`.

Create [`dependencies.py`](Agentic-RAG-project-agentops/src/dependencies.py):

```python
from functools import lru_cache
from typing import TYPE_CHECKING, Annotated, Generator, Optional

if TYPE_CHECKING:
    from fastapi import Depends, Request
    from sqlalchemy.orm import Session
else:
    try:
        from fastapi import Depends, Request
        from sqlalchemy.orm import Session
    except ImportError:
        pass

from src.config import Settings
from src.db.interfaces.base import BaseDatabase
from src.services.agents.agentic_rag import AgenticRAGService
from src.services.agents.factory import make_agentic_rag_service
from src.services.arxiv.client import ArxivClient
from src.services.cache.client import CacheClient
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.bedrock_guardrails.service import BedrockGuardrailsService
from src.services.llm_client_protocol import LLMClientProtocol
from src.services.opensearch.client import OpenSearchClient
from src.services.pdf_parser.parser import PDFParserService
from src.services.telegram.bot import TelegramBot


@lru_cache
def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


def get_request_settings(request: Request) -> Settings:
    """Get settings from the request state."""
    return request.app.state.settings


def get_database(request: Request) -> BaseDatabase:
    """Get database from the request state."""
    return request.app.state.database


def get_db_session(database: Annotated[BaseDatabase, Depends(get_database)]) -> Generator[Session, None, None]:
    """Get database session dependency."""
    with database.get_session() as session:
        yield session


def get_opensearch_client(request: Request) -> OpenSearchClient:
    """Get OpenSearch client from the request state."""
    return request.app.state.opensearch_client


def get_arxiv_client(request: Request) -> ArxivClient:
    """Get arXiv client from the request state."""
    return request.app.state.arxiv_client


def get_pdf_parser(request: Request) -> PDFParserService:
    """Get PDF parser service from the request state."""
    return request.app.state.pdf_parser


def get_embeddings_service(request: Request) -> JinaEmbeddingsClient:
    """Get embeddings service from the request state."""
    return request.app.state.embeddings_service


def get_llm_client(request: Request) -> LLMClientProtocol:
    """Get LLM client from the request state (OpenAI or Bedrock depending on PROVIDER)."""
    return request.app.state.llm_client


def get_guardrails_service(request: Request) -> BedrockGuardrailsService:
    """Get Bedrock Guardrails service from the request state."""
    return request.app.state.guardrails_service


def get_langfuse_tracer(request: Request) -> LangfuseTracer:
    """Get Langfuse tracer from the request state."""
    return request.app.state.langfuse_tracer


def get_cache_client(request: Request) -> CacheClient | None:
    """Get cache client from the request state."""
    return getattr(request.app.state, "cache_client", None)


def get_telegram_service(request: Request) -> Optional[TelegramBot]:
    """Get Telegram service from the request state."""
    return getattr(request.app.state, "telegram_service", None)


# Dependency annotations
SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
OpenSearchDep = Annotated[OpenSearchClient, Depends(get_opensearch_client)]
ArxivDep = Annotated[ArxivClient, Depends(get_arxiv_client)]
PDFParserDep = Annotated[PDFParserService, Depends(get_pdf_parser)]
EmbeddingsDep = Annotated[JinaEmbeddingsClient, Depends(get_embeddings_service)]
LLMDep = Annotated[LLMClientProtocol, Depends(get_llm_client)]
GuardrailsDep = Annotated[BedrockGuardrailsService, Depends(get_guardrails_service)]
LangfuseDep = Annotated[LangfuseTracer, Depends(get_langfuse_tracer)]
CacheDep = Annotated[CacheClient | None, Depends(get_cache_client)]
TelegramDep = Annotated[Optional[TelegramBot], Depends(get_telegram_service)]


def get_agentic_rag_service(
    opensearch: OpenSearchDep,
    llm: LLMDep,
    embeddings: EmbeddingsDep,
    langfuse: LangfuseDep,
    guardrails: GuardrailsDep,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AgenticRAGService:
    """Get agentic RAG service."""
    return make_agentic_rag_service(
        opensearch_client=opensearch,
        llm_client=llm,
        embeddings_client=embeddings,
        langfuse_tracer=langfuse,
        guardrails_service=guardrails,
    )


AgenticRAGDep = Annotated[AgenticRAGService, Depends(get_agentic_rag_service)]
```

**Explanation:** Each `get_*` provider reads a service from `request.app.state`. The `Annotated` aliases (`SettingsDep`, `DatabaseDep`, `OpenSearchDep`, `EmbeddingsDep`, `LLMDep`, `LangfuseDep`, `GuardrailsDep`, etc.) let routers declare dependencies concisely. `get_agentic_rag_service` composes the individual dependencies into a fresh `AgenticRAGService` via `make_agentic_rag_service`, and `AgenticRAGDep` is the reusable annotation.

---

### Step 6 — `routers/agentic_ask.py`

Create the agentic RAG router with the `/ask-agentic` and `/feedback` endpoints.

Create [`agentic_ask.py`](Agentic-RAG-project-agentops/src/routers/agentic_ask.py):

```python
from fastapi import APIRouter, HTTPException
from src.dependencies import AgenticRAGDep, LangfuseDep
from src.schemas.api.ask import AgenticAskResponse, AskRequest, FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/api/v1", tags=["agentic-rag"])


@router.post("/ask-agentic", response_model=AgenticAskResponse)
async def ask_agentic(
    request: AskRequest,
    agentic_rag: AgenticRAGDep,
) -> AgenticAskResponse:
    """
    Agentic RAG endpoint with intelligent retrieval and query refinement.

    Features:
    - Decides if retrieval is needed
    - Grades document relevance
    - Rewrites queries if needed
    - Provides reasoning transparency

    The agent will automatically:
    1. Determine if the question requires research paper retrieval
    2. If needed, search for relevant papers
    3. Grade retrieved documents for relevance
    4. Rewrite the query if documents aren't relevant
    5. Generate an answer with citations

    Args:
        request: Question and parameters
        agentic_rag: Injected agentic RAG service

    Returns:
        Answer with sources and reasoning steps

    Raises:
        HTTPException: If processing fails
    """
    try:
        result = await agentic_rag.ask(
            query=request.query,
            model=request.model,
        )

        return AgenticAskResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result.get("sources", []),
            chunks_used=request.top_k,
            search_mode="hybrid" if request.use_hybrid else "bm25",
            reasoning_steps=result.get("reasoning_steps", []),
            retrieval_attempts=result.get("retrieval_attempts", 0),
            rewritten_query=result.get("rewritten_query"),
            trace_id=result.get("trace_id"),
            guardrail_filter=result.get("guardrail_filter"),
            output_guardrail_filter=result.get("output_guardrail_filter"),
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    langfuse_tracer: LangfuseDep,
) -> FeedbackResponse:
    """
    Submit user feedback for an agentic RAG response.

    This endpoint allows users to rate the quality of answers and provide
    optional comments. Feedback is tracked in Langfuse for continuous improvement.

    Args:
        request: Feedback data including trace_id, score, and optional comment
        langfuse_tracer: Injected Langfuse tracer service

    Returns:
        FeedbackResponse indicating success or failure

    Raises:
        HTTPException: If feedback submission fails
    """
    try:
        if not langfuse_tracer:
            raise HTTPException(
                status_code=503,
                detail="Langfuse tracing is disabled. Cannot submit feedback."
            )

        success = langfuse_tracer.submit_feedback(
            trace_id=request.trace_id,
            score=request.score,
            comment=request.comment,
        )

        if success:
            # Flush to ensure feedback is sent immediately
            langfuse_tracer.flush()

            return FeedbackResponse(
                success=True,
                message="Feedback recorded successfully"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to submit feedback to Langfuse"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting feedback: {str(e)}"
        )
```

**Explanation:** `ask_agentic` injects `AgenticRAGDep` and calls `agentic_rag.ask(query, model)`, mapping the rich result dict into an `AgenticAskResponse`. `ValueError` (empty query) maps to 422; other errors map to 500. `submit_feedback` injects `LangfuseDep`, requires tracing to be enabled (503 otherwise), and records feedback via `langfuse_tracer.submit_feedback(...)` followed by `flush()`.

---

### Step 7 — `routers/supervisor_ask.py`

Create the supervisor router with the `/ask-supervisor` endpoint. This router uses `request.app.state.supervisor_agent` (app.state-based DI) rather than a `Depends` alias.

Create [`supervisor_ask.py`](Agentic-RAG-project-agentops/src/routers/supervisor_ask.py):

```python
from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from src.schemas.api.ask import AskRequest
from src.services.agents.supervisor_agent import SupervisorAgent, SupervisorResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["supervisor"])


@router.post("/ask-supervisor")
async def ask_supervisor(body: AskRequest, request: Request) -> dict:
    supervisor: SupervisorAgent = request.app.state.supervisor_agent
    logger.info("Supervisor ask: query_len=%d", len(body.query))

    result: SupervisorResult = await supervisor.ask(query=body.query)

    return {
        "query": body.query,
        "answer": result.answer,
        "intent": result.intent,
        "routed_to": result.routed_to,
        "sources": result.sources,
    }
```

**Explanation:** `ask_supervisor` reads the `SupervisorAgent` from `request.app.state.supervisor_agent` (populated in `main.py`), calls `supervisor.ask(query)`, and returns the answer along with the detected `intent` and `routed_to` target. This demonstrates the app.state-based DI pattern as an alternative to `Depends`.

---

### Step 8 — `routers/a2a.py`

Create the A2A router with agent discovery and task execution endpoints. Note that A2A uses OpenAI directly regardless of the global `PROVIDER` setting.

Create [`a2a.py`](Agentic-RAG-project-agentops/src/routers/a2a.py):

```python
from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Request
from src.config import Settings
from src.services.a2a.models import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    Part,
    Task,
    TaskSendParams,
    TaskStatus,
)
from src.services.openai_llm.client import OpenAILLMClient

logger = logging.getLogger(__name__)
router = APIRouter(tags=["a2a"])


@router.get("/.well-known/agent.json", response_model=AgentCard)
async def get_agent_card(request: Request) -> AgentCard:
    base_url = str(request.base_url).rstrip("/")
    return AgentCard(
        name="ArXiv RAG Agent",
        description="Answers questions about CS/AI/ML arXiv papers using hybrid search and LLMs.",
        url=f"{base_url}/a2a/tasks/send",
        version="1.0.0",
        capabilities=AgentCapabilities(),
        skills=[
            AgentSkill(
                id="arxiv-qa",
                name="ArXiv Paper Q&A",
                description="Answer natural language questions about recent CS/AI/ML research papers.",
            )
        ],
    )


@router.post("/a2a/tasks/send", response_model=Task)
async def send_task(params: TaskSendParams, request: Request) -> Task:
    query = " ".join(p.text for p in params.message.parts if p.type == "text")
    logger.info("A2A task received: task_id=%s query_len=%d", params.id, len(query))

    # A2A uses OpenAI directly regardless of the global PROVIDER setting
    settings = Settings()
    openai_client = OpenAILLMClient(settings)

    # Retrieve chunks via OpenSearch (reuses shared infra)
    embeddings_service = request.app.state.embeddings_service
    opensearch_client = request.app.state.opensearch_client

    query_embedding = await embeddings_service.embed_query(query)
    search_results = opensearch_client.search_unified(
        query=query,
        query_embedding=query_embedding,
        size=5,
        use_hybrid=True,
    )

    chunks: List[Dict[str, Any]] = search_results.get("hits", [])
    logger.info("A2A retrieved %d chunks for OpenAI generation", len(chunks))

    result = await openai_client.generate_rag_answer(
        query=query,
        chunks=chunks,
        model=settings.openai_model,
    )

    return Task(
        id=params.id,
        status=TaskStatus(state="completed"),
        artifacts=[Artifact(parts=[Part(text=result["answer"])])],
    )
```

**Explanation:** `get_agent_card` serves the A2A agent discovery document at `/.well-known/agent.json`. `send_task` extracts the text from the incoming message parts, builds an OpenAI client directly, retrieves chunks via the shared OpenSearch/embeddings services from `app.state`, generates an answer, and returns a completed `Task` with the answer as an artifact.

---

### Step 9 — `routers/hybrid_search.py`

Create the hybrid search router supporting all search modes.

Create [`hybrid_search.py`](Agentic-RAG-project-agentops/src/routers/hybrid_search.py):

```python
import logging

from fastapi import APIRouter, HTTPException
from src.dependencies import EmbeddingsDep, OpenSearchDep
from src.schemas.api.search import HybridSearchRequest, SearchHit, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hybrid-search", tags=["hybrid-search"])


@router.post("/", response_model=SearchResponse)
async def hybrid_search(
    request: HybridSearchRequest, opensearch_client: OpenSearchDep, embeddings_service: EmbeddingsDep
) -> SearchResponse:
    """
    Hybrid search endpoint supporting multiple search modes.
    """
    try:
        if not opensearch_client.health_check():
            raise HTTPException(status_code=503, detail="Search service is currently unavailable")

        query_embedding = None
        if request.use_hybrid:
            try:
                query_embedding = await embeddings_service.embed_query(request.query)
                logger.info("Generated query embedding for hybrid search")
            except Exception as e:
                logger.warning(f"Failed to generate embeddings, falling back to BM25: {e}")
                query_embedding = None

        logger.info(f"Hybrid search: '{request.query}' (hybrid: {request.use_hybrid and query_embedding is not None})")

        results = opensearch_client.search_unified(
            query=request.query,
            query_embedding=query_embedding,
            size=request.size,
            from_=request.from_,
            categories=request.categories,
            latest=request.latest_papers,
            use_hybrid=request.use_hybrid,
            min_score=request.min_score,
        )

        hits = []
        for hit in results.get("hits", []):
            hits.append(
                SearchHit(
                    arxiv_id=hit.get("arxiv_id", ""),
                    title=hit.get("title", ""),
                    authors=hit.get("authors"),
                    abstract=hit.get("abstract"),
                    published_date=hit.get("published_date"),
                    pdf_url=hit.get("pdf_url"),
                    score=hit.get("score", 0.0),
                    highlights=hit.get("highlights"),
                    chunk_text=hit.get("chunk_text"),
                    chunk_id=hit.get("chunk_id"),
                    section_name=hit.get("section_name"),
                )
            )

        search_response = SearchResponse(
            query=request.query,
            total=results.get("total", 0),
            hits=hits,
            size=request.size,
            **{"from": request.from_},
            search_mode="hybrid" if (request.use_hybrid and query_embedding) else "bm25",
        )

        logger.info(f"Search completed: {search_response.total} results returned")
        return search_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
```

**Explanation:** `hybrid_search` checks OpenSearch health (503 if down), optionally generates a query embedding (falling back to BM25 if embedding generation fails), then calls `search_unified` with all the request parameters. It maps each hit into a `SearchHit` and returns a `SearchResponse` with the total count and the effective `search_mode` (`hybrid` if embeddings were used, otherwise `bm25`).

---

### Step 10 — `routers/ping.py`

Create the health check router that reports the status of the database, OpenSearch, and OpenAI.

Create [`ping.py`](Agentic-RAG-project-agentops/src/routers/ping.py):

```python
from fastapi import APIRouter
from sqlalchemy import text

from ..dependencies import DatabaseDep, OpenSearchDep, SettingsDep
from ..schemas.api.health import HealthResponse, ServiceStatus
from ..services.openai_llm.client import OpenAILLMClient

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(settings: SettingsDep, database: DatabaseDep, opensearch_client: OpenSearchDep) -> HealthResponse:
    """Comprehensive health check endpoint for monitoring and load balancer probes.

    :returns: Service health status with version and connectivity checks
    :rtype: HealthResponse
    """
    services = {}
    overall_status = "ok"

    def _check_service(name: str, check_func, *args, **kwargs):
        """Helper to standardize service health checks."""
        try:
            if kwargs.get("is_async"):
                return check_func(*args)
            result = check_func(*args)
            services[name] = result
            if result.status != "healthy":
                nonlocal overall_status
                overall_status = "degraded"
        except Exception as e:
            services[name] = ServiceStatus(status="unhealthy", message=str(e))
            overall_status = "degraded"

    # Database check
    def _check_database():
        with database.get_session() as session:
            session.execute(text("SELECT 1"))
        return ServiceStatus(status="healthy", message="Connected successfully")

    # OpenSearch check
    def _check_opensearch():
        if not opensearch_client.health_check():
            return ServiceStatus(status="unhealthy", message="Not responding")
        stats = opensearch_client.get_index_stats()
        return ServiceStatus(
            status="healthy",
            message=f"Index '{stats.get('index_name', 'unknown')}' with {stats.get('document_count', 0)} documents",
        )

    # Run synchronous checks
    _check_service("database", _check_database)
    _check_service("opensearch", _check_opensearch)

    # OpenAI API health check
    try:
        llm_client = OpenAILLMClient(settings)
        openai_health = await llm_client.health_check()
        services["openai"] = ServiceStatus(status=openai_health["status"], message=openai_health["message"])
        if openai_health["status"] != "healthy":
            overall_status = "degraded"
    except Exception as e:
        services["openai"] = ServiceStatus(status="unhealthy", message=str(e))
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        service_name=settings.service_name,
        services=services,
    )
```

**Explanation:** `health_check` uses the `SettingsDep`, `DatabaseDep`, and `OpenSearchDep` annotations. The `_check_service` helper standardizes status collection and flips `overall_status` to `degraded` when any service is unhealthy. It runs a `SELECT 1` against the database, checks OpenSearch health and index stats, and performs an async OpenAI health check. The result is a `HealthResponse` with per-service statuses.

---

### Step 11 — `main.py` wiring

Wire the new routers and `app.state` services into the FastAPI application. This step connects the agentic RAG service, supervisor agent, and all routers.

Update [`main.py`](Agentic-RAG-project-agentops/src/main.py) with the following additions:

**Imports** (add near the top):

```python
from src.routers.supervisor_ask import router as supervisor_router
from src.services.agents.factory import make_agentic_rag_service
from src.services.arxiv.factory import make_arxiv_client
from src.services.bedrock_guardrails.factory import make_bedrock_guardrails_service
from src.services.bedrock_llm.factory import make_bedrock_llm_client
```

**Service initialization** (inside the app startup block, after the existing services are set on `app.state`):

```python
app.state.guardrails_service = make_bedrock_guardrails_service(settings)
guardrail_status = f"guardrail_id={settings.bedrock.guardrail_id}" if settings.bedrock.guardrail_id else "disabled (no guardrail_id)"

app.state.langfuse_tracer = make_langfuse_tracer()
app.state.cache_client = make_cache_client(settings)
logger.info("Services initialized: arXiv API client, PDF parser, OpenSearch, Embeddings, LLM, Guardrails, Langfuse, Cache")

# Create shared agentic RAG service (used by both MCP and Telegram)
agentic_rag_service = make_agentic_rag_service(
    opensearch_client=app.state.opensearch_client,
    llm_client=app.state.llm_client,
    embeddings_client=app.state.embeddings_service,
    langfuse_tracer=app.state.langfuse_tracer,
    guardrails_service=app.state.guardrails_service,
)
app.state.agentic_rag_service = agentic_rag_service

# Supervisor agent — reuses existing agentic_rag_service and context
from src.services.agents.context import Context
from src.services.agents.supervisor_agent import SupervisorAgent

supervisor_context = Context(
    llm_client=app.state.llm_client,
    opensearch_client=app.state.opensearch_client,
    embeddings_client=app.state.embeddings_service,
    langfuse_tracer=app.state.langfuse_tracer,
    guardrails_service=app.state.guardrails_service,
    model_name=(
        settings.bedrock.model_id if settings.provider == "bedrock" else settings.openai_model
    ),
)
app.state.supervisor_agent = SupervisorAgent(
    context=supervisor_context,
    agentic_rag_service=agentic_rag_service,
)
```

**Router includes** (at the bottom of the module):

```python
# Include routers
app.include_router(ping.router, prefix="/api/v1")
app.include_router(hybrid_search.router, prefix="/api/v1")
app.include_router(ask_router, prefix="/api/v1")
app.include_router(stream_router, prefix="/api/v1")
app.include_router(agentic_ask.router)
app.include_router(a2a_router)
app.include_router(supervisor_router)
```

**Explanation:** `main.py` populates `app.state` with the guardrails service, Langfuse tracer, cache client, the shared `agentic_rag_service`, and the `supervisor_agent`. The supervisor's `Context` is built with the same clients and a provider-aware `model_name`. Finally, all routers are included. Note that `agentic_ask.router`, `a2a_router`, and `supervisor_router` already carry their own `/api/v1` prefixes, so they are included without an additional prefix.

---

## 6. Configuration

The API routers require no new environment variables. They consume the services already configured in earlier phases and stored on `app.state`:

| Service | `app.state` key | Used By |
| --- | --- | --- |
| `agentic_rag_service` | `app.state.agentic_rag_service` | `AgenticRAGDep`, MCP, Telegram |
| `supervisor_agent` | `app.state.supervisor_agent` | `supervisor_ask` router |
| `guardrails_service` | `app.state.guardrails_service` | guardrail nodes, `AgenticRAGDep` |
| `langfuse_tracer` | `app.state.langfuse_tracer` | `LangfuseDep`, feedback endpoint |
| `embeddings_service` | `app.state.embeddings_service` | `EmbeddingsDep`, A2A, hybrid search |
| `opensearch_client` | `app.state.opensearch_client` | `OpenSearchDep`, A2A, hybrid search |
| `llm_client` | `app.state.llm_client` | `LLMDep`, `AgenticRAGDep` |

The A2A router constructs its own `OpenAILLMClient` directly (independent of the global `PROVIDER` setting). The supervisor's `Context.model_name` is resolved from `settings.provider` (`bedrock.model_id` for Bedrock, `openai_model` otherwise).

---

## 7. Verification

Verify the API layer works end to end:

1. **Import check** — confirm all routers and dependencies import cleanly:
   ```bash
   python -c "from src.dependencies import AgenticRAGDep, OpenSearchDep, EmbeddingsDep, LangfuseDep; from src.routers import agentic_ask, supervisor_ask, a2a, hybrid_search, ping"
   ```

2. **Unit tests** — run the API router test suites:
   ```bash
   pytest tests/api/routers/test_agentic_ask.py tests/api/routers/test_hybrid_search.py tests/api/routers/test_ping.py -v
   ```

3. **Start the server** and confirm the app boots without errors:
   ```bash
   uvicorn src.main:app --reload
   ```

4. **Health check** — `GET /api/v1/health` should return `status: ok` with database, opensearch, and openai service statuses.

5. **Agentic ask** — `POST /api/v1/ask-agentic` with a JSON body `{"query": "What are transformers?"}` should return an `AgenticAskResponse` with `answer`, `sources`, `reasoning_steps`, `retrieval_attempts`, `trace_id`, and guardrail filters.

6. **Supervisor ask** — `POST /api/v1/ask-supervisor` with a summarization-style query should return `intent: summarize` and `routed_to: SummarizerAgent`; a specific question should return `intent: rag_lookup` and `routed_to: AgenticRAGService`.

7. **A2A** — `GET /.well-known/agent.json` returns the `AgentCard`; `POST /a2a/tasks/send` with a `TaskSendParams` body returns a completed `Task` with an answer artifact.

8. **Hybrid search** — `POST /api/v1/hybrid-search/` with `{"query": "neural networks"}` returns a `SearchResponse` with hits.

---

## 8. Common Pitfalls

- **Router prefix duplication** — `agentic_ask.router`, `supervisor_ask.router`, and `a2a_router` already define `/api/v1` (or no) prefixes. Do **not** add another `/api/v1` prefix when including them in `main.py`, or paths will double up.
- **`app.state` must be populated before requests** — all `get_*` dependency providers read from `request.app.state`. If a service is not set in `main.py`'s startup block, the corresponding endpoint will fail with an `AttributeError`.
- **`AgenticRAGDep` builds a fresh service per request** — `get_agentic_rag_service` calls `make_agentic_rag_service` on each request. This is intentional for the `Depends` pattern; the compiled graph inside is stateless and reused. For a long-lived singleton, use `app.state.agentic_rag_service` instead.
- **A2A uses OpenAI directly** — the A2A router constructs `OpenAILLMClient(settings)` regardless of `PROVIDER`. Ensure an OpenAI API key is configured if you use the A2A endpoint.
- **Feedback requires Langfuse** — `submit_feedback` returns 503 if `langfuse_tracer` is falsy. Ensure Langfuse is enabled before testing feedback.
- **`from_` alias** — `HybridSearchRequest.from_` and `SearchResponse.from_` use the `from` alias with `populate_by_name = True`. Use `"from"` in JSON payloads.
- **Health check OpenAI** — `ping.py` always constructs `OpenAILLMClient` for the OpenAI health check. If OpenAI is unavailable, the overall status degrades but the endpoint still returns 200 with `degraded`.

---

## 9. Definition of Done

- [ ] [`schemas/api/ask.py`](Agentic-RAG-project-agentops/src/schemas/api/ask.py) defines `AskRequest`, `AskResponse`, `AgenticAskResponse`, `FeedbackRequest`, and `FeedbackResponse`.
- [ ] [`schemas/api/health.py`](Agentic-RAG-project-agentops/src/schemas/api/health.py) defines `ServiceStatus` and `HealthResponse`.
- [ ] [`schemas/api/search.py`](Agentic-RAG-project-agentops/src/schemas/api/search.py) defines `SearchRequest`, `HybridSearchRequest`, `SearchHit`, and `SearchResponse`.
- [ ] [`services/a2a/models.py`](Agentic-RAG-project-agentops/src/services/a2a/models.py) defines the A2A protocol models.
- [ ] [`dependencies.py`](Agentic-RAG-project-agentops/src/dependencies.py) defines all `Annotated` aliases and `AgenticRAGDep`.
- [ ] [`routers/agentic_ask.py`](Agentic-RAG-project-agentops/src/routers/agentic_ask.py) exposes `POST /api/v1/ask-agentic` and `POST /api/v1/feedback`.
- [ ] [`routers/supervisor_ask.py`](Agentic-RAG-project-agentops/src/routers/supervisor_ask.py) exposes `POST /api/v1/ask-supervisor`.
- [ ] [`routers/a2a.py`](Agentic-RAG-project-agentops/src/routers/a2a.py) exposes `GET /.well-known/agent.json` and `POST /a2a/tasks/send`.
- [ ] [`routers/hybrid_search.py`](Agentic-RAG-project-agentops/src/routers/hybrid_search.py) exposes `POST /api/v1/hybrid-search/`.
- [ ] [`routers/ping.py`](Agentic-RAG-project-agentops/src/routers/ping.py) exposes `GET /api/v1/health`.
- [ ] [`main.py`](Agentic-RAG-project-agentops/src/main.py) populates `app.state.agentic_rag_service` and `app.state.supervisor_agent`, and includes all routers.
- [ ] `pytest tests/api/routers/` passes.
- [ ] The server boots and all endpoints respond correctly.