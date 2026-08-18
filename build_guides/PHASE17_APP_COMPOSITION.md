# PHASE 17 — App Composition: MCP Server, Gradio UI, Telegram Bot & Observability

> Build guide for composing the full application: the Model Context Protocol (MCP) server with tools/resources, the Gradio chat UI, the Telegram bot, and Logfire observability — all wired together through the `main.py` composition root.

---

## 1. Phase Objective

In PHASE 16 you exposed the agentic RAG service and routers over HTTP. In this phase you will turn the application into a complete, multi-interface product by:

1. Creating the **MCP server** (`mcp_server/server.py`) with a global `MCPContext` and the `FastMCP` instance.
2. Creating the **MCP tools** (`ask.py`, `feedback.py`, `papers.py`, `search.py`) that expose the agentic pipeline and search over MCP.
3. Creating the **MCP resources** (`resources/papers.py`) that expose papers and index stats as MCP resources.
4. Creating the **Gradio UI** (`gradio_app.py`) as a pure HTTP client that streams answers from `/api/v1/stream`.
5. Creating the **Telegram bot** (`services/telegram/bot.py` + `factory.py`) for Q&A via the agentic pipeline.
6. Creating the **Logfire factory** (`services/logfire/factory.py`) for infrastructure observability.
7. Rewriting **`main.py`** as the composition root that wires all services, the MCP sub-app, the Telegram bot (with a single-worker lock), and all routers together.

By the end of this phase the application is a single bootable FastAPI app that serves HTTP, MCP, a Gradio UI, and a Telegram bot.

---

## 2. Prerequisites

- PHASE 16 completed: all routers (`ping`, `hybrid_search`, `ask`, `stream`, `agentic_ask`, `a2a`, `supervisor_ask`) exist and are wired in `main.py`.
- PHASE 15 completed: `AgenticRAGService` and `make_agentic_rag_service` exist in [`src/services/agents/`](Agentic-RAG-project-agentops/src/services/agents/).
- PHASE 13 completed: `Context` dataclass is available for building the supervisor context.
- PHASE 7 completed: `OpenSearchClient` with `search_unified()`, `health_check()`, `get_index_stats()`, and `setup_indices()`.
- PHASE 6 completed: `JinaEmbeddingsClient` with `embed_query()`.
- PHASE 5 completed: `LLMClientProtocol`, `OpenAILLMClient`, and `make_openai_llm_client` / `make_bedrock_llm_client`.
- PHASE 12 completed: `LangfuseTracer` with `submit_feedback()` and `flush()`.
- PHASE 3 completed: `BaseDatabase` interface and `make_database()`.
- PHASE 2 completed: `Settings` with `mcp`, `telegram`, and `logfire` config sections.
- The `Paper` model and `PaperRepository` exist (from earlier phases).

---

## 3. Dependencies to Install

Add the following packages to the project (they are used by the MCP server, Gradio UI, Telegram bot, and Logfire):

```bash
pip install fastmcp gradio python-telegram-bot logfire
```

- `fastmcp` — the FastMCP server framework (provides `FastMCP`, `@mcp.tool()`, `@mcp.resource()`, and `mcp.http_app()`).
- `gradio` — the chat UI.
- `python-telegram-bot` — the Telegram bot framework (provides `Application`, `CommandHandler`, `MessageHandler`, `filters`).
- `logfire` — observability (provides `logfire.configure`, `logfire.instrument_*`, and `logfire.span`).

`httpx`, `pydantic`, `sqlalchemy`, `langfuse`, and `langgraph` are already installed from earlier phases.

---

## 4. Directory Structure to Create

```
src/
├── main.py                          # composition root — wire everything (REWRITE)
├── gradio_app.py                    # Gradio chat UI (NEW)
├── mcp_server/
│   ├── server.py                    # FastMCP instance + MCPContext (NEW)
│   ├── tools/
│   │   ├── ask.py                   # @mcp.tool ask_question (NEW)
│   │   ├── feedback.py              # @mcp.tool submit_feedback, get_index_stats (NEW)
│   │   ├── papers.py                # @mcp.tool get_paper_details, list_recent_papers (NEW)
│   │   └── search.py                # @mcp.tool search_papers (NEW)
│   └── resources/
│       └── papers.py                # @mcp.resource papers://{id}, index://stats (NEW)
└── services/
    ├── logfire/
    │   └── factory.py               # configure_logfire (NEW)
    └── telegram/
        ├── bot.py                   # TelegramBot class (NEW)
        └── factory.py               # make_telegram_service (NEW)
```

---

## 5. Step-by-Step Implementation

### Step 1 — `services/logfire/factory.py`

Create the Logfire configuration factory. This wires stdlib logging and auto-instrumentation for infrastructure observability. It must run **before** any service factories are called so that SQLAlchemy/Redis/httpx are instrumented before their clients are created.

Create [`factory.py`](Agentic-RAG-project-agentops/src/services/logfire/factory.py):

```python
import logging

import logfire

from src.config import Settings

logger = logging.getLogger(__name__)


def configure_logfire(settings: Settings) -> None:
    """Configure Logfire and wire auto-instrumentation. Called once at startup.

    When LOGFIRE__TOKEN is blank, Logfire prints structured output to console
    but sends nothing to the cloud — safe for local dev with no account needed.
    Langfuse owns all LLM semantic traces; Logfire owns infrastructure observability.
    """
    if not settings.logfire.enabled:
        logger.info("Logfire disabled (LOGFIRE__ENABLED=false)")
        return

    logfire.configure(
        token=settings.logfire.token or None,
        service_name=settings.logfire.service_name,
        environment=settings.logfire.environment,
        send_to_logfire=settings.logfire.send_to_logfire,
    )

    # Infrastructure auto-instrumentation.
    # NOTE: intentionally NOT calling logfire.instrument_openai() —
    # Langfuse already owns rich LLM semantic traces (prompts, tokens, cost).
    # logfire.instrument_httpx() still captures raw HTTP to api.openai.com
    # at the transport level (URL, status, latency) which complements Langfuse.
    logfire.instrument_sqlalchemy()   # Neon PostgreSQL queries via SQLAlchemy
    logfire.instrument_redis()        # Upstash Redis cache GET/SET ops
    logfire.instrument_httpx()        # Jina embeddings, arXiv API, all outbound HTTP
    logfire.instrument_pydantic()     # Pydantic model validation events

    logger.info(
        f"Logfire configured — service={settings.logfire.service_name} "
        f"env={settings.logfire.environment} "
        f"send={settings.logfire.send_to_logfire}"
    )
```

**Explanation:** `configure_logfire` is a no-op when Logfire is disabled. When enabled, it configures Logfire with the token/service/environment from settings and calls the four `instrument_*` helpers. It deliberately does **not** call `instrument_openai()` because Langfuse already owns LLM semantic traces; `instrument_httpx()` still captures raw transport-level HTTP to the OpenAI API.

---

### Step 2 — `mcp_server/server.py`

Create the MCP server module. It defines the `FastMCP` instance, the `MCPContext` dataclass that carries all shared services, and the global context accessors. The tool/resource modules are imported at the **bottom** so that `mcp` is defined before their `@mcp.tool()` / `@mcp.resource()` decorators run.

Create [`server.py`](Agentic-RAG-project-agentops/src/mcp_server/server.py):

```python
import logging
from dataclasses import dataclass
from typing import Optional

from fastmcp import FastMCP
from src.db.interfaces.base import BaseDatabase
from src.services.agents.agentic_rag import AgenticRAGService
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.openai_llm.client import OpenAILLMClient
from src.services.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="arxiv-rag",
    instructions=(
        "Search and query arXiv CS/AI/ML papers using a production-grade hybrid RAG system. "
        "Available tools: search_papers (BM25+vector), ask_question (full agentic pipeline), "
        "get_paper_details, list_recent_papers, submit_feedback, get_index_stats."
    ),
)


@dataclass
class MCPContext:
    opensearch_client: OpenSearchClient
    embeddings_client: JinaEmbeddingsClient
    llm_client: OpenAILLMClient
    langfuse_tracer: Optional[LangfuseTracer]
    agentic_rag_service: AgenticRAGService
    database: BaseDatabase


_mcp_context: Optional[MCPContext] = None


def set_mcp_context(ctx: MCPContext) -> None:
    global _mcp_context
    _mcp_context = ctx
    logger.info("MCP context initialized")


def get_mcp_context() -> MCPContext:
    if _mcp_context is None:
        raise RuntimeError("MCP context not initialized — services must be started first")
    return _mcp_context


# Import tool/resource modules to trigger @mcp.tool() / @mcp.resource() registration.
# These imports MUST stay at the bottom — `mcp` must be defined first.
import src.mcp_server.tools.ask  # noqa: E402, F401
import src.mcp_server.tools.feedback  # noqa: E402, F401
import src.mcp_server.tools.papers  # noqa: E402, F401
import src.mcp_server.tools.search  # noqa: E402, F401
import src.mcp_server.resources.papers  # noqa: E402, F401
```

**Explanation:** The `FastMCP` instance is named `arxiv-rag` with instructions describing the available tools. `MCPContext` bundles the OpenSearch client, embeddings client, LLM client, Langfuse tracer, the shared `AgenticRAGService`, and the database. `set_mcp_context`/`get_mcp_context` provide a module-level global so tools can reach services without dependency injection. The bottom imports register all tools and resources.

---

### Step 3 — `mcp_server/tools/ask.py`

Create the `ask_question` MCP tool that runs the full agentic RAG pipeline.

Create [`ask.py`](Agentic-RAG-project-agentops/src/mcp_server/tools/ask.py):

```python
import logging
from typing import Any, Dict, Optional

import logfire
from src.mcp_server.server import get_mcp_context, mcp

logger = logging.getLogger(__name__)


@mcp.tool()
async def ask_question(
    query: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Ask a research question about arXiv papers using the full agentic RAG pipeline.

    The pipeline includes:
    - Guardrail check (CS/AI/ML scope validation)
    - Hybrid document retrieval (BM25 + vector)
    - LLM-based relevance grading
    - Query rewriting if documents are not relevant
    - Answer generation with source attribution

    Args:
        query: Research question about CS, AI, or ML papers
        model: Optional OpenAI model override (default: gpt-4o-mini)

    Returns:
        dict with keys: query, answer, sources, reasoning_steps,
        retrieval_attempts, rewritten_query, execution_time, guardrail_score,
        trace_id (pass to submit_feedback to rate this response)
    """
    with logfire.span("mcp:ask_question", query=query[:120], model=model or "gpt-4o-mini"):
        ctx = get_mcp_context()

        result = await ctx.agentic_rag_service.ask(
            query=query,
            user_id="mcp-client",
            model=model,
        )

        return {
            "query": result.get("query", query),
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "reasoning_steps": result.get("reasoning_steps", []),
            "retrieval_attempts": result.get("retrieval_attempts", 0),
            "rewritten_query": result.get("rewritten_query"),
            "execution_time_seconds": round(result.get("execution_time", 0.0), 2),
            "guardrail_score": result.get("guardrail_score"),
            "trace_id": result.get("trace_id"),
        }
```

**Explanation:** `ask_question` wraps the call in a `logfire.span`, retrieves the global MCP context, and delegates to `ctx.agentic_rag_service.ask(...)` with `user_id="mcp-client"`. It flattens the rich result dict into a clean MCP-friendly response, rounding `execution_time` to 2 decimals and passing through the `trace_id` for later feedback.

---

### Step 4 — `mcp_server/tools/feedback.py`

Create the `submit_feedback` and `get_index_stats` MCP tools.

Create [`feedback.py`](Agentic-RAG-project-agentops/src/mcp_server/tools/feedback.py):

```python
import logging
from typing import Any, Dict, Optional

import logfire
from src.mcp_server.server import get_mcp_context, mcp

logger = logging.getLogger(__name__)


@mcp.tool()
async def submit_feedback(
    trace_id: str,
    score: float,
    comment: Optional[str] = None,
) -> Dict[str, Any]:
    """Submit quality feedback for a previous ask_question response via Langfuse.

    Args:
        trace_id: Trace ID returned from a previous ask_question call (visible in Langfuse)
        score: Feedback score between -1.0 (bad) and 1.0 (good)
        comment: Optional free-text comment about the response quality

    Returns:
        dict with success bool and message
    """
    with logfire.span("mcp:submit_feedback", trace_id=trace_id, score=score):
        ctx = get_mcp_context()

        if ctx.langfuse_tracer is None:
            return {"success": False, "message": "Langfuse tracing is not configured"}

        if not (-1.0 <= score <= 1.0):
            return {"success": False, "message": "Score must be between -1.0 and 1.0"}

        success = ctx.langfuse_tracer.submit_feedback(
            trace_id=trace_id,
            score=score,
            name="mcp-feedback",
            comment=comment,
        )

        return {
            "success": success,
            "message": "Feedback submitted" if success else "Failed to submit feedback — check Langfuse configuration",
        }


@mcp.tool()
async def get_index_stats() -> Dict[str, Any]:
    """Get statistics about the OpenSearch paper index.

    Returns:
        dict with index_name, exists, document_count, size_in_bytes
    """
    with logfire.span("mcp:get_index_stats"):
        ctx = get_mcp_context()
        return ctx.opensearch_client.get_index_stats()
```

**Explanation:** `submit_feedback` validates that Langfuse is configured and that the score is within `[-1.0, 1.0]`, then delegates to `ctx.langfuse_tracer.submit_feedback(...)` with the name `mcp-feedback`. `get_index_stats` returns the OpenSearch index statistics directly from the context's OpenSearch client.

---

### Step 5 — `mcp_server/tools/papers.py`

Create the `get_paper_details` and `list_recent_papers` MCP tools, plus the `_paper_to_dict` helper.

Create [`papers.py`](Agentic-RAG-project-agentops/src/mcp_server/tools/papers.py):

```python
import logging
from typing import Any, Dict, List, Optional

import logfire
from src.mcp_server.server import get_mcp_context, mcp
from src.models.paper import Paper
from src.repositories.paper import PaperRepository

logger = logging.getLogger(__name__)


def _paper_to_dict(paper: Paper) -> Dict[str, Any]:
    return {
        "id": str(paper.id),
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "authors": paper.authors,
        "abstract": paper.abstract,
        "categories": paper.categories,
        "published_date": paper.published_date.isoformat() if paper.published_date else None,
        "pdf_url": paper.pdf_url,
        "pdf_processed": paper.pdf_processed,
        "created_at": paper.created_at.isoformat() if paper.created_at else None,
    }


@mcp.tool()
async def get_paper_details(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """Fetch full metadata for a specific arXiv paper by its ID.

    Args:
        arxiv_id: arXiv paper ID (e.g. "2310.12402" or "2310.12402v1")

    Returns:
        Paper metadata dict or null if not found in the database
    """
    with logfire.span("mcp:get_paper_details", arxiv_id=arxiv_id):
        ctx = get_mcp_context()

        with ctx.database.get_session() as session:
            repo = PaperRepository(session)
            # Try exact ID first, then without version suffix, then with v1 appended
            paper = repo.get_by_arxiv_id(arxiv_id)
            if paper is None and "v" in arxiv_id:
                paper = repo.get_by_arxiv_id(arxiv_id.split("v")[0])
            if paper is None and "v" not in arxiv_id:
                paper = repo.get_by_arxiv_id(f"{arxiv_id}v1")
            if paper is None:
                return None
            return _paper_to_dict(paper)


@mcp.tool()
async def list_recent_papers(
    limit: int = 10,
    offset: int = 0,
    processed_only: bool = False,
) -> List[Dict[str, Any]]:
    """List recently ingested arXiv papers from the database, ordered by publish date descending.

    Args:
        limit: Number of papers to return (default 10, max 50)
        offset: Pagination offset (default 0)
        processed_only: If True, return only papers with parsed PDF content

    Returns:
        List of paper metadata dicts
    """
    with logfire.span("mcp:list_recent_papers", limit=limit, offset=offset, processed_only=processed_only):
        ctx = get_mcp_context()
        limit = min(limit, 50)

        with ctx.database.get_session() as session:
            repo = PaperRepository(session)
            if processed_only:
                papers = repo.get_processed_papers(limit=limit, offset=offset)
            else:
                papers = repo.get_all(limit=limit, offset=offset)
            return [_paper_to_dict(p) for p in papers]
```

**Explanation:** `_paper_to_dict` converts a `Paper` ORM object into a JSON-serializable dict, ISO-formatting dates. `get_paper_details` tries the exact arXiv ID, then strips the version suffix, then appends `v1` — covering the common ID variants. `list_recent_papers` caps `limit` at 50 and optionally filters to papers with parsed PDF content via `get_processed_papers`.

---

### Step 6 — `mcp_server/tools/search.py`

Create the `search_papers` MCP tool with the `ChunkResult` Pydantic model.

Create [`search.py`](Agentic-RAG-project-agentops/src/mcp_server/tools/search.py):

```python
import logging
from typing import Any, Dict, List, Optional

import logfire
from pydantic import BaseModel
from src.mcp_server.server import get_mcp_context, mcp

logger = logging.getLogger(__name__)


class ChunkResult(BaseModel):
    chunk_id: str
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: Optional[str]
    chunk_text: str
    section_title: Optional[str]
    score: float


@mcp.tool()
async def search_papers(
    query: str,
    top_k: int = 5,
    use_hybrid: bool = True,
) -> List[Dict[str, Any]]:
    """Search indexed arXiv CS/AI/ML paper chunks using BM25 + semantic vector search (hybrid RRF).

    Args:
        query: Natural language search query
        top_k: Number of chunks to return (default 5, max 20)
        use_hybrid: Use hybrid BM25+vector search; False = BM25 only (default True)

    Returns:
        List of matching chunks with arxiv_id, title, authors, chunk_text, score
    """
    with logfire.span("mcp:search_papers", query=query[:120], top_k=top_k, use_hybrid=use_hybrid):
        ctx = get_mcp_context()
        top_k = min(top_k, 20)

        query_embedding: Optional[List[float]] = None
        if use_hybrid:
            try:
                query_embedding = await ctx.embeddings_client.embed_query(query)
            except Exception as e:
                logger.warning(f"Embedding failed, falling back to BM25: {e}")

        results = ctx.opensearch_client.search_unified(
            query=query,
            query_embedding=query_embedding,
            size=top_k,
            use_hybrid=use_hybrid and query_embedding is not None,
        )

        hits = results.get("hits", [])
        return [
            {
                "chunk_id": hit.get("chunk_id", ""),
                "arxiv_id": hit.get("arxiv_id", ""),
                "title": hit.get("title", ""),
                "authors": hit.get("authors", []),
                "abstract": hit.get("abstract"),
                "chunk_text": hit.get("chunk_text", ""),
                "section_title": hit.get("section_title"),
                "score": round(float(hit.get("score", 0.0)), 4),
            }
            for hit in hits
        ]
```

**Explanation:** `search_papers` caps `top_k` at 20. When `use_hybrid` is true it attempts to generate a query embedding, falling back to BM25 if embedding generation fails. It then calls `search_unified` with `use_hybrid` only when an embedding is available, and maps each hit into a clean dict with the score rounded to 4 decimals. `ChunkResult` is a Pydantic model that documents the result shape.

---

### Step 7 — `mcp_server/resources/papers.py`

Create the MCP resources for papers and index stats.

Create [`papers.py`](Agentic-RAG-project-agentops/src/mcp_server/resources/papers.py):

```python
import logging
from typing import Any, Dict

from src.mcp_server.server import get_mcp_context, mcp
from src.repositories.paper import PaperRepository

logger = logging.getLogger(__name__)


@mcp.resource("papers://{arxiv_id}")
async def get_paper_resource(arxiv_id: str) -> Dict[str, Any]:
    """Read full metadata and abstract for an arXiv paper.

    URI pattern: papers://{arxiv_id}
    Example:     papers://2310.12402
    """
    ctx = get_mcp_context()

    with ctx.database.get_session() as session:
        repo = PaperRepository(session)
        paper = repo.get_by_arxiv_id(arxiv_id)
        if paper is None and "v" in arxiv_id:
            paper = repo.get_by_arxiv_id(arxiv_id.split("v")[0])
        if paper is None and "v" not in arxiv_id:
            paper = repo.get_by_arxiv_id(f"{arxiv_id}v1")

    if paper is None:
        return {"error": f"Paper {arxiv_id} not found in database"}

    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "authors": paper.authors,
        "abstract": paper.abstract,
        "categories": paper.categories,
        "published_date": paper.published_date.isoformat() if paper.published_date else None,
        "pdf_url": paper.pdf_url,
        "pdf_processed": paper.pdf_processed,
    }


@mcp.resource("index://stats")
async def get_index_stats_resource() -> Dict[str, Any]:
    """Read current OpenSearch index statistics.

    URI: index://stats
    """
    ctx = get_mcp_context()
    return ctx.opensearch_client.get_index_stats()
```

**Explanation:** `get_paper_resource` is registered under the URI pattern `papers://{arxiv_id}` and applies the same ID-variant fallback logic as the tool. It returns an error dict if the paper is not found. `get_index_stats_resource` is registered under `index://stats` and returns the OpenSearch index statistics.

---

### Step 8 — `services/telegram/bot.py`

Create the `TelegramBot` class that answers questions via the agentic RAG pipeline and supports `/start`, `/help`, and `/search` commands.

Create [`bot.py`](Agentic-RAG-project-agentops/src/services/telegram/bot.py):

```python
import logging
from typing import Optional

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot for Q&A via agentic RAG pipeline."""

    def __init__(
        self,
        bot_token: str,
        opensearch_client,
        embeddings_client,
        llm_client,
        cache_client=None,
        agentic_rag_service=None,
    ):
        self.bot_token = bot_token
        self.opensearch = opensearch_client
        self.embeddings = embeddings_client
        self.llm = llm_client
        self.cache = cache_client
        self.agentic_rag_service = agentic_rag_service
        self.application: Optional[Application] = None

    async def start(self) -> None:
        """Start bot with polling."""
        logger.info("Starting Telegram bot...")
        self.application = Application.builder().token(self.bot_token).build()

        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("help", self._help_command))
        self.application.add_handler(CommandHandler("search", self._search_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_question))

        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        logger.info("Telegram bot started successfully")

    async def stop(self) -> None:
        """Stop bot."""
        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            logger.info("Telegram bot stopped")

    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start."""
        await update.message.reply_text(
            "Welcome to arXiv Paper Curator!\n\n"
            "Ask me questions about CS papers and I'll provide answers with sources.\n\n"
            "Commands:\n"
            "/help - Show this help\n"
            "/search <keywords> - Search papers"
        )

    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help."""
        await update.message.reply_text(
            "Send me any question about computer science research papers.\n\n"
            "Examples:\n"
            "- What are transformer architectures?\n"
            "- How does BERT work?\n"
            "- Explain attention mechanisms\n\n"
            "Use /search to find specific papers."
        )

    async def _search_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /search."""
        if not context.args:
            await update.message.reply_text("Usage: /search <keywords>\nExample: /search neural networks")
            return

        query = " ".join(context.args)
        await update.message.chat.send_action("typing")

        try:
            query_embedding = await self.embeddings.embed_query(query)
            results = self.opensearch.search_unified(
                query=query,
                query_embedding=query_embedding,
                size=10,
                use_hybrid=True,
            )

            hits = results.get("hits", [])
            if not hits:
                await update.message.reply_text("No papers found. Try different keywords.")
                return

            seen_ids: set = set()
            unique_papers = []
            for hit in hits:
                arxiv_id = hit.get("arxiv_id", "")
                if arxiv_id and arxiv_id not in seen_ids:
                    seen_ids.add(arxiv_id)
                    unique_papers.append(hit)
                if len(unique_papers) >= 5:
                    break

            response = f"Found {len(unique_papers)} papers:\n\n"
            for idx, hit in enumerate(unique_papers, 1):
                title = hit.get("title", "Untitled")
                arxiv_id = hit.get("arxiv_id", "")
                url = f"https://arxiv.org/abs/{arxiv_id}"
                response += f"{idx}. {title}\n{url}\n\n"

            await update.message.reply_text(response, disable_web_page_preview=True)

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            await update.message.reply_text(f"Search failed: {str(e)}")

    async def _handle_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle user questions via agentic RAG pipeline."""
        query = update.message.text
        user_id = str(update.effective_user.id) if update.effective_user else "telegram_user"
        await update.message.chat.send_action("typing")

        try:
            if self.agentic_rag_service is None:
                await update.message.reply_text("RAG service unavailable.")
                return

            result = await self.agentic_rag_service.ask(query=query, user_id=user_id)
            await self._send_agentic_answer(update, result)

        except Exception as e:
            logger.error(f"Question handling failed: {e}", exc_info=True)
            await update.message.reply_text(f"Error: {str(e)}")

    async def _send_agentic_answer(self, update: Update, result: dict) -> None:
        """Format and send agentic RAG response."""
        answer = result.get("answer", "No answer generated.")
        sources = result.get("sources", [])

        message = f"*Answer:*\n{answer}\n"

        if sources:
            message += "\n*Sources:*\n"
            for idx, source in enumerate(sources[:5], 1):
                if isinstance(source, dict):
                    arxiv_id = source.get("arxiv_id", "")
                    title = source.get("title", "")
                    label = title if title else arxiv_id
                    message += f"{idx}. [{label}](https://arxiv.org/abs/{arxiv_id})\n"
                else:
                    message += f"{idx}. {source}\n"

        rewritten = result.get("rewritten_query")
        if rewritten and rewritten != result.get("query", ""):
            message += f"\n_Query refined to: {rewritten}_"

        try:
            await update.message.reply_text(message, parse_mode="Markdown", disable_web_page_preview=True)
        except Exception:
            await update.message.reply_text(answer, disable_web_page_preview=True)
```

**Explanation:** `TelegramBot.start()` builds a `telegram.ext.Application`, registers the `/start`, `/help`, `/search` command handlers and a text-message handler, then initializes, starts, and begins polling. `/search` performs a hybrid search and deduplicates hits by `arxiv_id` (max 5 unique papers). `_handle_question` routes free-form text through `self.agentic_rag_service.ask(...)`. `_send_agentic_answer` formats the answer with Markdown sources and the rewritten query, falling back to plain text if Markdown parsing fails.

---

### Step 9 — `services/telegram/factory.py`

Create the Telegram service factory that returns `None` when the bot is disabled or no token is configured.

Create [`factory.py`](Agentic-RAG-project-agentops/src/services/telegram/factory.py):

```python
import logging
from typing import Optional

from src.config import get_settings
from src.services.telegram.bot import TelegramBot

logger = logging.getLogger(__name__)


def make_telegram_service(
    opensearch_client,
    embeddings_client,
    llm_client,
    cache_client=None,
    langfuse_tracer=None,
    agentic_rag_service=None,
) -> Optional[TelegramBot]:
    """Create Telegram bot if enabled and token is configured."""
    settings = get_settings()

    if not settings.telegram.enabled:
        logger.info("Telegram bot is disabled")
        return None

    if not settings.telegram.bot_token:
        logger.warning("Telegram bot token not configured")
        return None

    bot = TelegramBot(
        bot_token=settings.telegram.bot_token,
        opensearch_client=opensearch_client,
        embeddings_client=embeddings_client,
        llm_client=llm_client,
        cache_client=cache_client,
        agentic_rag_service=agentic_rag_service,
    )

    logger.info("Telegram bot created successfully")
    return bot
```

**Explanation:** `make_telegram_service` reads settings and returns `None` if the bot is disabled or the token is missing. Otherwise it constructs a `TelegramBot` with the shared clients and the agentic RAG service.

---

### Step 10 — `gradio_app.py`

Create the Gradio chat UI. It is a **pure HTTP client** — it does not import any service directly; instead it streams answers from the running API at `http://localhost:8000/api/v1/stream`.

Create [`gradio_app.py`](Agentic-RAG-project-agentops/src/gradio_app.py):

```python
import json
import logging
from typing import Iterator

import gradio as gr
import httpx

logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
DEFAULT_MODEL = "gpt-4o-mini"
AVAILABLE_CATEGORIES = ["cs.AI", "cs.LG"]


async def stream_response(
    query: str, top_k: int = 3, use_hybrid: bool = True, model: str = DEFAULT_MODEL, categories: str = ""
) -> Iterator[str]:
    """Stream response from the RAG API"""
    if not query.strip():
        yield "Please enter a question."
        return

    # Parse categories
    category_list = [cat.strip() for cat in categories.split(",") if cat.strip()] if categories else None

    # Prepare request payload
    payload = {"query": query, "top_k": top_k, "use_hybrid": use_hybrid, "model": model, "categories": category_list}

    try:
        url = f"{API_BASE_URL}/stream"
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload, headers={"Accept": "text/plain"}) as response:
                if response.status_code != 200:
                    yield f"Error: API returned status {response.status_code}"
                    return

                current_answer = ""
                sources = []
                chunks_used = 0
                search_mode = ""

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        try:
                            data = json.loads(data_str)

                            # Handle error
                            if "error" in data:
                                yield f"Error: {data['error']}"
                                return

                            # Handle metadata
                            if "sources" in data:
                                sources = data["sources"]
                                chunks_used = data.get("chunks_used", 0)
                                search_mode = data.get("search_mode", "unknown")
                                continue

                            # Handle streaming chunks
                            if "chunk" in data:
                                current_answer += data["chunk"]
                                # Format response with sources if we have them
                                formatted_response = current_answer
                                if sources or chunks_used:
                                    formatted_response += f"\n\n**Search Info:**\n"
                                    formatted_response += f"- Mode: {search_mode}\n"
                                    formatted_response += f"- Chunks used: {chunks_used}\n"
                                    if sources:
                                        formatted_response += f"- Sources: {len(sources)} papers\n"
                                        for i, source in enumerate(sources[:3], 1):  # Show first 3 sources
                                            formatted_response += f"  {i}. [{source.split('/')[-1]}]({source})\n"
                                        if len(sources) > 3:
                                            formatted_response += f"  ... and {len(sources) - 3} more\n"

                                yield formatted_response

                            # Handle completion
                            if data.get("done", False):
                                final_answer = data.get("answer", current_answer)
                                if final_answer != current_answer:
                                    current_answer = final_answer

                                # Final formatted response
                                formatted_response = current_answer
                                if sources or chunks_used:
                                    formatted_response += f"\n\n**Search Info:**\n"
                                    formatted_response += f"- Mode: {search_mode}\n"
                                    formatted_response += f"- Chunks used: {chunks_used}\n"
                                    if sources:
                                        formatted_response += f"- Sources: {len(sources)} papers\n"
                                        for i, source in enumerate(sources[:3], 1):
                                            formatted_response += f"  {i}. [{source.split('/')[-1]}]({source})\n"
                                        if len(sources) > 3:
                                            formatted_response += f"  ... and {len(sources) - 3} more\n"

                                yield formatted_response
                                break

                        except json.JSONDecodeError:
                            continue  # Skip malformed JSON lines

    except httpx.RequestError as e:
        yield f"Connection error: {str(e)}\nMake sure the API server is running at {API_BASE_URL}"
    except Exception as e:
        yield f"Unexpected error: {str(e)}"


def create_gradio_interface():
    """Create and configure the Gradio interface"""

    with gr.Blocks(
        title="arXiv Paper Curator - RAG Chat",
        theme=gr.themes.Soft(),
    ) as interface:
        gr.Markdown(
            """
            # 🔬 arXiv Paper Curator - RAG Chat
            
            Ask questions about machine learning and AI research papers from arXiv.
            The system will search through indexed papers and provide answers with sources.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Your Question", placeholder="What are transformers in machine learning?", lines=2, max_lines=5
                )

            with gr.Column(scale=1):
                submit_btn = gr.Button("Ask Question", variant="primary", size="lg")

        with gr.Row():
            with gr.Column():
                with gr.Accordion("Advanced Options", open=False):
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Number of chunks to retrieve",
                        info="More chunks = more context but slower generation",
                    )

                    use_hybrid = gr.Checkbox(
                        value=True,
                        label="Use hybrid search (BM25 + vector embeddings)",
                        info="Usually better results than keyword-only search",
                    )

                    model_choice = gr.Dropdown(
                        choices=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
                        value=DEFAULT_MODEL,
                        label="OpenAI Model",
                        info="gpt-4o-mini is fastest; gpt-4o gives richer answers",
                    )

                    categories = gr.Textbox(
                        label="arXiv Categories (optional)",
                        placeholder="cs.AI, cs.LG, cs.CL",
                        info="Comma-separated. Leave empty for all categories",
                    )

        response_output = gr.Markdown(
            label="Answer", value="Ask a question to get started!", height=400, elem_classes=["response-markdown"]
        )

        # Examples
        gr.Examples(
            examples=[
                ["What are transformers in machine learning?", 3, True, "gpt-4o-mini", "cs.AI, cs.LG"],
                ["How do convolutional neural networks work?", 5, True, "gpt-4o-mini", "cs.CV, cs.LG"],
                ["What is attention mechanism in deep learning?", 4, False, "gpt-4o-mini", "cs.AI"],
                ["Explain reinforcement learning algorithms", 3, True, "gpt-4o-mini", "cs.LG, cs.AI"],
                ["What are the latest developments in NLP?", 5, True, "gpt-4o-mini", "cs.CL"],
            ],
            inputs=[query_input, top_k, use_hybrid, model_choice, categories],
        )

        # Handle submission
        submit_btn.click(
            fn=stream_response,
            inputs=[query_input, top_k, use_hybrid, model_choice, categories],
            outputs=[response_output],
            show_progress=True,
        )

        # Handle Enter key
        query_input.submit(
            fn=stream_response,
            inputs=[query_input, top_k, use_hybrid, model_choice, categories],
            outputs=[response_output],
            show_progress=True,
        )

        gr.Markdown(
            """
            ---
            
            **Note**: Make sure the RAG API server is running at `http://localhost:8000` before using this interface.
            
            **Categories**: cs.AI (Artificial Intelligence), cs.LG (Machine Learning), cs.CL (Computational Linguistics),
            cs.CV (Computer Vision), cs.NE (Neural Networks), stat.ML (Statistics - Machine Learning)
            """
        )

    return interface


def main():
    """Main entry point for the Gradio app"""
    print("🚀 Starting arXiv Paper Curator Gradio Interface...")
    print(f"📡 API Base URL: {API_BASE_URL}")

    interface = create_gradio_interface()

    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Changed to avoid port conflict
        share=False,
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    main()
```

**Explanation:** `stream_response` is an async generator that POSTs to `/api/v1/stream` and iterates over the SSE lines. It accumulates the answer from `chunk` events, captures `sources`/`chunks_used`/`search_mode` from metadata events, and yields a formatted Markdown response. `create_gradio_interface` builds the `gr.Blocks` UI with a query box, advanced options (top_k, hybrid toggle, model dropdown, categories), a Markdown output, examples, and click/submit handlers. `main()` launches the interface on port `7861`.

---

### Step 11 — `main.py` composition root

Rewrite `main.py` as the composition root. It creates the MCP HTTP sub-app at module level, configures Logfire first, initializes every service into `app.state`, builds the shared `agentic_rag_service` and `SupervisorAgent`, wires the MCP context, starts the Telegram bot (guarded by a single-worker `fcntl.flock` lock), mounts the MCP sub-app, and includes all routers.

Create [`main.py`](Agentic-RAG-project-agentops/src/main.py):

```python
import fcntl
import logging
import os
from contextlib import asynccontextmanager

import logfire
import uvicorn
from fastapi import FastAPI
from src.config import get_settings
from src.db.factory import make_database
from src.mcp_server.server import MCPContext, mcp, set_mcp_context
from src.routers import agentic_ask, hybrid_search, ping
from src.routers.a2a import router as a2a_router
from src.routers.ask import ask_router, stream_router
from src.routers.supervisor_ask import router as supervisor_router
from src.services.agents.factory import make_agentic_rag_service
from src.services.arxiv.factory import make_arxiv_client
from src.services.bedrock_guardrails.factory import make_bedrock_guardrails_service
from src.services.bedrock_llm.factory import make_bedrock_llm_client
from src.services.cache.factory import make_cache_client
from src.services.embeddings.factory import make_embeddings_service
from src.services.langfuse.factory import make_langfuse_tracer
from src.services.logfire.factory import configure_logfire
from src.services.openai_llm.factory import make_openai_llm_client
from src.services.opensearch.factory import make_opensearch_client
from src.services.pdf_parser.factory import make_pdf_parser_service
from src.services.telegram.factory import make_telegram_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create MCP HTTP app once at module level.
# path="/" places the route at "/" inside the sub-app so it matches when mounted at /mcp.
_mcp_http_app = mcp.http_app(path="/", stateless_http=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan for the API."""
    # MCP session manager must be started before any requests arrive.
    # All service init happens inside this context so the manager is alive
    # for the full duration the app is running.
    async with _mcp_http_app.lifespan(app):
        logger.info("Starting RAG API...")

        settings = get_settings()
        app.state.settings = settings

        # Configure Logfire first — wires stdlib logging bridge + auto-instrumentation.
        # Must run before any service factories so SQLAlchemy/Redis/httpx are instrumented
        # before their clients are created.
        configure_logfire(settings)
        if settings.logfire.enabled:
            logfire.instrument_fastapi(app, request_attributes_mapper=_skip_health)

        database = make_database()
        app.state.database = database
        logger.info("Database connected")

        # Initialize search service
        opensearch_client = make_opensearch_client()
        app.state.opensearch_client = opensearch_client

        if opensearch_client.health_check():
            logger.info("OpenSearch connected successfully")

            setup_results = opensearch_client.setup_indices(force=False)
            if setup_results.get("hybrid_index"):
                logger.info("Hybrid index created")
            else:
                logger.info("Hybrid index already exists")

            try:
                stats = opensearch_client.client.count(index=opensearch_client.index_name)
                logger.info(f"OpenSearch ready: {stats['count']} documents indexed")
            except Exception:
                logger.info("OpenSearch index ready (stats unavailable)")
        else:
            logger.warning("OpenSearch connection failed - search features will be limited")

        # Initialize other services
        app.state.arxiv_client = make_arxiv_client()
        app.state.pdf_parser = make_pdf_parser_service()
        app.state.embeddings_service = make_embeddings_service()
        if settings.provider == "bedrock":
            app.state.llm_client = make_bedrock_llm_client(settings)
            logger.info(f"LLM provider: AWS Bedrock (model={settings.bedrock.model_id})")
        else:
            app.state.llm_client = make_openai_llm_client()
            logger.info(f"LLM provider: OpenAI (model={settings.openai_model})")

        app.state.guardrails_service = make_bedrock_guardrails_service(settings)
        guardrail_status = f"guardrail_id={settings.bedrock.guardrail_id}" if settings.bedrock.guardrail_id else "disabled (no guardrail_id)"
        logger.info(f"Bedrock Guardrails: {guardrail_status}")

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
        logger.info("SupervisorAgent initialized")

        # Wire MCP context so tools can reach all services
        if settings.mcp.enabled:
            set_mcp_context(
                MCPContext(
                    opensearch_client=app.state.opensearch_client,
                    embeddings_client=app.state.embeddings_service,
                    llm_client=app.state.llm_client,
                    langfuse_tracer=app.state.langfuse_tracer,
                    agentic_rag_service=agentic_rag_service,
                    database=app.state.database,
                )
            )
            logger.info(f"MCP server context (mounted at {settings.mcp.path})")

        # Initialize Telegram bot (Phase 7)
        # Only one worker process may run the polling bot; others skip.
        # fcntl.flock gives an exclusive non-blocking lock; kernel releases it when the
        # process exits, so a container restart cleanly re-acquires it.
        _telegram_lock_fd = None
        _telegram_lock_acquired = False
        try:
            _telegram_lock_fd = open("/tmp/telegram_bot.lock", "w")
            fcntl.flock(_telegram_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _telegram_lock_acquired = True
        except IOError:
            logger.info("Telegram bot lock held by another worker — skipping in this worker")

        if _telegram_lock_acquired:
            telegram_service = make_telegram_service(
                opensearch_client=app.state.opensearch_client,
                embeddings_client=app.state.embeddings_service,
                llm_client=app.state.llm_client,
                cache_client=app.state.cache_client,
                langfuse_tracer=app.state.langfuse_tracer,
                agentic_rag_service=agentic_rag_service,
            )

            if telegram_service:
                app.state.telegram_service = telegram_service
                try:
                    await telegram_service.start()
                    logger.info("Telegram bot started successfully")
                except Exception as e:
                    logger.error(f"Failed to start Telegram bot: {e}")
            else:
                logger.info("Telegram bot not configured - skipping initialization")

        logger.info("API ready")
        yield

        # Cleanup
        if hasattr(app.state, "telegram_service") and app.state.telegram_service:
            await app.state.telegram_service.stop()
            logger.info("Telegram bot stopped")

        if _telegram_lock_fd:
            fcntl.flock(_telegram_lock_fd, fcntl.LOCK_UN)
            _telegram_lock_fd.close()

        database.teardown()
        logger.info("API shutdown complete")


app = FastAPI(
    title="arXiv Paper Curator API",
    description="Personal arXiv CS.AI paper curator with RAG capabilities",
    version=os.getenv("APP_VERSION", "0.1.0"),
    lifespan=lifespan,
)

def _skip_health(request, attributes):
    return {} if request.url.path == "/api/v1/health" else attributes

# Include routers
app.include_router(ping.router, prefix="/api/v1")
app.include_router(hybrid_search.router, prefix="/api/v1")
app.include_router(ask_router, prefix="/api/v1")
app.include_router(stream_router, prefix="/api/v1")
app.include_router(agentic_ask.router)
app.include_router(a2a_router)
app.include_router(supervisor_router)

# Mount MCP sub-app (lifespan is composed inside the main lifespan above)
_mcp_settings = get_settings().mcp
if _mcp_settings.enabled:
    app.mount(_mcp_settings.path, _mcp_http_app)


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
```

**Explanation:** `main.py` is the composition root. The MCP HTTP sub-app is created once at module level with `path="/"` so it matches when mounted at `/mcp`. The `lifespan` context manager wraps everything inside `_mcp_http_app.lifespan(app)` so the MCP session manager is alive for the app's full lifetime. Logfire is configured first, then every service is created and stored on `app.state`. The shared `agentic_rag_service` and `SupervisorAgent` are built, the MCP context is wired via `set_mcp_context`, and the Telegram bot is started only by the worker that acquires the `fcntl.flock` lock on `/tmp/telegram_bot.lock`. Finally all routers are included and the MCP sub-app is mounted at `settings.mcp.path`.

---

## 6. Configuration

The composition root consumes the existing `Settings` sections. The relevant configuration groups are:

| Setting group | Key settings | Purpose |
| --- | --- | --- |
| `logfire` | `enabled`, `token`, `service_name`, `environment`, `send_to_logfire` | Controls Logfire observability and auto-instrumentation |
| `mcp` | `enabled`, `path` | Controls whether the MCP sub-app is mounted and at which path (default `/mcp`) |
| `telegram` | `enabled`, `bot_token` | Controls whether the Telegram bot starts and with which token |
| `provider` | `provider`, `openai_model`, `bedrock.model_id` | Selects the LLM provider and default model |

Example `.env` additions:

```bash
# Logfire
LOGFIRE__ENABLED=true
LOGFIRE__TOKEN=
LOGFIRE__SERVICE_NAME=arxiv-rag-api
LOGFIRE__ENVIRONMENT=development
LOGFIRE__SEND_TO_LOGFIRE=false

# MCP
MCP__ENABLED=true
MCP__PATH=/mcp

# Telegram
TELEGRAM__ENABLED=false
TELEGRAM__BOT_TOKEN=
```

The Gradio UI is configured via module-level constants in `gradio_app.py` (`API_BASE_URL`, `DEFAULT_MODEL`, `AVAILABLE_CATEGORIES`) and launched on port `7861`.

---

## 7. Verification

Verify the composed application works end to end:

1. **Import check** — confirm all new modules import cleanly:
   ```bash
   python -c "from src.mcp_server.server import mcp, MCPContext, set_mcp_context, get_mcp_context; from src.gradio_app import create_gradio_interface; from src.services.telegram.bot import TelegramBot; from src.services.logfire.factory import configure_logfire"
   ```

2. **MCP tool registration** — confirm the tools and resources are registered:
   ```bash
   python -c "from src.mcp_server.server import mcp; print([t.name for t in mcp._tool_manager.list_tools()]); print([r.uri for r in mcp._resource_manager.list_resources()])"
   ```
   Expected tools: `ask_question`, `submit_feedback`, `get_index_stats`, `get_paper_details`, `list_recent_papers`, `search_papers`. Expected resources: `papers://{arxiv_id}`, `index://stats`.

3. **Start the server** and confirm the app boots without errors:
   ```bash
   uvicorn src.main:app --reload
   ```
   The logs should show Logfire configuration, database connection, OpenSearch connection, all services initialized, `SupervisorAgent initialized`, MCP context, and `API ready`.

4. **Health check** — `GET /api/v1/health` should return `status: ok`.

5. **MCP endpoint** — with `MCP__ENABLED=true`, the MCP sub-app should be reachable at `http://localhost:8000/mcp` (e.g. `GET /mcp` returns the MCP server info).

6. **Gradio UI** — launch the UI and confirm it streams answers:
   ```bash
   python -m src.gradio_app
   ```
   Open `http://localhost:7861`, ask a question, and confirm the answer streams with search info and sources.

7. **Telegram bot** — with `TELEGRAM__ENABLED=true` and a valid `TELEGRAM__BOT_TOKEN`, confirm the bot starts (log: `Telegram bot started successfully`) and responds to `/start`, `/help`, `/search`, and free-text questions.

---

## 8. Common Pitfalls

- **MCP tool/resource imports must stay at the bottom of `server.py`** — the `@mcp.tool()` / `@mcp.resource()` decorators need `mcp` to be defined first. If you import the tool modules at the top, you'll get a `NameError` or circular import.
- **`get_mcp_context()` raises before startup** — if a tool is called before `set_mcp_context` runs (i.e. before the lifespan startup), it raises `RuntimeError`. Ensure MCP is only used after the app has started.
- **Logfire must be configured before service factories** — `configure_logfire` must run before `make_database`, `make_cache_client`, etc., so the SQLAlchemy/Redis/httpx clients are instrumented at creation time.
- **MCP sub-app path** — `mcp.http_app(path="/", stateless_http=True)` places the route at `/` inside the sub-app so it matches when mounted at `/mcp`. Do not change `path` to `/mcp` inside the sub-app or the mounted path will double up.
- **Telegram bot single-worker lock** — only one worker may run the polling bot. The `fcntl.flock` lock on `/tmp/telegram_bot.lock` ensures only one worker starts it. On Windows, `fcntl` is unavailable — this code is intended for Linux/container environments.
- **Gradio is a separate process** — `gradio_app.py` is a pure HTTP client; it does not import services. The API server must be running at `http://localhost:8000` before the UI can answer questions.
- **`_skip_health` must be defined before use** — `logfire.instrument_fastapi(app, request_attributes_mapper=_skip_health)` references `_skip_health`, which is defined after the `app` object. This works because the call happens inside `lifespan` at runtime, not at module import time.

---

## 9. Definition of Done

- [ ] [`services/logfire/factory.py`](Agentic-RAG-project-agentops/src/services/logfire/factory.py) defines `configure_logfire(settings)`.
- [ ] [`mcp_server/server.py`](Agentic-RAG-project-agentops/src/mcp_server/server.py) defines the `arxiv-rag` `FastMCP` instance, `MCPContext`, `set_mcp_context`, and `get_mcp_context`.
- [ ] [`mcp_server/tools/ask.py`](Agentic-RAG-project-agentops/src/mcp_server/tools/ask.py) registers the `ask_question` tool.
- [ ] [`mcp_server/tools/feedback.py`](Agentic-RAG-project-agentops/src/mcp_server/tools/feedback.py) registers `submit_feedback` and `get_index_stats`.
- [ ] [`mcp_server/tools/papers.py`](Agentic-RAG-project-agentops/src/mcp_server/tools/papers.py) registers `get_paper_details` and `list_recent_papers`.
- [ ] [`mcp_server/tools/search.py`](Agentic-RAG-project-agentops/src/mcp_server/tools/search.py) registers `search_papers` and defines `ChunkResult`.
- [ ] [`mcp_server/resources/papers.py`](Agentic-RAG-project-agentops/src/mcp_server/resources/papers.py) registers `papers://{arxiv_id}` and `index://stats` resources.
- [ ] [`services/telegram/bot.py`](Agentic-RAG-project-agentops/src/services/telegram/bot.py) defines the `TelegramBot` class with `start`, `stop`, and command handlers.
- [ ] [`services/telegram/factory.py`](Agentic-RAG-project-agentops/src/services/telegram/factory.py) defines `make_telegram_service`.
- [ ] [`gradio_app.py`](Agentic-RAG-project-agentops/src/gradio_app.py) defines `stream_response`, `create_gradio_interface`, and `main`.
- [ ] [`main.py`](Agentic-RAG-project-agentops/src/main.py) wires all services into `app.state`, sets the MCP context, starts the Telegram bot with the flock lock, mounts the MCP sub-app, and includes all routers.
- [ ] The server boots with `uvicorn src.main:app` and `GET /api/v1/health` returns `status: ok`.
- [ ] The MCP tools and resources are registered and reachable.
- [ ] The Gradio UI streams answers from the API.
- [ ] The Telegram bot starts and responds to commands and questions.
