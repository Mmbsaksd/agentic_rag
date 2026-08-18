# PHASE 15 — Agentic RAG Graph

> Build guide for wiring the PHASE 14 agent nodes into a compiled LangGraph state graph, plus the retriever tool, the `AgenticRAGService` facade, the factory, and the Supervisor/Summarizer agents. This phase produces the fully executable agentic RAG workflow that PHASE 16 exposes over HTTP.

---

## 1. Phase Objective

In PHASE 14 you implemented the individual agent nodes as pure functions that receive `(state, runtime)`. In this phase you will:

1. Create the **retriever tool** (`tools.py`) that wraps OpenSearch + embeddings into a LangChain `@tool` callable by the `ToolNode`.
2. Build the **LangGraph state graph** in `AgenticRAGService._build_graph()` that wires all nodes together with conditional routing.
3. Implement the **`AgenticRAGService.ask()`** entry point that initializes state, builds the runtime `Context`, and invokes the compiled graph with Langfuse tracing.
4. Add **graph introspection helpers** (`get_graph_visualization`, `get_graph_mermaid`, `get_graph_ascii`).
5. Create the **factory** `make_agentic_rag_service()` for dependency-injected construction.
6. Implement the **`SupervisorAgent`** and **`SummarizerAgent`** that route between RAG lookup and topic summarization.
7. Update the `agents/__init__.py` public exports.

By the end of this phase you will have a complete, runnable agentic RAG service ready to be wired into FastAPI routers in PHASE 16.

---

## 2. Prerequisites

- PHASE 13 completed: `AgentState`, `Context`, and `GraphConfig` exist in [`src/services/agents/`](Agentic-RAG-project-agentops/src/services/agents/).
- PHASE 14 completed: all 8 node functions exist in [`src/services/agents/nodes/`](Agentic-RAG-project-agentops/src/services/agents/nodes/) and are re-exported from [`nodes/__init__.py`](Agentic-RAG-project-agentops/src/services/agents/nodes/__init__.py).
- PHASE 6 completed: `JinaEmbeddingsClient` with `embed_query()`.
- PHASE 7 completed: `OpenSearchClient` with `search_unified()`.
- PHASE 5 completed: `LLMClientProtocol` with `generate_rag_answer()`.
- PHASE 12 completed: Langfuse tracer (`LangfuseTracer`) and Bedrock Guardrails service are available.

---

## 3. Dependencies to Install

The dependencies for this phase are already declared in `pyproject.toml` from earlier phases. Confirm the following are present:

```bash
pip install langgraph langchain-core langfuse
```

- `langgraph` — for `StateGraph`, `START`, `END`, `ToolNode`, `tools_condition`, and `Runtime`.
- `langchain-core` — for `HumanMessage`, `AIMessage`, `Document`, and the `@tool` decorator.
- `langfuse` — for `CallbackHandler` and the v4 SDK `start_as_current_observation` tracing API.

No new packages are required beyond what PHASE 13 introduced.

---

## 4. Directory Structure to Create

All files in this phase live inside the existing [`src/services/agents/`](Agentic-RAG-project-agentops/src/services/agents/) package. No new directories are required. The final structure is:

```
src/services/agents/
├── __init__.py          # public exports (update)
├── agentic_rag.py       # AgenticRAGService + graph build + ask()  (NEW)
├── config.py            # GraphConfig (PHASE 13)
├── context.py           # Context dataclass (PHASE 13)
├── factory.py           # make_agentic_rag_service() (NEW)
├── models.py            # shared models (PHASE 13)
├── prompts.py           # shared prompts (PHASE 13)
├── state.py             # AgentState (PHASE 13)
├── supervisor_agent.py  # SupervisorAgent (NEW)
├── summarizer_agent.py  # SummarizerAgent (NEW)
├── tools.py             # create_retriever_tool() (NEW)
└── nodes/               # PHASE 14 node implementations
```

---

## 5. Step-by-Step Implementation

### Step 1 — `tools.py` (retriever tool)

The retriever tool wraps the OpenSearch + embeddings clients into a LangChain `@tool` so it can be executed by the `ToolNode` in the graph. It embeds the query, calls `search_unified`, and converts each `SearchHit` into a LangChain `Document` with rich metadata.

Create [`tools.py`](Agentic-RAG-project-agentops/src/services/agents/tools.py):

```python
import logging

from langchain_core.documents import Document
from langchain_core.tools import tool
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)


def create_retriever_tool(
    opensearch_client: OpenSearchClient,
    embeddings_client: JinaEmbeddingsClient,
    top_k: int = 3,
    use_hybrid: bool = True,
):
    """Create a retriever tool that wraps OpenSearch service.

    :param opensearch_client: Existing OpenSearch service
    :param embeddings_client: Existing Jina embeddings service
    :param top_k: Number of chunks to retrieve
    :param use_hybrid: Use hybrid search (BM25 + vector)
    :returns: LangChain tool for retrieving papers
    """

    @tool
    async def retrieve_papers(query: str) -> list[Document]:
        """Search and return relevant arXiv research papers.

        Use this tool when the user asks about:
        - Machine learning concepts or techniques
        - Deep learning architectures
        - Natural language processing
        - Computer vision methods
        - AI research topics
        - Specific algorithms or models

        :param query: The search query describing what papers to find
        :returns: List of relevant paper excerpts with metadata
        """
        logger.info(f"Retrieving papers for query: {query[:100]}...")
        logger.debug(f"Search mode: {'hybrid' if use_hybrid else 'bm25'}, top_k: {top_k}")

        # Generate query embedding
        logger.debug("Generating query embedding")
        query_embedding = await embeddings_client.embed_query(query)
        logger.debug(f"Generated embedding with {len(query_embedding)} dimensions")

        # Search using OpenSearch
        logger.debug("Searching OpenSearch")
        search_results = opensearch_client.search_unified(
            query=query,
            query_embedding=query_embedding,
            size=top_k,
            use_hybrid=use_hybrid,
        )

        # Convert SearchHit to LangChain Document
        documents = []
        hits = search_results.get("hits", [])
        logger.info(f"Found {len(hits)} documents from OpenSearch")

        for hit in hits:
            doc = Document(
                page_content=hit["chunk_text"],
                metadata={
                    "arxiv_id": hit["arxiv_id"],
                    "title": hit.get("title", ""),
                    "authors": hit.get("authors", ""),
                    "score": hit.get("score", 0.0),
                    "source": f"https://arxiv.org/pdf/{hit['arxiv_id']}.pdf",
                    "section": hit.get("section_name", ""),
                    "search_mode": "hybrid" if use_hybrid else "bm25",
                    "top_k": top_k,
                },
            )
            documents.append(doc)

        logger.debug(f"Converted {len(documents)} hits to LangChain Documents")
        logger.info(f"✓ Retrieved {len(documents)} papers successfully")

        return documents

    return retrieve_papers
```

**Explanation:** `create_retriever_tool` is a factory that closes over the OpenSearch and embeddings clients and returns an async `@tool` named `retrieve_papers`. The tool's docstring doubles as the LLM-facing description that helps the model decide when to call it. Inside, it embeds the query, runs `search_unified` (hybrid or BM25 based on `use_hybrid`), and maps each hit to a `Document` whose `page_content` is the chunk text and whose `metadata` carries `arxiv_id`, `title`, `authors`, `score`, `source` URL, `section`, `search_mode`, and `top_k`. This metadata is later parsed by `extract_sources_from_tool_messages` in PHASE 14's `utils.py`.

---

### Step 2 — `agentic_rag.py` (graph build + service facade)

This is the core file of the phase. `AgenticRAGService` builds the compiled LangGraph once in `__init__`, then exposes an `ask()` method that initializes state, constructs the runtime `Context`, and invokes the graph.

Create [`agentic_rag.py`](Agentic-RAG-project-agentops/src/services/agents/agentic_rag.py):

```python
import logging
import time
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from src.services.bedrock_guardrails.service import BedrockGuardrailsService
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.llm_client_protocol import LLMClientProtocol
from src.services.opensearch.client import OpenSearchClient

from .config import GraphConfig
from .context import Context
from .nodes import (
    ainvoke_generate_answer_step,
    ainvoke_grade_documents_step,
    ainvoke_guardrail_step,
    ainvoke_out_of_scope_step,
    ainvoke_output_guardrail_step,
    ainvoke_retrieve_step,
    ainvoke_rewrite_query_step,
    continue_after_guardrail,
)
from .state import AgentState
from .tools import create_retriever_tool

logger = logging.getLogger(__name__)


class AgenticRAGService:
    """Agentic RAG service

    This implementation uses:
    - context_schema for dependency injection
    - Runtime[Context] for type-safe access in nodes
    - Direct client invocation (no pre-built runnables)
    - Lightweight nodes as pure functions
    """

    def __init__(
        self,
        opensearch_client: OpenSearchClient,
        llm_client: LLMClientProtocol,
        embeddings_client: JinaEmbeddingsClient,
        langfuse_tracer: Optional[LangfuseTracer] = None,
        guardrails_service: Optional[BedrockGuardrailsService] = None,
        graph_config: Optional[GraphConfig] = None,
    ):
        """Initialize agentic RAG service.

        :param opensearch_client: Client for document search
        :param llm_client: LLM client (OpenAI or Bedrock)
        :param embeddings_client: Client for embeddings
        :param langfuse_tracer: Optional Langfuse tracer
        :param guardrails_service: Optional Bedrock Guardrails service
        :param graph_config: Configuration for graph execution
        """
        self.opensearch = opensearch_client
        self.llm = llm_client
        self.embeddings = embeddings_client
        self.langfuse_tracer = langfuse_tracer
        self.guardrails_service = guardrails_service
        self.graph_config = graph_config or GraphConfig()

        logger.info("Initializing AgenticRAGService with configuration:")
        logger.info(f"  Model: {self.graph_config.model}")
        logger.info(f"  Top-k: {self.graph_config.top_k}")
        logger.info(f"  Hybrid search: {self.graph_config.use_hybrid}")
        logger.info(f"  Max retrieval attempts: {self.graph_config.max_retrieval_attempts}")
        logger.info(f"  Guardrail threshold: {self.graph_config.guardrail_threshold}")

        # Build graph once (no runnables needed!)
        self.graph = self._build_graph()
        logger.info("✓ AgenticRAGService initialized successfully")

    def _build_graph(self):
        """Build and compile the LangGraph workflow.

        Uses context_schema for type-safe dependency injection.
        Nodes are lightweight functions that receive Runtime[Context].

        :returns: Compiled graph ready for invocation
        """
        logger.info("Building LangGraph workflow with context_schema")

        # Create workflow with AgentState and Context schema
        workflow = StateGraph(AgentState, context_schema=Context)

        # Create tools (these still need to be created upfront for ToolNode)
        retriever_tool = create_retriever_tool(
            opensearch_client=self.opensearch,
            embeddings_client=self.embeddings,
            top_k=self.graph_config.top_k,
            use_hybrid=self.graph_config.use_hybrid,
        )
        tools = [retriever_tool]

        # Add nodes (just function references - no closures needed!)
        logger.info("Adding nodes to workflow graph")
        workflow.add_node("guardrail", ainvoke_guardrail_step)
        workflow.add_node("out_of_scope", ainvoke_out_of_scope_step)
        workflow.add_node("retrieve", ainvoke_retrieve_step)
        workflow.add_node("tool_retrieve", ToolNode(tools))
        workflow.add_node("grade_documents", ainvoke_grade_documents_step)
        workflow.add_node("rewrite_query", ainvoke_rewrite_query_step)
        workflow.add_node("generate_answer", ainvoke_generate_answer_step)
        workflow.add_node("output_guardrail", ainvoke_output_guardrail_step)

        # Add edges
        logger.info("Configuring graph edges and routing logic")

        # Start → guardrail validation
        workflow.add_edge(START, "guardrail")

        # Guardrail → route based on score
        workflow.add_conditional_edges(
            "guardrail",
            continue_after_guardrail,
            {
                "continue": "retrieve",
                "out_of_scope": "out_of_scope",
            },
        )

        # Out of scope → END
        workflow.add_edge("out_of_scope", END)

        # Retrieve node creates tool call
        workflow.add_conditional_edges(
            "retrieve",
            tools_condition,
            {
                "tools": "tool_retrieve",
                END: END,
            },
        )

        # After tool retrieval → grade documents
        workflow.add_edge("tool_retrieve", "grade_documents")

        # After grading → route based on relevance
        workflow.add_conditional_edges(
            "grade_documents",
            lambda state: state.get("routing_decision", "generate_answer"),
            {
                "generate_answer": "generate_answer",
                "rewrite_query": "rewrite_query",
            },
        )

        # After rewriting → try retrieve again
        workflow.add_edge("rewrite_query", "retrieve")

        # After answer generation → output guardrail → done
        workflow.add_edge("generate_answer", "output_guardrail")
        workflow.add_edge("output_guardrail", END)

        # Compile graph
        logger.info("Compiling LangGraph workflow")
        compiled_graph = workflow.compile()
        logger.info("✓ Graph compilation successful")

        return compiled_graph
```

**Explanation:** `_build_graph()` constructs a `StateGraph(AgentState, context_schema=Context)`. It creates the retriever tool and wraps it in a `ToolNode` named `tool_retrieve`. All 8 PHASE 14 nodes are added by function reference. The edges implement the full agentic loop:

- `START → guardrail`
- `guardrail` conditionally routes via `continue_after_guardrail` to `retrieve` (in-scope) or `out_of_scope` (blocked).
- `out_of_scope → END`
- `retrieve` uses `tools_condition` to route to `tool_retrieve` (when a tool call is present) or `END` (when the retrieval loop is exhausted and a fallback message was emitted).
- `tool_retrieve → grade_documents`
- `grade_documents` routes via `routing_decision` to `generate_answer` or `rewrite_query`.
- `rewrite_query → retrieve` (retry loop, capped by `max_retrieval_attempts`).
- `generate_answer → output_guardrail → END`.

The graph is compiled once in `__init__` and reused across requests.

---

### Step 3 — `agentic_rag.py` (ask + workflow execution)

Continue appending to [`agentic_rag.py`](Agentic-RAG-project-agentops/src/services/agents/agentic_rag.py) with the `ask()` method, the `_run_workflow()` executor, and the result-extraction helpers:

```python
    async def ask(
        self,
        query: str,
        user_id: str = "api_user",
        model: Optional[str] = None,
    ) -> dict:
        """Ask a question using agentic RAG.

        :param query: User question
        :param user_id: User identifier for tracing
        :param model: Optional model override
        :returns: Dictionary with answer, sources, reasoning steps, and metadata
        :raises ValueError: If query is empty
        """
        model_to_use = model or self.graph_config.model

        logger.info("=" * 80)
        logger.info("Starting Agentic RAG Request")
        logger.info(f"Query: {query}")
        logger.info(f"User ID: {user_id}")
        logger.info(f"Model: {model_to_use}")
        logger.info("=" * 80)

        # Validate input
        if not query or len(query.strip()) == 0:
            logger.error("Empty query received")
            raise ValueError("Query cannot be empty")

        # Create trace if Langfuse is enabled (v3 SDK)
        trace = None
        if self.langfuse_tracer and self.langfuse_tracer.client:
            logger.info("Creating Langfuse trace (v3 SDK)")
            metadata = {
                "env": self.graph_config.settings.environment,
                "service": "agentic_rag",
                "top_k": self.graph_config.top_k,
                "use_hybrid": self.graph_config.use_hybrid,
                "model": model_to_use,
            }
            # V4 SDK: start_as_current_observation replaces start_as_current_span
            trace = self.langfuse_tracer.client.start_as_current_observation(
                name="agentic_rag_request",
                as_type="span",
            )

        # Use proper context manager pattern
        async def _execute_with_trace():
            """Execute the workflow with or without tracing context."""
            if trace is not None:
                with trace as trace_obj:
                    trace_obj.update(
                        input={"query": query},
                        metadata=metadata,
                        user_id=user_id,
                        session_id=f"session_{user_id}",
                    )
                    logger.debug(f"Trace created: {trace_obj}")
                    return await self._run_workflow(query, model_to_use, user_id, trace_obj)
            else:
                return await self._run_workflow(query, model_to_use, user_id, None)

        try:
            return await _execute_with_trace()
        except Exception as e:
            logger.error(f"Error in Agentic RAG execution: {str(e)}")
            logger.exception("Full traceback:")
            raise

    async def _run_workflow(self, query: str, model_to_use: str, user_id: str, trace) -> dict:
        """Execute the workflow with the given trace context."""
        try:
            start_time = time.time()

            # Capture trace_id now while inside active trace context
            trace_id = self.langfuse_tracer.get_trace_id() if self.langfuse_tracer else None

            logger.info("Invoking LangGraph workflow")

            # State initialization
            state_input = {
                "messages": [HumanMessage(content=query)],
                "retrieval_attempts": 0,
                "guardrail_result": None,
                "routing_decision": None,
                "sources": None,
                "relevant_sources": [],
                "relevant_tool_artefacts": None,
                "grading_results": [],
                "metadata": {},
                "original_query": None,
                "rewritten_query": None,
                "sanitized_query": None,
                "output_guardrail_filter": None,
            }

            # Runtime context (dependencies)
            runtime_context = Context(
                llm_client=self.llm,
                opensearch_client=self.opensearch,
                embeddings_client=self.embeddings,
                langfuse_tracer=self.langfuse_tracer,
                guardrails_service=self.guardrails_service,
                trace=trace,
                langfuse_enabled=self.langfuse_tracer is not None and self.langfuse_tracer.client is not None,
                model_name=model_to_use,
                temperature=self.graph_config.temperature,
                top_k=self.graph_config.top_k,
                max_retrieval_attempts=self.graph_config.max_retrieval_attempts,
                guardrail_threshold=self.graph_config.guardrail_threshold,
            )

            # Create config with CallbackHandler if Langfuse is enabled (v3 SDK)
            config = {"thread_id": f"user_{user_id}_session_{int(time.time())}"}

            # Add CallbackHandler for automatic LLM tracing
            # IMPORTANT: CallbackHandler automatically inherits the current span context
            # Since we're inside start_as_current_span, it will be linked automatically
            if self.langfuse_tracer and trace:
                try:
                    # V3 SDK: CallbackHandler() automatically uses current trace context
                    # No need to pass trace explicitly - it's handled by context propagation
                    callback_handler = CallbackHandler()
                    config["callbacks"] = [callback_handler]
                    logger.info("✓ CallbackHandler added (will auto-link to current trace)")
                except Exception as e:
                    logger.warning(f"Failed to create CallbackHandler: {e}")

            result = await self.graph.ainvoke(
                state_input,
                config=config,
                context=runtime_context,
            )

            execution_time = time.time() - start_time
            logger.info(f"✓ Graph execution completed in {execution_time:.2f}s")

            # Extract results
            answer = self._extract_answer(result)
            sources = self._extract_sources(result)
            retrieval_attempts = result.get("retrieval_attempts", 0)
            reasoning_steps = self._extract_reasoning_steps(result)

            # Update trace (cleanup handled by context manager)
            if trace:
                trace.update(
                    output={
                        "answer": answer,
                        "sources_count": len(sources),
                        "retrieval_attempts": retrieval_attempts,
                        "reasoning_steps": reasoning_steps,
                        "execution_time": execution_time,
                    }
                )
                trace.end()
                self.langfuse_tracer.flush()

            logger.info("=" * 80)
            logger.info("Agentic RAG Request Completed Successfully")
            logger.info(f"Answer length: {len(answer)} characters")
            logger.info(f"Sources found: {len(sources)}")
            logger.info(f"Retrieval attempts: {retrieval_attempts}")
            logger.info(f"Execution time: {execution_time:.2f}s")
            logger.info("=" * 80)

            return {
                "query": query,
                "answer": answer,
                "sources": sources,
                "reasoning_steps": reasoning_steps,
                "retrieval_attempts": retrieval_attempts,
                "rewritten_query": result.get("rewritten_query"),
                "execution_time": execution_time,
                "guardrail_score": result.get("guardrail_result").score if result.get("guardrail_result") else None,
                "guardrail_filter": result.get("guardrail_result").reason if result.get("guardrail_result") else None,
                "output_guardrail_filter": result.get("output_guardrail_filter"),
                "trace_id": trace_id,
            }

        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}")
            logger.exception("Full traceback:")

            # Update trace with error (cleanup handled by context manager)
            if trace:
                trace.update(output={"error": str(e)}, level="ERROR")
                trace.end()
                self.langfuse_tracer.flush()

            raise
```

**Explanation:** `ask()` validates the query, creates a Langfuse trace via the v4 SDK `start_as_current_observation` (span type), and executes the workflow inside a context manager so the trace is properly scoped. `_run_workflow()` builds the full `state_input` dict (initializing every `AgentState` key), constructs the runtime `Context` from the injected clients and `GraphConfig`, adds a `CallbackHandler` to the config for automatic LLM tracing, and invokes `self.graph.ainvoke(state_input, config=config, context=runtime_context)`. After execution it extracts the answer, sources, retrieval attempts, and reasoning steps, updates the trace, and returns a rich result dict including `guardrail_score`, `guardrail_filter`, `output_guardrail_filter`, and `trace_id`.

---

### Step 4 — `agentic_rag.py` (extractors + graph introspection)

Append the result-extraction and graph-introspection helpers to [`agentic_rag.py`](Agentic-RAG-project-agentops/src/services/agents/agentic_rag.py):

```python
    def _extract_answer(self, result: dict) -> str:
        """Extract final answer from graph result."""
        messages = result.get("messages", [])
        if not messages:
            return "No answer generated."

        final_message = messages[-1]
        return final_message.content if hasattr(final_message, "content") else str(final_message)

    def _extract_sources(self, result: dict) -> List[dict]:
        """Extract sources from graph result."""
        sources = []
        relevant_sources = result.get("relevant_sources", [])

        for source in relevant_sources:
            if hasattr(source, "to_dict"):
                sources.append(source.to_dict())
            elif isinstance(source, dict):
                sources.append(source)

        return sources

    def _extract_reasoning_steps(self, result: dict) -> List[str]:
        """Extract reasoning steps from graph result."""
        steps = []
        retrieval_attempts = result.get("retrieval_attempts", 0)
        guardrail_result = result.get("guardrail_result")
        grading_results = result.get("grading_results", [])

        if guardrail_result:
            steps.append(f"Validated query scope (score: {guardrail_result.score}/100)")

        if retrieval_attempts > 0:
            steps.append(f"Retrieved documents ({retrieval_attempts} attempt(s))")

        if grading_results:
            relevant_count = sum(1 for g in grading_results if g.is_relevant)
            steps.append(f"Graded documents ({relevant_count} relevant)")

        if result.get("rewritten_query"):
            steps.append("Rewritten query for better results")

        steps.append("Generated answer from context")

        return steps

    def get_graph_visualization(self) -> bytes:
        """Get the LangGraph workflow visualization as PNG.

        This method generates a visual representation of the graph workflow
        using mermaid diagram format, then converts it to PNG.

        :returns: PNG image bytes
        :raises ImportError: If required dependencies (pygraphviz/graphviz) are not installed
        :raises Exception: If graph visualization generation fails

        Example:
            >>> service = AgenticRAGService(...)
            >>> png_bytes = service.get_graph_visualization()
            >>> with open("graph.png", "wb") as f:
            ...     f.write(png_bytes)
        """
        try:
            logger.info("Generating graph visualization as PNG")
            png_bytes = self.graph.get_graph().draw_mermaid_png()
            logger.info(f"✓ Generated PNG visualization ({len(png_bytes)} bytes)")
            return png_bytes
        except ImportError as e:
            logger.error(f"Failed to generate visualization - missing dependencies: {e}")
            logger.error("Install with: pip install pygraphviz or apt-get install graphviz")
            raise ImportError(
                "Graph visualization requires pygraphviz. "
                "Install with: pip install pygraphviz (requires graphviz system package)"
            ) from e
        except Exception as e:
            logger.error(f"Failed to generate graph visualization: {e}")
            raise

    def get_graph_mermaid(self) -> str:
        """Get the LangGraph workflow as a mermaid diagram string.

        This method generates the graph workflow representation in mermaid
        diagram syntax, which can be rendered in markdown or mermaid viewers.

        :returns: Mermaid diagram syntax as string

        Example:
            >>> service = AgenticRAGService(...)
            >>> mermaid = service.get_graph_mermaid()
            >>> print(mermaid)
            graph TD
                __start__ --> guardrail
                ...
        """
        try:
            logger.info("Generating graph as mermaid diagram")
            mermaid_str = self.graph.get_graph().draw_mermaid()
            logger.info(f"✓ Generated mermaid diagram ({len(mermaid_str)} characters)")
            return mermaid_str
        except Exception as e:
            logger.error(f"Failed to generate mermaid diagram: {e}")
            raise

    def get_graph_ascii(self) -> str:
        """Get ASCII representation of the graph.

        This method generates a simple ASCII art representation of the
        graph structure, useful for quick inspection in terminals.

        :returns: ASCII art representation of the graph

        Example:
            >>> service = AgenticRAGService(...)
            >>> print(service.get_graph_ascii())
        """
        try:
            logger.info("Generating ASCII graph representation")
            ascii_str = self.graph.get_graph().print_ascii()
            logger.info("✓ Generated ASCII graph representation")
            return ascii_str
        except Exception as e:
            logger.error(f"Failed to generate ASCII graph: {e}")
            raise
```

**Explanation:** `_extract_answer` returns the content of the last message. `_extract_sources` converts `relevant_sources` items (which may be `SourceItem` objects with `to_dict()` or plain dicts) into serializable dicts. `_extract_reasoning_steps` builds a human-readable list of steps from guardrail, retrieval, grading, and rewrite state. The three `get_graph_*` methods expose the compiled graph for visualization/debugging: PNG bytes, a mermaid string, and ASCII art respectively.

---

### Step 5 — `factory.py`

The factory provides a dependency-injected constructor for `AgenticRAGService`, picking the model ID based on the configured provider and building a `GraphConfig`.

Create [`factory.py`](Agentic-RAG-project-agentops/src/services/agents/factory.py):

```python
from typing import Optional

from src.config import Settings, get_settings
from src.services.bedrock_guardrails.service import BedrockGuardrailsService
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.llm_client_protocol import LLMClientProtocol
from src.services.opensearch.client import OpenSearchClient

from .agentic_rag import AgenticRAGService
from .config import GraphConfig


def make_agentic_rag_service(
    opensearch_client: OpenSearchClient,
    llm_client: LLMClientProtocol,
    embeddings_client: JinaEmbeddingsClient,
    langfuse_tracer: Optional[LangfuseTracer] = None,
    guardrails_service: Optional[BedrockGuardrailsService] = None,
    top_k: int = 3,
    use_hybrid: bool = True,
    settings: Optional[Settings] = None,
) -> AgenticRAGService:
    """Create AgenticRAGService with dependency injection.

    Args:
        opensearch_client: Client for document search
        llm_client: LLM client (OpenAI or Bedrock)
        embeddings_client: Client for embeddings
        langfuse_tracer: Optional Langfuse tracer for observability
        guardrails_service: Optional Bedrock Guardrails service
        top_k: Number of documents to retrieve (default: 3)
        use_hybrid: Use hybrid search (default: True)
        settings: Application settings (reads model from .env)

    Returns:
        Configured AgenticRAGService instance
    """
    if settings is None:
        settings = get_settings()

    # Pick the model ID based on provider
    if settings.provider == "bedrock":
        model = settings.bedrock.model_id
    else:
        model = settings.openai_model

    graph_config = GraphConfig(
        top_k=top_k,
        use_hybrid=use_hybrid,
        model=model,
    )

    return AgenticRAGService(
        opensearch_client=opensearch_client,
        llm_client=llm_client,
        embeddings_client=embeddings_client,
        langfuse_tracer=langfuse_tracer,
        guardrails_service=guardrails_service,
        graph_config=graph_config,
    )
```

**Explanation:** `make_agentic_rag_service` centralizes construction. It resolves the model ID from `settings.provider` (`bedrock.model_id` for Bedrock, `openai_model` otherwise), builds a `GraphConfig` with the requested `top_k` and `use_hybrid`, and returns a fully wired `AgenticRAGService`. This factory is what PHASE 16's dependency provider will call.

---

### Step 6 — `summarizer_agent.py`

The `SummarizerAgent` handles "summarize" intents: it performs a BM25 text search for the topic, then asks the LLM to summarize the top chunks.

Create [`summarizer_agent.py`](Agentic-RAG-project-agentops/src/services/agents/summarizer_agent.py):

```python
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.services.agents.context import Context

logger = logging.getLogger(__name__)


@dataclass
class SummaryResult:
    summary: str
    topic: str
    chunks_used: int


class SummarizerAgent:
    """Minimal summarizer: search topic → ask LLM to summarize results."""

    def __init__(self, context: Context) -> None:
        self._ctx = context

    async def summarize(self, topic: str) -> SummaryResult:
        logger.info("SummarizerAgent: summarizing topic=%s", topic)

        # Reuse existing opensearch client for text search
        results = self._ctx.opensearch_client.search_unified(
            query=topic,
            size=5,
            use_hybrid=False,
        )
        hits = results.get("hits", [])
        chunks = [hit.get("text", "") for hit in hits]
        chunks = [c for c in chunks if c]

        if not chunks:
            return SummaryResult(
                summary=f"No papers found about '{topic}'.",
                topic=topic,
                chunks_used=0,
            )

        context_text = "\n\n".join(chunks[:5])
        prompt = (
            f"Summarize the following research excerpts about '{topic}' in 3-5 sentences:\n\n"
            f"{context_text}"
        )

        # Reuse existing LLM client
        llm_result = await self._ctx.llm_client.generate_rag_answer(
            query=prompt,
            chunks=chunks,
            model=self._ctx.model_name,
        )
        summary = llm_result.get("answer", "")

        return SummaryResult(
            summary=summary,
            topic=topic,
            chunks_used=len(chunks),
        )
```

**Explanation:** `SummarizerAgent` reuses the `Context`'s OpenSearch and LLM clients. `summarize()` runs a BM25 search (`use_hybrid=False`) for the topic, collects up to 5 non-empty chunks, builds a summarization prompt, and calls `generate_rag_answer`. It returns a `SummaryResult` with the summary text, the topic, and the number of chunks used. If no chunks are found it returns a friendly "No papers found" message.

---

### Step 7 — `supervisor_agent.py`

The `SupervisorAgent` classifies a query's intent (`rag_lookup` vs `summarize`) using the LLM, then routes to either the `AgenticRAGService` or the `SummarizerAgent`.

Create [`supervisor_agent.py`](Agentic-RAG-project-agentops/src/services/agents/supervisor_agent.py):

```python
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from src.services.agents.context import Context
from src.services.agents.summarizer_agent import SummarizerAgent, SummaryResult

logger = logging.getLogger(__name__)

IntentType = Literal["rag_lookup", "summarize"]

INTENT_PROMPT = """You are a router. Classify the user query intent as exactly one word.

Respond with ONLY one of:
- "summarize" — if the user wants a summary or overview of a topic/paper/area
- "rag_lookup" — if the user wants a specific answer, explanation, or comparison

Query: {query}

Intent:"""


@dataclass
class SupervisorResult:
    answer: str
    intent: IntentType
    routed_to: str
    sources: list = field(default_factory=list)


class SupervisorAgent:
    """Routes queries to RAG agent or Summarizer agent based on LLM intent classification."""

    def __init__(self, context: Context, agentic_rag_service) -> None:
        self._ctx = context
        self._rag_agent = agentic_rag_service
        self._summarizer = SummarizerAgent(context)

    async def _classify_intent(self, query: str) -> IntentType:
        prompt = INTENT_PROMPT.format(query=query)
        raw = await self._ctx.llm_client.generate_rag_answer(
            query=prompt,
            chunks=[],
            model=self._ctx.model_name,
        )
        intent_raw = raw.get("answer", "").strip().lower().strip('"').strip("'")
        if "summarize" in intent_raw:
            return "summarize"
        return "rag_lookup"

    async def ask(self, query: str) -> SupervisorResult:
        intent = await self._classify_intent(query)
        logger.info("SupervisorAgent: query routed to intent=%s", intent)

        if intent == "summarize":
            result: SummaryResult = await self._summarizer.summarize(topic=query)
            return SupervisorResult(
                answer=result.summary,
                intent=intent,
                routed_to="SummarizerAgent",
            )

        # Default: delegate to existing RAG agent
        rag_result = await self._rag_agent.ask(query=query)
        return SupervisorResult(
            answer=rag_result["answer"],
            intent=intent,
            routed_to="AgenticRAGService",
            sources=rag_result.get("sources", []),
        )
```

**Explanation:** `SupervisorAgent` holds a `Context` and an `AgenticRAGService`. `_classify_intent` formats the `INTENT_PROMPT` with the query and asks the LLM to return exactly one word (`summarize` or `rag_lookup`), defaulting to `rag_lookup` if the word isn't recognized. `ask()` routes accordingly: summarize intents go to `SummarizerAgent.summarize()`, everything else delegates to `AgenticRAGService.ask()`. It returns a `SupervisorResult` with the answer, the detected intent, the routing target, and (for RAG lookups) the sources.

---

### Step 8 — `__init__.py` (public exports)

Update the package exports so the service, config, context, state, and factory are importable from `src.services.agents`.

Update [`__init__.py`](Agentic-RAG-project-agentops/src/services/agents/__init__.py):

```python
from .agentic_rag import AgenticRAGService
from .config import GraphConfig
from .context import Context
from .factory import make_agentic_rag_service
from .state import AgentState

__all__ = [
    "AgenticRAGService",
    "GraphConfig",
    "Context",
    "AgentState",
    "make_agentic_rag_service",
]
```

**Explanation:** This keeps the public surface of the `agents` package stable and explicit. Consumers (like the PHASE 16 dependency provider and routers) can import `AgenticRAGService` and `make_agentic_rag_service` directly from `src.services.agents`.

---

## 6. Configuration

The agentic RAG graph is configured through `GraphConfig` (defined in PHASE 13) and the runtime `Context`. The relevant knobs are:

| Setting | Default | Description |
| --- | --- | --- |
| `max_retrieval_attempts` | `2` | Maximum retrieval attempts before the loop falls back to a message |
| `guardrail_threshold` | `40` | Score threshold (0-100) for guardrail validation |
| `model` | `"gpt-4o-mini"` | Default LLM model (overridden by `make_agentic_rag_service` based on provider) |
| `temperature` | `0.0` | LLM generation temperature (deterministic) |
| `top_k` | `3` | Number of documents retrieved per search |
| `use_hybrid` | `True` | Whether to use hybrid search (BM25 + vector) |
| `enable_tracing` | `True` | Whether Langfuse tracing is enabled |
| `settings` | `get_settings()` | Application settings for environment and service config |

The factory `make_agentic_rag_service` resolves the model ID from `settings.provider`:
- `bedrock` → `settings.bedrock.model_id`
- otherwise → `settings.openai_model`

No new environment variables are required for this phase. Tracing is enabled automatically when a `LangfuseTracer` with a live client is passed in.

---

## 7. Verification

Verify the agentic RAG graph works end to end:

1. **Import check** — confirm the package and service import cleanly:
   ```bash
   python -c "from src.services.agents import AgenticRAGService, make_agentic_rag_service, GraphConfig, Context, AgentState"
   ```

2. **Unit tests** — run the agent test suites:
   ```bash
   pytest tests/unit/services/agents/test_agentic_rag.py tests/unit/services/agents/test_tools.py -v
   ```

3. **Graph introspection** — instantiate the service (or use a test fixture) and confirm the graph builds and can be rendered:
   ```python
   service = make_agentic_rag_service(opensearch_client, llm_client, embeddings_client)
   print(service.get_graph_ascii())
   print(service.get_graph_mermaid())
   ```

4. **End-to-end ask** — with OpenSearch populated, run a query through `ask()` and confirm the returned dict contains `answer`, `sources`, `reasoning_steps`, `retrieval_attempts`, `guardrail_score`, `guardrail_filter`, `output_guardrail_filter`, and `trace_id`.

5. **Retrieval loop** — ask a query that yields low-relevance results and confirm the graph rewrites the query and retries up to `max_retrieval_attempts`.

6. **Supervisor routing** — call `SupervisorAgent.ask()` with a summarization-style query and confirm it routes to `SummarizerAgent`; with a specific question, confirm it routes to `AgenticRAGService`.

---

## 8. Common Pitfalls

- **`context_schema` mismatch** — `StateGraph(AgentState, context_schema=Context)` must match the `Runtime[Context]` type used in every node. A mismatch causes a runtime type error when invoking the graph.
- **ToolNode requires the tool upfront** — the retriever tool must be created before `ToolNode(tools)` and passed in; it cannot be created lazily inside a node.
- **`tools_condition` routing** — the `retrieve` node must return an `AIMessage` with `tool_calls` for `tools_condition` to route to `tool_retrieve`. When the retrieval loop is exhausted, the node returns a plain `AIMessage` (no tool calls), which routes to `END`.
- **State keys must be initialized** — `_run_workflow` initializes every `AgentState` key in `state_input`. Missing keys cause `KeyError` in nodes that read them with `state.get(...)` defaults.
- **Langfuse v4 SDK** — use `start_as_current_observation(name=..., as_type="span")` (v4) rather than the deprecated `start_as_current_span`. The `CallbackHandler` auto-links to the current trace via context propagation.
- **`graph.ainvoke(..., context=runtime_context)`** — the runtime `Context` must be passed via the `context=` keyword argument, not inside `config`.
- **Graph built once** — `_build_graph()` is called in `__init__`; do not rebuild the graph per request (it is stateless and reused).

---

## 9. Definition of Done

- [ ] [`tools.py`](Agentic-RAG-project-agentops/src/services/agents/tools.py) defines `create_retriever_tool()` returning an async `retrieve_papers` `@tool`.
- [ ] [`agentic_rag.py`](Agentic-RAG-project-agentops/src/services/agents/agentic_rag.py) defines `AgenticRAGService` with `_build_graph()`, `ask()`, `_run_workflow()`, result extractors, and graph introspection helpers.
- [ ] The compiled graph wires all 8 PHASE 14 nodes with correct conditional routing and a retrieval retry loop capped at `max_retrieval_attempts`.
- [ ] [`factory.py`](Agentic-RAG-project-agentops/src/services/agents/factory.py) defines `make_agentic_rag_service()` with provider-aware model selection.
- [ ] [`summarizer_agent.py`](Agentic-RAG-project-agentops/src/services/agents/summarizer_agent.py) defines `SummarizerAgent` with `summarize()`.
- [ ] [`supervisor_agent.py`](Agentic-RAG-project-agentops/src/services/agents/supervisor_agent.py) defines `SupervisorAgent` with intent classification and routing.
- [ ] [`__init__.py`](Agentic-RAG-project-agentops/src/services/agents/__init__.py) exports `AgenticRAGService`, `GraphConfig`, `Context`, `AgentState`, and `make_agentic_rag_service`.
- [ ] `pytest tests/unit/services/agents/test_agentic_rag.py tests/unit/services/agents/test_tools.py` passes.
- [ ] The service is ready to be wired into FastAPI routers and the dependency provider in **PHASE 16**.