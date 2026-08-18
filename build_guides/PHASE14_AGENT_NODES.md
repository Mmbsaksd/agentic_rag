# PHASE 14 — Agent Nodes

> **Build guide for the from-scratch reconstruction of the arXiv Paper Curator (Agentic RAG).**
>
> This phase implements the **eight LangGraph node functions** that make up the agentic RAG workflow, plus the shared [`utils.py`](../Agentic-RAG-project-agentops/src/services/agents/nodes/utils.py) helpers. Each node is an **async function** that receives `(state: AgentState, runtime: Runtime[Context])` and returns a partial state update dict. The nodes are pure, dependency-injected functions — they read clients and tuning parameters from `runtime.context` (the `Context` dataclass built in PHASE 13) and never construct their own dependencies. The graph wiring that connects these nodes is built in PHASE 15. The guardrail nodes depend on the `BedrockGuardrailsService` from `src/services/bedrock_guardrails/` (built in a prior phase).

---

## 1. Phase Objective

By the end of this phase you will have:

- The `src/services/agents/nodes/` subpackage with `__init__.py` re-exporting all node functions.
- [`utils.py`](../Agentic-RAG-project-agentops/src/services/agents/nodes/utils.py) with the shared helpers: `get_latest_query`, `get_latest_context`, `extract_sources_from_tool_messages`, `extract_tool_artefacts`, `create_reasoning_step`, `filter_messages`.
- The **input guardrail node** [`guardrail_node.py`](../Agentic-RAG-project-agentops/src/services/agents/nodes/guardrail_node.py) with `ainvoke_guardrail_step` and the `continue_after_guardrail` conditional-edge function.
- The **out-of-scope node** [`out_of_scope_node.py`](../Agentic-RAG-project-agentops/src/services/agents/nodes/out_of_scope_node.py) with `ainvoke_out_of_scope_step`.
- The **retrieve node** [`retrieve_node.py`](../Agentic-RAG-project-agentops/src/services/agents/nodes/retrieve_node.py) with `ainvoke_retrieve_step` (creates a `retrieve_papers` tool call, or returns a fallback when `max_retrieval_attempts` is reached).
- The **grade-documents node** [`grade_documents_node.py`](../Agentic-RAG-project-agentops/src/services/agents/nodes/grade_documents_node.py) with `ainvoke_grade_documents_step` (LLM relevance grading → routing decision).
- The **rewrite-query node** [`rewrite_query_node.py`](../Agentic-RAG-project-agentops/src/services/agents/nodes/rewrite_query_node.py) with `ainvoke_rewrite_query_step` (LLM query rewriting with fallback).
- The **generate-answer node** [`generate_answer_node.py`](../Agentic-RAG-project-agentops/src/services/agents/nodes/generate_answer_node.py) with `ainvoke_generate_answer_step`.
- The **output guardrail node** [`output_guardrail_node.py`](../Agentic-RAG-project-agentops/src/services/agents/nodes/output_guardrail_node.py) with `ainvoke_output_guardrail_step` (grounding/content check on the generated answer).

---

## 2. Prerequisites

- Completion of **PHASE 13 (Agent State & Context)** so that `AgentState`, `Context`, the Pydantic models (`GuardrailScoring`, `GradingResult`, `SourceItem`, `ToolArtefact`, `ReasoningStep`), and the prompt templates in `prompts.py` all exist.
- Completion of **PHASE 5 (LLM Providers)** so that an LLM client satisfying `LLMClientProtocol` exists, including the `get_langchain_model(...)` method used by the grade/rewrite/generate nodes.
- Completion of **PHASE 6 (Embeddings)** so that `JinaEmbeddingsClient` exists.
- Completion of **PHASE 7 (OpenSearch)** so that `OpenSearchClient` exists (used by the retriever tool in PHASE 15, not directly by nodes).
- Completion of **PHASE 12 (RAG Service)** so that `LangfuseTracer` (`src/services/langfuse/client.py`) exists with `create_span`, `end_span`, `update_span`, and `flush`.
- The `BedrockGuardrailsService` from `src/services/bedrock_guardrails/service.py` (built in a prior phase) — imported here for type hints and used at runtime. It may be `None` (fail-open).
- Python 3.11+ and the project's virtual environment active.
- `langgraph`, `langchain-core`, and `logfire` installed (see Section 3).

---

## 3. Dependencies to Install

This phase requires the following third-party packages (all introduced in prior phases, but confirm they are present):

```bash
uv add langgraph langchain-core logfire
```

> `langgraph` provides `Runtime[Context]` (from `langgraph.runtime`) — the type used for the second parameter of every node function. `langchain-core` provides `AIMessage`, `HumanMessage`, and `ToolMessage` (from `langchain_core.messages`). `logfire` provides the `@logfire.instrument("node:...", extract_args=False)` decorator used on every node. `pydantic` (already installed) is used for the `QueryRewriteOutput` structured-output model in the rewrite node.

If you are not using `uv`, install with pip:

```bash
pip install langgraph langchain-core logfire
```

> **Note:** The node files import `logfire` at module top level. If `logfire` is not installed or not configured, the `@logfire.instrument` decorator still works (it is a no-op when Logfire is not initialized), but the import must succeed. Ensure `logfire` is in your dependencies.

---

## 4. Directory Structure to Create

This phase creates the `src/services/agents/nodes/` subpackage:

```
src/
└── services/
    └── agents/
        ├── __init__.py              # (PHASE 13) package marker
        ├── state.py                 # (PHASE 13) AgentState
        ├── context.py               # (PHASE 13) Context
        ├── models.py                # (PHASE 13) Pydantic models
        ├── config.py                # (PHASE 13) GraphConfig
        ├── prompts.py               # (PHASE 13) prompt templates
        └── nodes/                   # THIS PHASE
            ├── __init__.py          # Re-exports all node functions
            ├── utils.py             # Shared helper functions
            ├── guardrail_node.py    # Input guardrail node + continue_after_guardrail
            ├── out_of_scope_node.py # Out-of-scope response node
            ├── retrieve_node.py     # Retrieval initiation node
            ├── grade_documents_node.py  # Document relevance grading node
            ├── rewrite_query_node.py    # Query rewriting node
            ├── generate_answer_node.py  # Answer generation node
            └── output_guardrail_node.py # Output grounding/content guardrail node
```

> All eight files plus `utils.py` and `__init__.py` are created in this phase. The `agentic_rag.py`, `factory.py`, `tools.py`, `supervisor_agent.py`, and `summarizer_agent.py` files are built in PHASE 15.

---

## 5. Step-by-Step Implementation

### Step 1 — Create the nodes package marker

**Full file path:** `src/services/agents/nodes/__init__.py`

```python
from .generate_answer_node import ainvoke_generate_answer_step
from .grade_documents_node import ainvoke_grade_documents_step
from .guardrail_node import ainvoke_guardrail_step, continue_after_guardrail
from .out_of_scope_node import ainvoke_out_of_scope_step
from .output_guardrail_node import ainvoke_output_guardrail_step
from .retrieve_node import ainvoke_retrieve_step
from .rewrite_query_node import ainvoke_rewrite_query_step

__all__ = [
    "ainvoke_guardrail_step",
    "continue_after_guardrail",
    "ainvoke_out_of_scope_step",
    "ainvoke_retrieve_step",
    "ainvoke_grade_documents_step",
    "ainvoke_rewrite_query_step",
    "ainvoke_generate_answer_step",
    "ainvoke_output_guardrail_step",
]
```

**Explanation:** This `__init__.py` re-exports every node function so that PHASE 15 can import them with a single `from .nodes import (...)` statement. Note it does **not** re-export `utils.py` helpers — those are imported directly from `.utils` where needed. The `__all__` list documents the public surface of the package.

---

### Step 2 — Create the shared node utilities

**Full file path:** `src/services/agents/nodes/utils.py`

```python
import logging
import re
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..models import ReasoningStep, SourceItem, ToolArtefact

logger = logging.getLogger(__name__)


def extract_sources_from_tool_messages(messages: List) -> List[SourceItem]:
    """Extract sources from tool messages in conversation.

    Parses the string representation of LangChain Documents serialized
    by LangGraph's ToolNode from the retrieve_papers tool output.

    :param messages: List of messages from graph state
    :returns: List of SourceItem objects
    """
    sources = []
    seen_arxiv_ids: set = set()

    for msg in messages:
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "retrieve_papers":
            content = msg.content
            if not content:
                continue

            # Parse document metadata from ToolNode's string representation of list[Document]
            # Format: [..., metadata={'arxiv_id': 'X', 'title': 'Y', 'score': Z, 'source': 'URL', ...}, ...]
            arxiv_ids = re.findall(r"'arxiv_id':\s*'([^']+)'", content)
            titles = re.findall(r"'title':\s*'([^']*)'", content)
            source_urls = re.findall(r"'source':\s*'(https?://[^']+)'", content)
            scores = re.findall(r"'score':\s*([\d.]+)", content)
            authors_matches = re.findall(r"'authors':\s*'([^']*)'", content)

            for i, arxiv_id in enumerate(arxiv_ids):
                if arxiv_id in seen_arxiv_ids:
                    continue
                seen_arxiv_ids.add(arxiv_id)

                url = source_urls[i] if i < len(source_urls) else f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                title = titles[i] if i < len(titles) else ""
                relevance_score = float(scores[i]) if i < len(scores) else 0.0
                authors_str = authors_matches[i] if i < len(authors_matches) else ""
                authors = [a.strip() for a in authors_str.split(",") if a.strip()] if authors_str else []

                sources.append(SourceItem(
                    arxiv_id=arxiv_id,
                    title=title,
                    authors=authors,
                    url=url,
                    relevance_score=relevance_score,
                ))

    logger.debug(f"Extracted {len(sources)} sources from tool messages")
    return sources


def extract_tool_artefacts(messages: List) -> List[ToolArtefact]:
    """Extract tool artifacts from messages.

    :param messages: List of messages from graph state
    :returns: List of ToolArtefact objects
    """
    artefacts = []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            artefact = ToolArtefact(
                tool_name=getattr(msg, "name", "unknown"),
                tool_call_id=getattr(msg, "tool_call_id", ""),
                content=msg.content,
                metadata={},
            )
            artefacts.append(artefact)

    return artefacts


def create_reasoning_step(
    step_name: str,
    description: str,
    metadata: Optional[Dict] = None,
) -> ReasoningStep:
    """Create a reasoning step record.

    :param step_name: Name of the step/node
    :param description: Human-readable description
    :param metadata: Additional metadata
    :returns: ReasoningStep object
    """
    return ReasoningStep(
        step_name=step_name,
        description=description,
        metadata=metadata or {},
    )


def filter_messages(messages: List) -> List[AIMessage | HumanMessage]:
    """Filter messages to include only HumanMessage and AIMessage types.

    Excludes tool messages and other internal message types.

    :param messages: List of messages to filter
    :returns: Filtered list of messages
    """
    return [msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage))]


def get_latest_query(messages: List) -> str:
    """Get the latest user query from messages.

    :param messages: List of messages
    :returns: Latest query text
    :raises ValueError: If no user query found
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content

    raise ValueError("No user query found in messages")


def get_latest_context(messages: List) -> str:
    """Get the latest context from tool messages.

    :param messages: List of messages
    :returns: Latest context text or empty string
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return msg.content if hasattr(msg, "content") else ""

    return ""
```

**Explanation:** These helpers are the shared building blocks used by the nodes. The two most important are `get_latest_query` (walks messages **in reverse** to find the most recent `HumanMessage` — this is how the graph tracks the current query after rewrites) and `get_latest_context` (walks messages in reverse to find the most recent `ToolMessage`, which contains the retrieved document text). `extract_sources_from_tool_messages` parses the string representation of LangChain `Document` objects serialized by LangGraph's `ToolNode` using regex — it deduplicates by `arxiv_id` and builds `SourceItem` objects. `extract_tool_artefacts`, `create_reasoning_step`, and `filter_messages` are supporting helpers retained for completeness (the current nodes primarily use `get_latest_query`, `get_latest_context`, and `extract_sources_from_tool_messages`).

---

### Step 3 — Create the input guardrail node

**Full file path:** `src/services/agents/nodes/guardrail_node.py`

```python
import logging
import time
from typing import Dict, Literal

import logfire
from langgraph.runtime import Runtime

from ..context import Context
from ..models import GuardrailScoring
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)


def continue_after_guardrail(state: AgentState, runtime: Runtime[Context]) -> Literal["continue", "out_of_scope"]:
    """Determine whether to continue or reject based on guardrail results.

    :param state: Current agent state with guardrail results
    :param runtime: Runtime context containing guardrail threshold
    :returns: "continue" if score >= threshold, "out_of_scope" otherwise
    """
    guardrail_result = state.get("guardrail_result")
    if not guardrail_result:
        logger.warning("No guardrail result found, defaulting to continue")
        return "continue"

    score = guardrail_result.score
    threshold = runtime.context.guardrail_threshold

    logger.info(f"Guardrail score: {score}, threshold: {threshold}")
    return "continue" if score >= threshold else "out_of_scope"


@logfire.instrument("node:guardrail", extract_args=False)
async def ainvoke_guardrail_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, GuardrailScoring]:
    """Asynchronously invoke the guardrail validation step.

    Uses AWS Bedrock Guardrails when guardrails_service is configured.
    Falls back to fail-open (allow all) when Bedrock is not configured.
    Guardrail result is mapped to GuardrailScoring for state compatibility:
      - allowed → score=100
      - blocked → score=0

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Dictionary with guardrail_result
    """
    logger.info("NODE: guardrail_validation")
    start_time = time.time()

    query = get_latest_query(state["messages"])
    logger.debug(f"Evaluating query: {query[:100]}...")

    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="guardrail_validation",
                input_data={
                    "query": query,
                    "threshold": runtime.context.guardrail_threshold,
                    "guardrails_provider": "bedrock" if runtime.context.guardrails_service else "none",
                },
                metadata={"node": "guardrail"},
            )
        except Exception as e:
            logger.warning(f"Failed to create Langfuse span for guardrail: {e}")

    try:
        if runtime.context.guardrails_service:
            result = await runtime.context.guardrails_service.check_input(query)
            score = 100 if result.allowed else 0
            reason = result.reason
            sanitized_query = result.outputs[0] if result.outputs and result.allowed else None
            if sanitized_query:
                logger.info(f"Bedrock guardrail: PII anonymized, using sanitized query")
            logger.info(f"Bedrock guardrail: action={result.action}, allowed={result.allowed}, reason={reason}")
        else:
            # No guardrails configured — fail-open
            score = 100
            reason = "No guardrail service configured — passing through"
            sanitized_query = None
            logger.debug(reason)

        response = GuardrailScoring(score=score, reason=reason)

        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.end_span(
                span,
                output={
                    "score": response.score,
                    "reason": response.reason,
                    "decision": "continue" if response.score >= runtime.context.guardrail_threshold else "out_of_scope",
                },
                metadata={"execution_time_ms": execution_time},
            )

    except Exception as e:
        logger.error(f"Guardrail validation failed: {e}, falling back to allow")
        response = GuardrailScoring(
            score=100,
            reason=f"Guardrail check failed (fail-open): {str(e)}",
        )
        sanitized_query = None
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.update_span(
                span,
                output={"score": response.score, "reason": response.reason, "error": str(e)},
                metadata={"execution_time_ms": execution_time, "fallback": True},
                level="WARNING",
            )
            runtime.context.langfuse_tracer.end_span(span)

    result = {"guardrail_result": response}
    if sanitized_query:
        result["sanitized_query"] = sanitized_query
    return result
```

**Explanation:** This node has two exported functions. `continue_after_guardrail` is the **conditional-edge function** (used in PHASE 15) that reads `state["guardrail_result"]` and compares its `score` against `runtime.context.guardrail_threshold`, returning the literal `"continue"` or `"out_of_scope"`. `ainvoke_guardrail_step` is the node itself: it calls `guardrails_service.check_input(query)` when a service is configured, mapping `allowed → score=100` and `blocked → score=0`. When no service is configured it **fails open** with `score=100`. When Bedrock PII-anonymizes the input, `result.outputs[0]` becomes the sanitized query, which is returned as `sanitized_query` in the state update. The whole check is wrapped in a try/except that fails open on any error. Note the return type is `Dict[str, GuardrailScoring]` but the function may also add a `sanitized_query` key — this is a known type looseness in the reference; keep it as-is.

---

### Step 4 — Create the out-of-scope node

**Full file path:** `src/services/agents/nodes/out_of_scope_node.py`

```python
import logging
from typing import Dict, List

import logfire

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)


@logfire.instrument("node:out_of_scope", extract_args=False)
async def ainvoke_out_of_scope_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, List[AIMessage]]:
    """Handle out-of-scope queries with a helpful message.

    This node responds to queries that are outside the domain of
    CS/AI/ML research papers with a polite, informative message.

    :param state: Current agent state
    :param runtime: Runtime context (not used in this node)
    :returns: Dictionary with messages containing the out-of-scope response
    """
    logger.info("NODE: out_of_scope")

    question = get_latest_query(state["messages"])

    # Generate helpful response message
    response_text = (
        "I apologize, but I can only help with questions about academic research papers "
        "in Computer Science, Artificial Intelligence, and Machine Learning from arXiv.\n\n"
        f"Your question: '{question}'\n\n"
        "This appears to be outside my domain of expertise. For questions like this, you might want to try:\n"
        "- General-purpose AI assistants for broad knowledge questions\n"
        "- Domain-specific resources for topics outside CS/AI/ML\n"
        "- Technical documentation if asking about specific software/tools\n\n"
        "If you have a question about AI/ML research papers, I'd be happy to help!"
    )

    logger.info("Responding with out-of-scope message")

    return {"messages": [AIMessage(content=response_text)]}
```

**Explanation:** This node returns a **hardcoded** helpful message (it does not call the LLM). It reads the latest query via `get_latest_query` to echo it back in the response, then returns a single `AIMessage` in the `messages` key. Because `messages` uses the `add_messages` reducer (PHASE 13), this `AIMessage` is appended to the conversation. The `runtime` parameter is unused in this node but must remain in the signature because LangGraph passes it to every node. Note: the `DIRECT_RESPONSE_PROMPT` in `prompts.py` is **not** used here — the reference returns a hardcoded string.

---

### Step 5 — Create the retrieve node

**Full file path:** `src/services/agents/nodes/retrieve_node.py`

```python
import logging
import time
from typing import Dict, Union

import logfire
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)


@logfire.instrument("node:retrieve", extract_args=False)
async def ainvoke_retrieve_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, Union[int, str, list]]:
    """Initiate retrieval or return fallback if max attempts reached.

    This node creates a tool call to retrieve documents, or returns a fallback
    message if the maximum number of retrieval attempts has been reached.

    :param state: Current agent state
    :param runtime: Runtime context containing max_retrieval_attempts
    :returns: Dictionary with updated state (retrieval_attempts, messages, original_query)
    """
    logger.info("NODE: retrieve")
    start_time = time.time()

    messages = state["messages"]
    question = state.get("sanitized_query") or get_latest_query(messages)
    current_attempts = state.get("retrieval_attempts", 0)

    # Get max attempts from context
    max_attempts = runtime.context.max_retrieval_attempts

    # Store original query if not set
    updates = {}
    if state.get("original_query") is None:
        updates["original_query"] = question
        logger.debug(f"Stored original query: {question[:100]}...")

    # Create span for retrieval initiation
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="document_retrieval_initiation",
                input_data={
                    "query": question,
                    "attempt": current_attempts + 1,
                    "max_attempts": max_attempts,
                },
                metadata={
                    "node": "retrieve",
                    "top_k": runtime.context.top_k,
                },
            )
            logger.debug(f"Created Langfuse span for retrieval attempt {current_attempts + 1}")
        except Exception as e:
            logger.warning(f"Failed to create span for retrieve node: {e}")

    # Check if max attempts reached
    if current_attempts >= max_attempts:
        logger.warning(f"Max retrieval attempts ({max_attempts}) reached")
        fallback_msg = (
            f"I apologize, but I couldn't find relevant research papers after {max_attempts} attempts.\n"
            "This may be because:\n"
            "1. No papers in the database contain relevant information\n"
            "2. The query terms don't match the indexed content\n\n"
            "Please try rephrasing your question with more specific technical terms."
        )

        # Update span with max attempts reached
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"status": "max_attempts_reached", "fallback": True},
                metadata={"execution_time_ms": execution_time},
            )

        return {**updates, "messages": [AIMessage(content=fallback_msg)]}

    # Increment retrieval attempts
    new_attempt_count = current_attempts + 1
    updates["retrieval_attempts"] = new_attempt_count
    logger.info(f"Retrieval attempt {new_attempt_count}/{max_attempts}")

    # Create tool call for retrieval
    updates["messages"] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": f"retrieve_{new_attempt_count}",
                    "name": "retrieve_papers",
                    "args": {"query": question},
                }
            ],
        )
    ]

    logger.debug(f"Created tool call for query: {question[:100]}...")

    # Update span with successful tool call creation
    if span:
        execution_time = (time.time() - start_time) * 1000
        runtime.context.langfuse_tracer.end_span(
            span,
            output={
                "status": "tool_call_created",
                "query": question,
                "attempt": new_attempt_count,
            },
            metadata={"execution_time_ms": execution_time},
        )

    return updates
```

**Explanation:** This node does **not** perform the search itself — it creates a **tool call** (`AIMessage` with `tool_calls=[{"name": "retrieve_papers", "args": {"query": question}}]`) that LangGraph's `ToolNode` (wired in PHASE 15) will execute. The query used is `state.get("sanitized_query") or get_latest_query(messages)` — preferring the Bedrock-sanitized query when present. It increments `retrieval_attempts` on each call. When `current_attempts >= max_attempts` (from `runtime.context.max_retrieval_attempts`, default 2), it returns a fallback `AIMessage` instead of a tool call — this is the **retrieval loop cap**. It also stores `original_query` the first time it runs. The `tools_condition` edge (PHASE 15) routes to `tool_retrieve` when a tool call is present, or to `END` when the fallback message is returned.

---

### Step 6 — Create the grade-documents node

**Full file path:** `src/services/agents/nodes/grade_documents_node.py`

```python
import logging
import time
from typing import Dict

import logfire

from langgraph.runtime import Runtime

from ..context import Context
from ..models import GradingResult
from ..prompts import GRADE_DOCUMENTS_PROMPT
from ..state import AgentState
from .utils import extract_sources_from_tool_messages, get_latest_context, get_latest_query

logger = logging.getLogger(__name__)


@logfire.instrument("node:grade_documents", extract_args=False)
async def ainvoke_grade_documents_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, str | list]:
    """Grade retrieved documents for relevance using LLM.

    This function uses an LLM to evaluate whether the retrieved documents
    are relevant to the user's query and decides whether to generate an
    answer or rewrite the query for better results.

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Dictionary with routing_decision and grading_results
    """
    logger.info("NODE: grade_documents")
    start_time = time.time()

    # Get query and context
    question = state.get("sanitized_query") or get_latest_query(state["messages"])
    context = get_latest_context(state["messages"])

    # Extract document chunks from context for logging
    chunks_preview = []
    if context:
        # Context is a string containing all documents concatenated
        # Let's show a preview of what was retrieved
        context_preview = context[:500] + "..." if len(context) > 500 else context
        chunks_preview = [{"text_preview": context_preview, "length": len(context)}]

    # Create span for document grading
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="document_grading",
                input_data={
                    "query": question,
                    "context_length": len(context) if context else 0,
                    "has_context": context is not None,
                    "chunks_received": chunks_preview,
                },
                metadata={
                    "node": "grade_documents",
                    "model": runtime.context.model_name,
                },
            )
            logger.debug("Created Langfuse span for document grading")
        except Exception as e:
            logger.warning(f"Failed to create span for grade_documents node: {e}")

    if not context:
        logger.warning("No context found, routing to rewrite_query")

        # Update span with no context result
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"routing_decision": "rewrite_query", "reason": "no_context"},
                metadata={"execution_time_ms": execution_time},
            )

        return {"routing_decision": "rewrite_query", "grading_results": []}

    logger.debug(f"Grading context of length {len(context)} characters")

    # Use LLM to grade document relevance (plain text — avoids structured output failures on small models)
    try:
        grading_prompt = GRADE_DOCUMENTS_PROMPT.format(
            context=context,
            question=question,
        )

        llm = runtime.context.llm_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=0.0,
        )

        logger.info("Invoking LLM for document grading (plain text)")
        grading_response = await llm.ainvoke(grading_prompt)
        response_text = grading_response.content if hasattr(grading_response, "content") else str(grading_response)

        # Look for yes/no in first 300 chars; treat ambiguous output as relevant (fail open)
        snippet = response_text.lower()[:300]
        if '"binary_score": "no"' in snippet or "'binary_score': 'no'" in snippet:
            is_relevant = False
        elif "binary_score" in snippet and "no" in snippet and "yes" not in snippet:
            is_relevant = False
        else:
            is_relevant = True

        score = 1.0 if is_relevant else 0.0
        logger.info(f"LLM grading result: is_relevant={is_relevant}, response_snippet={snippet[:100]}")

        grading_result = GradingResult(
            document_id="retrieved_docs",
            is_relevant=is_relevant,
            score=score,
            reasoning=response_text[:500],
        )

    except Exception as e:
        logger.error(f"LLM grading failed: {e}, failing open")
        # Fail open: if we retrieved any context, attempt to generate an answer
        is_relevant = bool(context.strip())
        grading_result = GradingResult(
            document_id="retrieved_docs",
            is_relevant=is_relevant,
            score=1.0 if is_relevant else 0.0,
            reasoning=f"Fallback (LLM error): {'proceeding with context' if is_relevant else 'no context available'}",
        )

    # Determine routing
    route = "generate_answer" if is_relevant else "rewrite_query"

    logger.info(f"Grading result: {'relevant' if is_relevant else 'not relevant'}, routing to: {route}")

    # Update span with grading result
    if span:
        execution_time = (time.time() - start_time) * 1000
        runtime.context.langfuse_tracer.end_span(
            span,
            output={
                "routing_decision": route,
                "is_relevant": is_relevant,
                "score": grading_result.score,
                "reasoning": grading_result.reasoning,
            },
            metadata={
                "execution_time_ms": execution_time,
                "context_length": len(context),
            },
        )

    relevant_sources = extract_sources_from_tool_messages(state["messages"]) if is_relevant else []

    return {
        "routing_decision": route,
        "grading_results": [grading_result],
        "relevant_sources": relevant_sources,
    }
```

**Explanation:** This node grades the retrieved context for relevance. It gets the context via `get_latest_context(state["messages"])`. If there is **no context**, it routes to `rewrite_query` immediately. Otherwise it invokes the LLM with `GRADE_DOCUMENTS_PROMPT` using **plain text** (not structured output — this avoids failures on small models) and parses the response by scanning the first 300 characters for a `"binary_score": "no"` pattern. Ambiguous output is treated as **relevant (fail open)**. On any LLM error it also fails open (`is_relevant = bool(context.strip())`). It sets `routing_decision` to `"generate_answer"` or `"rewrite_query"`, records a `GradingResult`, and — when relevant — extracts `relevant_sources` from the tool messages. The `routing_decision` field drives the conditional edge in PHASE 15.

---

### Step 7 — Create the rewrite-query node

**Full file path:** `src/services/agents/nodes/rewrite_query_node.py`

```python
import logging
import time
from typing import Dict, List

import logfire

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from ..context import Context
from ..prompts import REWRITE_PROMPT
from ..state import AgentState

logger = logging.getLogger(__name__)


class QueryRewriteOutput(BaseModel):
    """Structured output for query rewriting."""

    rewritten_query: str = Field(
        description="The improved query optimized for document retrieval"
    )
    reasoning: str = Field(
        description="Brief explanation of how the query was improved"
    )


@logfire.instrument("node:rewrite_query", extract_args=False)
async def ainvoke_rewrite_query_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, str | List]:
    """Rewrite the original query for better document retrieval using LLM.

    This node uses an LLM to intelligently rewrite the user's query
    to improve the chances of finding relevant documents.

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Dictionary with rewritten_query and updated messages
    """
    logger.info("NODE: rewrite_query")
    start_time = time.time()

    # Get original query
    original_question = state.get("original_query") or state["messages"][0].content
    current_attempt = state.get("retrieval_attempts", 0)

    logger.debug(f"Rewriting query using LLM: {original_question[:100]}...")

    # Create span for query rewriting
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="query_rewriting",
                input_data={
                    "original_query": original_question,
                    "attempt": current_attempt,
                },
                metadata={
                    "node": "rewrite_query",
                    "strategy": "llm_based_expansion",
                    "model": runtime.context.model_name,
                },
            )
            logger.debug("Created Langfuse span for query rewriting")
        except Exception as e:
            logger.warning(f"Failed to create span for rewrite_query node: {e}")

    # Use LLM to rewrite the query intelligently
    try:
        # Create structured LLM for query rewriting
        llm = runtime.context.llm_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=0.3,  # Lower temperature for more focused rewriting
        )
        structured_llm = llm.with_structured_output(QueryRewriteOutput)

        # Format prompt with original question
        prompt = REWRITE_PROMPT.format(question=original_question)

        logger.debug(f"Invoking LLM for query rewriting (model: {runtime.context.model_name})")
        llm_start = time.time()

        # Get rewritten query from LLM
        result: QueryRewriteOutput = await structured_llm.ainvoke(prompt)

        # Validate LLM output
        if not result or not result.rewritten_query:
            raise ValueError("LLM failed to return valid structured output for query rewriting")

        rewritten_query = result.rewritten_query.strip()
        if not rewritten_query:
            raise ValueError("LLM returned empty rewritten query")

        reasoning = result.reasoning

        llm_duration = time.time() - llm_start
        logger.info(
            f"Query rewritten in {llm_duration:.2f}s: "
            f"'{original_question[:50]}...' -> '{rewritten_query[:50]}...'"
        )
        logger.debug(f"Rewriting reasoning: {reasoning}")

    except Exception as e:
        logger.error(f"Failed to rewrite query using LLM: {e}")
        logger.warning("Falling back to simple keyword expansion")
        # Fallback to simple expansion if LLM fails
        rewritten_query = f"{original_question} research paper arxiv machine learning"
        reasoning = "Fallback: Simple keyword expansion due to LLM error"

    # Update span with rewriting result
    if span:
        execution_time = (time.time() - start_time) * 1000
        runtime.context.langfuse_tracer.end_span(
            span,
            output={
                "rewritten_query": rewritten_query,
                "reasoning": reasoning,
                "original_query": original_question,
            },
            metadata={
                "execution_time_ms": execution_time,
                "original_length": len(original_question),
                "rewritten_length": len(rewritten_query),
                "llm_duration_seconds": llm_duration if 'llm_duration' in locals() else None,
            },
        )

    return {
        "messages": [HumanMessage(content=rewritten_query)],
        "rewritten_query": rewritten_query,
    }
```

**Explanation:** This node rewrites the query using the LLM with **structured output** (`QueryRewriteOutput` via `llm.with_structured_output`). It uses the `original_query` stored by the retrieve node (falling back to `state["messages"][0].content`). It invokes the LLM with `REWRITE_PROMPT` at `temperature=0.3`. On any failure it **falls back to simple keyword expansion** (`f"{original_question} research paper arxiv machine learning"`). It returns a new `HumanMessage` containing the rewritten query (so `get_latest_query` in the next retrieve pass picks it up) plus the `rewritten_query` field. This node is what makes the retrieval loop "agentic" — each loop iteration uses a progressively better query.

---

### Step 8 — Create the generate-answer node

**Full file path:** `src/services/agents/nodes/generate_answer_node.py`

```python
import logging
import time
from typing import Dict, List

import logfire

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..prompts import GENERATE_ANSWER_PROMPT
from ..state import AgentState
from .utils import get_latest_context, get_latest_query

logger = logging.getLogger(__name__)


@logfire.instrument("node:generate_answer", extract_args=False)
async def ainvoke_generate_answer_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, List[AIMessage]]:
    """Generate final answer using retrieved documents.

    This node generates a comprehensive answer to the
    user's question based on the retrieved context using an LLM.

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Dictionary with messages containing the generated answer
    """
    logger.info("NODE: generate_answer")
    start_time = time.time()

    # Get question and context
    question = state.get("sanitized_query") or get_latest_query(state["messages"])
    context = get_latest_context(state["messages"])

    # Count sources from relevant_sources
    sources_count = len(state.get("relevant_sources", []))

    if not context:
        context = "No relevant documents found."
        logger.warning("No context available for answer generation")

    logger.debug(f"Generating answer for query: {question[:100]}...")
    logger.debug(f"Using context of length: {len(context)} characters")

    # Extract document chunks preview for logging
    chunks_preview = []
    if context:
        context_preview = context[:1000] + "..." if len(context) > 1000 else context
        chunks_preview = [{"text_preview": context_preview, "length": len(context)}]

    # Create span for answer generation
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="answer_generation",
                input_data={
                    "query": question,
                    "context_length": len(context),
                    "sources_count": sources_count,
                    "chunks_used": chunks_preview,
                },
                metadata={
                    "node": "generate_answer",
                    "model": runtime.context.model_name,
                    "temperature": runtime.context.temperature,
                },
            )
            logger.debug("Created Langfuse span for answer generation")
        except Exception as e:
            logger.warning(f"Failed to create span for generate_answer node: {e}")

    try:
        # Create answer generation prompt from template
        answer_prompt = GENERATE_ANSWER_PROMPT.format(
            context=context,
            question=question,
        )

        # Get LLM from runtime context
        llm = runtime.context.llm_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=runtime.context.temperature,
        )

        # Invoke LLM for answer generation
        logger.info("Invoking LLM for answer generation")
        response = await llm.ainvoke(answer_prompt)

        # Extract content from response
        answer = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"Generated answer of length: {len(answer)} characters")

        # Update span with successful result
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.end_span(
                span,
                output={
                    "answer_length": len(answer),
                    "sources_used": sources_count,
                },
                metadata={
                    "execution_time_ms": execution_time,
                    "context_length": len(context),
                },
            )

    except Exception as e:
        logger.error(f"LLM answer generation failed: {e}, falling back to error message")

        # Fallback to error message if LLM fails
        answer = f"I apologize, but I encountered an error while generating the answer: {str(e)}\n\nPlease try again or rephrase your question."

        # Update span with error
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.update_span(
                span,
                output={"error": str(e), "fallback": True},
                metadata={"execution_time_ms": execution_time},
                level="ERROR",
            )
            runtime.context.langfuse_tracer.end_span(span)

    return {"messages": [AIMessage(content=answer)]}
```

**Explanation:** This node generates the final answer. It gets the question (preferring `sanitized_query`) and the retrieved context via `get_latest_context`. If there is no context it substitutes `"No relevant documents found."`. It invokes the LLM with `GENERATE_ANSWER_PROMPT` at `runtime.context.temperature`. On any LLM error it returns a polite error message as the answer. It returns a single `AIMessage` with the answer. This is the terminal content node — the `output_guardrail` node runs after it in PHASE 15 to verify grounding.

---

### Step 9 — `output_guardrail_node.py`

This node runs **after** the answer is generated. It verifies that the generated answer is actually grounded in the retrieved source documents by calling the Bedrock Guardrails **OUTPUT** side (`check_output`). If the answer is not sufficiently grounded, it replaces the answer with a polite `GROUNDING_FAIL_MESSAGE` and records the filter reason in state. If no guardrails service is configured, it is a pure pass-through.

Create [`output_guardrail_node.py`](Agentic-RAG-project-agentops/src/services/agents/nodes/output_guardrail_node.py):

```python
import logging
import time
from typing import Dict, List

import logfire
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..state import AgentState

logger = logging.getLogger(__name__)

GROUNDING_FAIL_MESSAGE = (
    "I'm sorry, but I cannot provide an answer that is sufficiently supported by the "
    "retrieved research papers. The documents retrieved may not contain enough information "
    "to reliably answer your question. Please try rephrasing your query or asking about "
    "a different aspect of the topic."
)


@logfire.instrument("node:output_guardrail", extract_args=False)
async def ainvoke_output_guardrail_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, List[AIMessage]]:
    """Verify the generated answer is grounded in retrieved source documents.

    Uses the Bedrock Guardrails OUTPUT side to check whether the answer is
    sufficiently supported by the retrieved sources. If it is not, the answer is
    replaced with a polite failure message.
    """
    logger.info("NODE: output_guardrail")

    if not runtime.context.guardrails_service:
        logger.debug("No guardrails service — output_guardrail is a pass-through")
        return {}

    # Extract the generated answer from the last message
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_msg = messages[-1]
    answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    if not answer or not answer.strip():
        return {}

    # Get the original query for grounding evaluation
    from .utils import get_latest_query
    query = state.get("sanitized_query") or get_latest_query(state.get("messages", []))

    # Collect source document texts for grounding check
    source_texts: List[str] = []
    from .utils import get_latest_context
    context = get_latest_context(state.get("messages", []))
    if context:
        source_texts = [chunk.strip() for chunk in context.split("\n\n") if chunk.strip()]

    # Fall back to titles if no context available
    if not source_texts:
        for source in state.get("relevant_sources", []):
            title = getattr(source, "title", "")
            abstract = getattr(source, "abstract", "")
            arxiv_id = getattr(source, "arxiv_id", "")
            text = abstract or title
            if text:
                source_texts.append(f"{text} (arXiv:{arxiv_id})")

    # If no source texts available, skip grounding check
    if not source_texts:
        logger.debug("No source texts available — skipping output grounding check")
        return {}

    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="output_guardrail_check",
                input_data={
                    "query": query,
                    "answer_length": len(answer),
                    "source_count": len(source_texts),
                },
                metadata={"node": "output_guardrail"},
            )
        except Exception as e:
            logger.warning(f"Failed to create Langfuse span for output guardrail: {e}")

    start_time = time.time()
    try:
        result = await runtime.context.guardrails_service.check_output(answer, source_texts, query=query)

        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.end_span(
                span,
                output={
                    "action": result.action,
                    "allowed": result.allowed,
                    "reason": result.reason,
                },
                metadata={"execution_time_ms": execution_time},
            )

        if not result.allowed:
            logger.warning(f"Output guardrail blocked answer: {result.reason}")
            return {
                "messages": [AIMessage(content=GROUNDING_FAIL_MESSAGE)],
                "output_guardrail_filter": result.reason,
            }

        logger.info(f"Output guardrail passed: {result.reason}")
        return {"output_guardrail_filter": result.reason}

    except Exception as e:
        logger.error(f"Output guardrail check failed: {e} — passing answer through")
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.update_span(
                span,
                output={"error": str(e), "fallback": True},
                metadata={"execution_time_ms": execution_time},
                level="WARNING",
            )
            runtime.context.langfuse_tracer.end_span(span)
        return {}
```

**Explanation:** This node first checks whether a `guardrails_service` exists in the runtime context — if not, it returns `{}` (pass-through). It extracts the generated answer from the last message, determines the query (preferring `sanitized_query`), and collects source texts from the latest tool context (falling back to `relevant_sources` titles/abstracts). If there are no source texts, it skips the grounding check. It creates a Langfuse span, calls `check_output(answer, source_texts, query=query)`, and:
- If **not allowed**: replaces the answer with `GROUNDING_FAIL_MESSAGE` and sets `output_guardrail_filter` to the reason.
- If **allowed**: sets `output_guardrail_filter` to the reason.
- On **error**: logs and returns `{}` (fail-open, answer passes through).

---

## 6. Configuration

The agent nodes require no additional environment variables beyond those already configured in PHASE 2 and PHASE 5. The relevant settings are consumed through the `Context` dataclass built in PHASE 13:

| Setting | Source | Used By |
| --- | --- | --- |
| `guardrails_service` | `make_bedrock_guardrails_service(settings)` | `guardrail_node`, `output_guardrail_node` |
| `llm_client` | provider factory (bedrock / openai) | all LLM nodes |
| `embeddings_client` | embeddings factory | `retrieve_node` (via tool) |
| `opensearch_client` | opensearch factory | `retrieve_node` (via tool) |
| `langfuse_tracer` | langfuse factory | all nodes (tracing) |
| `temperature` | `GraphConfig.temperature` (default `0.0`) | `generate_answer_node` |
| `max_retrieval_attempts` | `GraphConfig.max_retrieval_attempts` (default `2`) | `retrieve_node` |
| `guardrail_threshold` | `GraphConfig.guardrail_threshold` (default `40`) | guardrail services |

The Bedrock Guardrails service is created once in `main.py` (PHASE 16) and passed into the `Context` for each request. When no guardrails service is available (e.g., local development without AWS), every node degrades gracefully to fail-open behavior.

---

## 7. Verification

Verify the nodes work correctly with the following checks:

1. **Import check** — confirm all nodes import cleanly:
   ```bash
   python -c "from src.services.agents.nodes import ainvoke_guardrail_step, ainvoke_out_of_scope_step, ainvoke_retrieve_step, ainvoke_grade_documents_step, ainvoke_rewrite_query_step, ainvoke_generate_answer_step, ainvoke_output_guardrail_step, continue_after_guardrail"
   ```

2. **Unit tests** — run the existing node test suite:
   ```bash
   pytest tests/unit/services/agents/test_nodes.py -v
   ```

3. **Guardrail fail-open** — with no guardrails service configured, confirm `ainvoke_guardrail_step` returns a score of `100` (allowed) and `ainvoke_output_guardrail_step` returns `{}` (pass-through).

4. **Retrieval loop cap** — confirm the `retrieve` node stops after `max_retrieval_attempts` and returns a fallback `AIMessage` when the loop is exhausted.

5. **Rewrite fallback** — confirm `ainvoke_rewrite_query_step` falls back to `f"{original_question} research paper arxiv machine learning"` when the LLM call fails.

6. **Output grounding** — with a guardrails service configured, confirm a hallucinated answer is replaced with `GROUNDING_FAIL_MESSAGE` and `output_guardrail_filter` is populated.

---

## 8. Common Pitfalls

- **Forgetting the `Runtime[Context]` second parameter** — every node signature must be `async def node(state: AgentState, runtime: Runtime[Context])`. Omitting `runtime` breaks LangGraph's context injection.
- **Mutable state keys** — only `messages` uses the `add_messages` reducer. All other keys (`retrieval_attempts`, `original_query`, `rewritten_query`, `sanitized_query`, `routing_decision`, `relevant_sources`, `output_guardrail_filter`) are plain overwrites; do not append to them across steps.
- **Fail-open vs fail-closed** — the guardrail nodes are intentionally **fail-open** (score `100` / pass-through on error) so a guardrails outage never blocks the whole pipeline. Do not change this to fail-closed without a deliberate decision.
- **`get_latest_query` / `get_latest_context`** — these walk the message list in reverse to find the most recent `HumanMessage` / `ToolMessage`. Ensure the message ordering is preserved (the `add_messages` reducer appends in order).
- **Structured output availability** — `rewrite_query_node` uses `llm.with_structured_output(QueryRewriteOutput)`. This requires a provider that supports structured output (OpenAI / Bedrock with tool-calling). Always keep the plain-text fallback.
- **Not decorating with `@logfire.instrument`** — every node should carry the `@logfire.instrument("node:<name>", extract_args=False)` decorator for consistent observability.

---

## 9. Definition of Done

- [ ] All 8 node files exist in [`src/services/agents/nodes/`](Agentic-RAG-project-agentops/src/services/agents/nodes/): `guardrail_node.py`, `out_of_scope_node.py`, `retrieve_node.py`, `grade_documents_node.py`, `rewrite_query_node.py`, `generate_answer_node.py`, `output_guardrail_node.py`, and `utils.py`.
- [ ] [`nodes/__init__.py`](Agentic-RAG-project-agentops/src/services/agents/nodes/__init__.py) re-exports all node functions and `continue_after_guardrail`.
- [ ] Every node is decorated with `@logfire.instrument` and follows the `(state, runtime)` signature.
- [ ] Guardrail nodes are fail-open and degrade gracefully without a guardrails service.
- [ ] The retrieval loop is capped at `max_retrieval_attempts`.
- [ ] The rewrite node has a plain-text fallback query.
- [ ] The output guardrail replaces ungrounded answers with `GROUNDING_FAIL_MESSAGE`.
- [ ] `pytest tests/unit/services/agents/test_nodes.py` passes.
- [ ] All nodes are ready to be wired into the LangGraph state graph in **PHASE 15**.