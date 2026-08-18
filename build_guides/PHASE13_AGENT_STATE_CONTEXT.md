# PHASE 13 — Agent State & Context

> **Build guide for the from-scratch reconstruction of the arXiv Paper Curator (Agentic RAG).**
>
> This phase lays the **data foundation for the LangGraph agentic RAG workflow**. It defines the [`AgentState`](../Agentic-RAG-project-agentops/src/services/agents/state.py) `TypedDict` that flows between graph nodes, the [`Context`](../Agentic-RAG-project-agentops/src/services/agents/context.py) dataclass used as LangGraph's `context_schema` for dependency injection, the Pydantic domain models in [`models.py`](../Agentic-RAG-project-agentops/src/services/agents/models.py), the [`GraphConfig`](../Agentic-RAG-project-agentops/src/services/agents/config.py) execution configuration, and the shared prompt templates in [`prompts.py`](../Agentic-RAG-project-agentops/src/services/agents/prompts.py). No graph nodes or graph wiring are built here — those come in PHASE 14 and PHASE 15.

---

## 1. Phase Objective

By the end of this phase you will have:

- The [`AgentState`](../Agentic-RAG-project-agentops/src/services/agents/state.py) `TypedDict` with all fields the agent workflow tracks between nodes (`messages`, `original_query`, `rewritten_query`, `sanitized_query`, `retrieval_attempts`, `guardrail_result`, `output_guardrail_filter`, `routing_decision`, `sources`, `relevant_sources`, `relevant_tool_artefacts`, `grading_results`, `metadata`), using the `add_messages` reducer for `messages`.
- The [`Context`](../Agentic-RAG-project-agentops/src/services/agents/context.py) dataclass that carries immutable runtime dependencies (`llm_client`, `opensearch_client`, `embeddings_client`, `langfuse_tracer`, `guardrails_service`, `trace`) and tuning parameters (`model_name`, `temperature`, `top_k`, `max_retrieval_attempts`, `guardrail_threshold`).
- The Pydantic domain models in [`models.py`](../Agentic-RAG-project-agentops/src/services/agents/models.py): `GuardrailScoring`, `GradeDocuments`, `SourceItem`, `ToolArtefact`, `RoutingDecision`, `GradingResult`, `ReasoningStep`.
- The [`GraphConfig`](../Agentic-RAG-project-agentops/src/services/agents/config.py) Pydantic model that controls graph execution (retrieval attempts, guardrail threshold, model, temperature, top_k, hybrid flag, tracing, metadata, settings).
- The shared prompt templates in [`prompts.py`](../Agentic-RAG-project-agentops/src/services/agents/prompts.py) used by the nodes in PHASE 14.

---

## 2. Prerequisites

- Completion of **PHASE 5 (LLM Providers)** so that an LLM client satisfying `LLMClientProtocol` exists and `src/services/llm_client_protocol.py` is present.
- Completion of **PHASE 6 (Embeddings)** so that `JinaEmbeddingsClient` exists.
- Completion of **PHASE 7 (OpenSearch)** so that `OpenSearchClient` exists.
- Completion of **PHASE 12 (RAG Service)** so that the `LangfuseTracer` (`src/services/langfuse/client.py`) and the `LLMClientProtocol` are available.
- The `BedrockGuardrailsService` from `src/services/bedrock_guardrails/service.py` (built in a prior phase) — imported here for type hints. It may be `None` at runtime.
- Python 3.11+ and the project's virtual environment active.
- `langgraph` and `langchain-core` installed (LangGraph is introduced here; see Section 3).

---

## 3. Dependencies to Install

This phase requires the following third-party packages:

```bash
uv add langgraph langchain-core
```

> `langgraph` provides `StateGraph`, `Runtime`, and the `add_messages` reducer used by `AgentState`. `langchain-core` provides `AnyMessage` (used in the `messages` field) and `HumanMessage`/`AIMessage`/`ToolMessage` (used by nodes in PHASE 14). `pydantic` (already installed) is used for the domain models and `GraphConfig`. `logfire` (already installed) is imported by the node files in PHASE 14, not here.

If you are not using `uv`, install with pip:

```bash
pip install langgraph langchain-core
```

> **Note:** `langgraph` pulls in `langchain-core` transitively, but pinning both explicitly documents the dependency. The `Runtime[Context]` type used by nodes in PHASE 14 comes from `langgraph.runtime`.

---

## 4. Directory Structure to Create

This phase creates the `src/services/agents/` package and its core files:

```
src/
└── services/
    └── agents/
        ├── __init__.py              # Package marker
        ├── state.py                 # AgentState TypedDict
        ├── context.py               # Context dataclass (LangGraph context_schema)
        ├── models.py                # Pydantic domain models
        ├── config.py                # GraphConfig
        ├── prompts.py               # Shared prompt templates
        ├── factory.py               # (PHASE 15) make_agentic_rag_service
        ├── agentic_rag.py           # (PHASE 15) AgenticRAGService
        ├── tools.py                 # (PHASE 15) create_retriever_tool
        ├── supervisor_agent.py      # (PHASE 15) SupervisorAgent
        ├── summarizer_agent.py      # (PHASE 15) SummarizerAgent
        └── nodes/                   # (PHASE 14) node functions
```

> Only `__init__.py`, `state.py`, `context.py`, `models.py`, `config.py`, and `prompts.py` are created in this phase. The `nodes/` subpackage and the remaining agent files are built in PHASE 14 and PHASE 15. Create an empty `__init__.py` in `src/services/agents/` so the package imports cleanly.

---

## 5. Step-by-Step Implementation

### Step 1 — Create the agents package marker

**Full file path:** `src/services/agents/__init__.py`

```python
"""Agentic RAG workflow package."""
```

**Explanation:** A minimal package marker so `src.services.agents` is importable. It intentionally does not re-export anything — the submodules are imported directly where needed.

---

### Step 2 — Define the Pydantic domain models

**Full file path:** `src/services/agents/models.py`

```python
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class GuardrailScoring(BaseModel):
    """Scoring result of a user query for guardrail validation.

    :param score: Relevance score between 0 and 100
    :param reason: Brief explanation for the score
    """

    score: int = Field(ge=0, le=100, description="Relevance score between 0 and 100")
    reason: str = Field(description="Brief reason for the score")


class GradeDocuments(BaseModel):
    """Binary score for document relevance check.

    :param binary_score: Relevance score: 'yes' or 'no'
    :param reasoning: Explanation for the relevance decision
    """

    binary_score: Literal["yes", "no"] = Field(description="Document relevance: 'yes' or 'no'")
    reasoning: str = Field(default="", description="Explanation for the decision")


class SourceItem(BaseModel):
    """Source item from retrieved documents.

    :param arxiv_id: arXiv paper ID
    :param title: Paper title
    :param authors: List of authors
    :param url: Link to the paper
    :param relevance_score: Relevance score from retrieval
    """

    arxiv_id: str = Field(description="arXiv paper ID")
    title: str = Field(description="Paper title")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    url: str = Field(description="Link to paper")
    relevance_score: float = Field(default=0.0, description="Relevance score from search")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "url": self.url,
            "relevance_score": self.relevance_score,
        }


class ToolArtefact(BaseModel):
    """Artifact returned by tool calls with metadata.

    :param tool_name: Name of the tool that generated this artifact
    :param tool_call_id: Unique ID of the tool call
    :param content: The actual content/result from the tool
    :param metadata: Additional metadata about the tool execution
    """

    tool_name: str = Field(description="Name of the tool")
    tool_call_id: str = Field(description="Unique tool call ID")
    content: Any = Field(description="Tool result content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RoutingDecision(BaseModel):
    """Routing decision for graph navigation.

    :param route: The next node to route to
    :param reason: Explanation for the routing decision
    """

    route: Literal["retrieve", "out_of_scope", "generate_answer", "rewrite_query"] = Field(
        description="Next node to route to"
    )
    reason: str = Field(default="", description="Reason for routing decision")


class GradingResult(BaseModel):
    """Result of document grading with details.

    :param document_id: Identifier for the graded document
    :param is_relevant: Whether document is relevant
    :param score: Relevance score
    :param reasoning: Explanation for the grade
    """

    document_id: str = Field(description="Document identifier")
    is_relevant: bool = Field(description="Relevance flag")
    score: float = Field(default=0.0, description="Relevance score")
    reasoning: str = Field(default="", description="Grading reasoning")


class ReasoningStep(BaseModel):
    """A reasoning step in the agent workflow.

    :param step_name: Name of the step/node
    :param description: Human-readable description
    :param metadata: Additional step metadata
    """

    step_name: str = Field(description="Name of the reasoning step")
    description: str = Field(description="Human-readable description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Step metadata")
```

**Explanation:** These are the Pydantic models that give structure to the values stored in `AgentState`. `GuardrailScoring` is produced by the guardrail node and consumed by the `continue_after_guardrail` edge function. `RoutingDecision` captures the `route` literal used by the grade-documents conditional edge. `SourceItem` represents a retrieved paper and provides `to_dict()` for JSON serialization in the API response. `GradingResult` records per-document relevance. `GradeDocuments` and `ReasoningStep` are supporting models (the former mirrors the LLM's structured grading output; the latter records reasoning steps). Note `SourceItem` has no `abstract` field — the output guardrail node reads `abstract` defensively via `getattr`.

---

### Step 3 — Define the AgentState TypedDict

**Full file path:** `src/services/agents/state.py`

```python
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from .models import GradingResult, GuardrailScoring, RoutingDecision, SourceItem, ToolArtefact


class AgentState(TypedDict):
    """State class for the Agentic RAG workflow.

    TypedDict-based state following LangGraph 2025 best practices.
    Tracks all data that needs to be passed between nodes.

    :cvar messages:
        List of messages in the conversation. Uses add_messages reducer
        to append new messages rather than overwrite.
    :type messages: Annotated[list[AnyMessage], add_messages]

    :cvar original_query:
        The original user query before any rewrites.
    :type original_query: Optional[str]

    :cvar rewritten_query:
        The rewritten query after optimization for better retrieval.
    :type rewritten_query: Optional[str]

    :cvar retrieval_attempts:
        Number of retrieval attempts made (for max attempt tracking).
    :type retrieval_attempts: int

    :cvar guardrail_result:
        Result from guardrail validation with score and reasoning.
    :type guardrail_result: Optional[GuardrailScoring]

    :cvar routing_decision:
        The routing decision determining the next node in the graph.
    :type routing_decision: Optional[RoutingDecision]

    :cvar sources:
        Dictionary mapping tool_call_id to their output sources.
    :type sources: Optional[Dict[str, Any]]

    :cvar relevant_sources:
        List of relevant sources to display to the user.
    :type relevant_sources: List[SourceItem]

    :cvar relevant_tool_artefacts:
        List of tool artifacts with metadata from tool executions.
    :type relevant_tool_artefacts: Optional[List[ToolArtefact]]

    :cvar grading_results:
        List of grading results for each retrieved document.
    :type grading_results: List[GradingResult]

    :cvar metadata:
        Runtime metadata for tracing and analytics.
    :type metadata: Dict[str, Any]
    """

    messages: Annotated[list[AnyMessage], add_messages]
    original_query: Optional[str]
    rewritten_query: Optional[str]
    sanitized_query: Optional[str]
    retrieval_attempts: int
    guardrail_result: Optional[GuardrailScoring]
    output_guardrail_filter: Optional[str]
    routing_decision: Optional[RoutingDecision]
    sources: Optional[Dict[str, Any]]
    relevant_sources: List[SourceItem]
    relevant_tool_artefacts: Optional[List[ToolArtefact]]
    grading_results: List[GradingResult]
    metadata: Dict[str, Any]
```

**Explanation:** `AgentState` is a `TypedDict` (not a Pydantic model) — the recommended LangGraph pattern. The `messages` field uses the `Annotated[list[AnyMessage], add_messages]` annotation so that when a node returns `{"messages": [...]}`, LangGraph **appends** to the existing message list rather than overwriting it. Every other field is a plain key that is overwritten when a node returns it. The `sanitized_query` field is set by the guardrail node when Bedrock PII-anonymizes the input; nodes prefer it over the raw query via `state.get("sanitized_query") or get_latest_query(...)`. `sources`, `relevant_tool_artefacts`, and `metadata` are vestigial in the current implementation (kept for schema completeness) — the graph primarily uses `messages`, `relevant_sources`, `grading_results`, `retrieval_attempts`, `guardrail_result`, `output_guardrail_filter`, `routing_decision`, `original_query`, `rewritten_query`, and `sanitized_query`.

---

### Step 4 — Define the Context dataclass

**Full file path:** `src/services/agents/context.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from langfuse._client.span import LangfuseSpan
from src.services.bedrock_guardrails.service import BedrockGuardrailsService
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.llm_client_protocol import LLMClientProtocol
from src.services.opensearch.client import OpenSearchClient


@dataclass
class Context:
    """Runtime context for agent dependencies.

    This contains immutable dependencies that nodes need but don't modify.

    :param llm_client: LLM client (OpenAI or Bedrock — satisfies LLMClientProtocol)
    :param opensearch_client: Client for document search
    :param embeddings_client: Client for embeddings
    :param langfuse_tracer: Optional tracer for observability
    :param guardrails_service: Optional Bedrock Guardrails service
    :param trace: Current Langfuse trace object (if enabled)
    :param langfuse_enabled: Whether Langfuse tracing is enabled
    :param model_name: Model to use for LLM calls
    :param temperature: Temperature for generation
    :param top_k: Number of documents to retrieve
    :param max_retrieval_attempts: Maximum retrieval attempts
    :param guardrail_threshold: Threshold for guardrail validation (0-100)
    """

    llm_client: LLMClientProtocol
    opensearch_client: OpenSearchClient
    embeddings_client: JinaEmbeddingsClient
    langfuse_tracer: Optional[LangfuseTracer]
    guardrails_service: Optional[BedrockGuardrailsService] = None
    trace: Optional["LangfuseSpan"] = None
    langfuse_enabled: bool = False
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    top_k: int = 3
    max_retrieval_attempts: int = 2
    guardrail_threshold: int = 60
```

**Explanation:** `Context` is the LangGraph `context_schema` — a dataclass of **immutable dependencies and tuning parameters** that nodes read via `runtime.context` but never modify. The first four fields (`llm_client`, `opensearch_client`, `embeddings_client`, `langfuse_tracer`) are required; the rest have defaults. `guardrails_service` defaults to `None` (fail-open). `trace` holds the current Langfuse span object and `langfuse_enabled` gates tracing. Note the defaults here (`guardrail_threshold=60`, `max_retrieval_attempts=2`) differ from `GraphConfig` defaults (`guardrail_threshold=40`) — the `AgenticRAGService` in PHASE 15 explicitly passes `GraphConfig` values into `Context`, so the `Context` defaults are only used when a `Context` is constructed directly (e.g., the supervisor agent in `main.py`).

---

### Step 5 — Define the GraphConfig

**Full file path:** `src/services/agents/config.py`

```python
from typing import Any, Dict

from pydantic import BaseModel, Field
from src.config import Settings, get_settings


class GraphConfig(BaseModel):
    """Configuration for the entire graph execution.

    This is the configuration used by AgenticRAGService for controlling
    graph behavior, retrieval settings, and execution parameters.

    :param max_retrieval_attempts: Maximum number of retrieval attempts before fallback
    :param guardrail_threshold: Threshold score for guardrail validation (0-100)
    :param model: Default model to use for LLM calls (e.g., "qwen3:4b")
    :param temperature: Temperature for LLM generation (0.0 = deterministic)
    :param top_k: Number of documents to retrieve from search
    :param use_hybrid: Whether to use hybrid search (BM25 + vector)
    :param enable_tracing: Whether to enable Langfuse tracing
    :param metadata: Additional runtime metadata for tracking and analytics
    :param settings: Application settings instance for environment and service config
    """

    max_retrieval_attempts: int = 2
    guardrail_threshold: int = 40
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    top_k: int = 3
    use_hybrid: bool = True
    enable_tracing: bool = True
    metadata: Dict[str, Any] = {}
    settings: Settings = Field(default_factory=get_settings)
```

**Explanation:** `GraphConfig` is the Pydantic model that controls graph execution. It is constructed by `make_agentic_rag_service` (PHASE 15) with `top_k`, `use_hybrid`, and a provider-derived `model`, then passed to `AgenticRAGService`, which copies its values into the per-request `Context`. The `settings` field defaults to `get_settings()` so the service can read `environment` for trace metadata. `max_retrieval_attempts=2` caps the retrieval loop.

---

### Step 6 — Define the shared prompt templates

**Full file path:** `src/services/agents/prompts.py`

```python
# Grade documents for relevance (used in grade_documents_node)
GRADE_DOCUMENTS_PROMPT = """You are a grader assessing relevance of retrieved documents to a user question.

Retrieved Documents:
{context}

User Question: {question}

If the documents contain keywords or semantic meaning related to the question, grade them as relevant.
Give a binary score 'yes' or 'no' to indicate whether the documents are relevant to the question.
Also provide brief reasoning for your decision.

Respond in JSON format with 'binary_score' (yes/no) and 'reasoning' fields."""

# Rewrite query for better retrieval
REWRITE_PROMPT = """You are a question re-writer that converts an input question to a better version that is optimized for retrieving relevant documents.

Look at the initial question and try to reason about the underlying semantic intent or meaning.

Here is the initial question:
{question}

Formulate an improved question that will retrieve more relevant documents.
Provide only the improved question without any preamble or explanation."""

# System message for query generation/response
SYSTEM_MESSAGE = """You are an AI assistant specializing in academic research papers from arXiv.
Your domain of expertise is: Computer Science, Machine Learning, AI, and related technical research.

You have access to a tool to retrieve relevant research papers. Use this tool when:
- The user asks about specific research topics in CS/AI/ML
- The question requires knowledge from academic papers (e.g., "What are transformer architectures?")
- You need context from scientific literature (e.g., "How does BERT work?")

Do NOT use the tool when:
- The question is about general knowledge unrelated to research (e.g., "What is the meaning of dog?")
- The question is simple factual or mathematical (e.g., "what is 2+2?")
- The question is conversational, greeting, or personal
- The question is about topics outside CS/AI/ML research (e.g., cooking, history, medicine)

When you use the retrieval tool, you will receive relevant paper excerpts to help answer the question."""

# Decision prompt for routing
DECISION_PROMPT = """You are an AI assistant that ONLY helps with academic research papers from arXiv in Computer Science, AI, and Machine Learning.

Question: "{question}"

Is this question about CS/AI/ML research that requires academic papers?

CRITICAL RULES:
- RETRIEVE: ONLY if the question is specifically about AI/ML/CS research topics (neural networks, algorithms, models, techniques)
- RESPOND: For EVERYTHING else (general knowledge, definitions, greetings, non-research questions)

Examples:
- "What are transformer architectures in deep learning?" -> RETRIEVE
- "Explain BERT model" -> RETRIEVE
- "What is the meaning of dog?" -> RESPOND (general dictionary definition)
- "What is a dog?" -> RESPOND (not about research)
- "Hello" -> RESPOND (greeting)
- "What is 2+2?" -> RESPOND (math, not research)

Answer with ONLY ONE WORD: "RETRIEVE" or "RESPOND"

Your answer:"""

# Direct response prompt (no retrieval)
DIRECT_RESPONSE_PROMPT = """You are an AI assistant specializing in academic research papers from arXiv (Computer Science, AI, ML).

The following question appears to be outside the scope of academic research papers or doesn't require retrieval from research literature:

Question: {question}

Explain that this question is outside your domain of expertise (arXiv research papers in CS/AI/ML) and that you cannot answer it accurately. Be helpful by suggesting what kind of resource would be more appropriate for this question.

Answer:"""

# Guardrail validation prompt (used in guardrail_node)
GUARDRAIL_PROMPT = """You are a guardrail evaluator assessing whether a user query is within the scope of academic research papers from arXiv in Computer Science, AI, and Machine Learning.

User Query: {question}

Evaluate whether this query is:
- About CS/AI/ML research topics (neural networks, algorithms, models, architectures, techniques, etc.)
- Requires academic paper knowledge to answer
- Within the domain of Computer Science research

Assign a relevance score (0-100):
- 80-100: Clearly about CS/AI/ML research (e.g., "What are transformer architectures?", "How does BERT work?")
- 60-79: Potentially research-related but unclear (e.g., "Tell me about attention mechanisms")
- 40-59: Borderline or ambiguous (e.g., "What is machine learning?")
- 0-39: NOT about research papers (e.g., "What is a dog?", "Hello", "What is 2+2?")

Provide:
1. A score between 0 and 100
2. A brief reason explaining why you gave this score

Respond in JSON format with 'score' (integer 0-100) and 'reason' (string) fields."""

# Answer generation prompt (used in generate_answer_node)
GENERATE_ANSWER_PROMPT = """You are an AI research assistant specializing in academic papers from arXiv in Computer Science, AI, and Machine Learning.

Your task is to answer the user's question using ONLY the information from the retrieved research papers provided below.

Retrieved Research Papers:
{context}

User Question: {question}

Instructions:
- Provide a comprehensive, accurate answer based ONLY on the retrieved papers
- Cite specific papers when making claims (use paper titles or arxiv IDs)
- If the papers don't contain enough information to fully answer the question, acknowledge this
- Structure your answer clearly and professionally
- Focus on the key insights and findings from the papers
- Do NOT make up information or cite papers not in the retrieved context

Answer:"""
```

**Explanation:** These are the shared prompt templates used by the PHASE 14 nodes. `GRADE_DOCUMENTS_PROMPT` and `REWRITE_PROMPT` are actively used by the grade-documents and rewrite-query nodes. `GENERATE_ANSWER_PROMPT` is used by the generate-answer node. `SYSTEM_MESSAGE`, `DECISION_PROMPT`, `DIRECT_RESPONSE_PROMPT`, and `GUARDRAIL_PROMPT` are **partially dead** in the current implementation — they are retained for reference but the guardrail node uses AWS Bedrock Guardrails (not `GUARDRAIL_PROMPT`), and the out-of-scope node returns a hardcoded message. Keep them for completeness and future use, but do not wire them into nodes.

---

## 6. Configuration

The agent workflow is configured via the `GraphConfig` model (Section 5, Step 5), which reads the application `Settings` from [`src/config.py`](../Agentic-RAG-project-agentops/src/config.py). The relevant settings come from prior phases:

| Setting | Source | Used for |
| --- | --- | --- |
| `settings.provider` | `PROVIDER` env var | Choosing `bedrock` vs `openai` model ID in `make_agentic_rag_service` (PHASE 15) |
| `settings.bedrock.model_id` | `BEDROCK__MODEL_ID` | Default model when provider is `bedrock` |
| `settings.openai_model` | `OPENAI_MODEL` | Default model when provider is `openai` |
| `settings.environment` | `ENVIRONMENT` | Trace metadata (`env`) in `AgenticRAGService.ask` |
| `settings.bedrock.guardrail_id` | `BEDROCK__GUARDRAIL_ID` | Whether `BedrockGuardrailsService` is active (fail-open when empty) |

**Example `.env` snippet (relevant to this phase):**

```dotenv
PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
ENVIRONMENT=development
BEDROCK__GUARDRAIL_ID=
```

> **Note:** `GraphConfig` defaults (`guardrail_threshold=40`, `max_retrieval_attempts=2`, `top_k=3`, `use_hybrid=True`) are the effective runtime values when `make_agentic_rag_service` does not override them. The `Context` dataclass has its own defaults (`guardrail_threshold=60`) that only apply when a `Context` is constructed directly (e.g., the supervisor agent in `main.py`).

---

## 7. Verification

Verify the agent state/context layer imports and validates correctly. No graph execution is possible yet (nodes and wiring come in PHASE 14/15), so verification is limited to import and model-construction checks.

### 7.1 Import smoke test

```bash
python -c "
from src.services.agents.state import AgentState
from src.services.agents.context import Context
from src.services.agents.models import GuardrailScoring, SourceItem, RoutingDecision, GradingResult
from src.services.agents.config import GraphConfig
from src.services.agents import prompts

print('AgentState fields:', list(AgentState.__annotations__.keys()))
print('GraphConfig model:', GraphConfig().model)
print('GraphConfig max_retrieval_attempts:', GraphConfig().max_retrieval_attempts)
print('Prompts defined:', all(hasattr(prompts, n) for n in [
    'GRADE_DOCUMENTS_PROMPT', 'REWRITE_PROMPT', 'SYSTEM_MESSAGE', 'DECISION_PROMPT',
    'DIRECT_RESPONSE_PROMPT', 'GUARDRAIL_PROMPT', 'GENERATE_ANSWER_PROMPT']))
"
```

**Expected:** All modules import without error. `AgentState` lists all 13 fields, `GraphConfig` defaults are `gpt-4o-mini` / `2`, and all 7 prompt constants exist.

### 7.2 Model construction test

```bash
python -c "
from src.services.agents.models import GuardrailScoring, SourceItem, RoutingDecision, GradingResult

g = GuardrailScoring(score=100, reason='pass')
print('GuardrailScoring:', g.score, g.reason)

s = SourceItem(arxiv_id='1706.03762', title='Attention', url='https://arxiv.org/pdf/1706.03762.pdf')
print('SourceItem to_dict:', s.to_dict())

r = RoutingDecision(route='generate_answer', reason='relevant')
print('RoutingDecision route:', r.route)

gr = GradingResult(document_id='d1', is_relevant=True, score=1.0)
print('GradingResult:', gr.is_relevant, gr.score)
"
```

**Expected:** All four models construct and validate. `SourceItem.to_dict()` returns the serializable dict with `arxiv_id`, `title`, `authors`, `url`, `relevance_score`.

### 7.3 Context construction test

```bash
python -c "
from src.services.agents.context import Context
from src.services.agents.config import GraphConfig

# Context requires the four client deps; verify defaults on a partial via dataclass defaults
import dataclasses
fields = {f.name: f.default for f in dataclasses.fields(Context)}
print('Context defaults:', fields)

cfg = GraphConfig()
print('GraphConfig guardrail_threshold:', cfg.guardrail_threshold)
print('GraphConfig top_k:', cfg.top_k)
"
```

**Expected:** `Context` exposes the four required client fields and the defaulted tuning fields (`guardrail_threshold=60`, `max_retrieval_attempts=2`, `top_k=3`, `model_name='gpt-4o-mini'`). `GraphConfig` defaults differ (`guardrail_threshold=40`) — this is expected and intentional.

### 7.4 LangGraph import check

```bash
python -c "
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
print('LangGraph imports OK')
"
```

**Expected:** LangGraph imports successfully, confirming the dependency is installed for PHASE 14/15.

---

## 8. Common Pitfalls

- **`messages` must use the `add_messages` reducer.** If you annotate `messages` as a plain `list[AnyMessage]` without `Annotated[..., add_messages]`, every node returning `{"messages": [...]}` will **overwrite** the conversation instead of appending. This breaks the retrieve → tool → grade → generate flow because earlier messages (the tool result) would be lost.

- **`Context` is the `context_schema`, not part of `AgentState`.** Dependencies live in `Context` and are accessed via `runtime.context` in nodes. Do not put clients (LLM, OpenSearch, embeddings) into `AgentState` — state should only carry data that flows between nodes.

- **`Context` defaults differ from `GraphConfig` defaults.** `Context.guardrail_threshold` defaults to `60` while `GraphConfig.guardrail_threshold` defaults to `40`. When `AgenticRAGService` builds a `Context` (PHASE 15) it explicitly passes `GraphConfig` values, so the `Context` defaults are only used for direct construction (e.g., the supervisor agent). Do not rely on `Context` defaults matching `GraphConfig` defaults.

- **`guardrails_service` is optional and fail-open.** The guardrail node treats a `None` service as "allow all" (score 100). The `Context` field defaults to `None`. Ensure your code checks `if runtime.context.guardrails_service:` before calling `check_input`/`check_output`.

- **`sanitized_query` is set only when Bedrock PII-anonymizes.** The guardrail node returns `sanitized_query` only when `result.outputs` is non-empty and `result.allowed`. Nodes must fall back to `get_latest_query(state["messages"])` when `sanitized_query` is absent (they do via `state.get("sanitized_query") or get_latest_query(...)`).

- **`SourceItem` has no `abstract` field.** The output guardrail node reads `getattr(source, "abstract", "")` defensively. Do not add an `abstract` field unless you also update the node logic; the current schema intentionally omits it.

- **`sources`, `relevant_tool_artefacts`, and `metadata` are vestigial.** They are declared in `AgentState` for schema completeness but are not populated by the current nodes. Keep them declared to match the reference, but do not build logic that depends on them.

- **`prompts.py` contains partially-dead constants.** `SYSTEM_MESSAGE`, `DECISION_PROMPT`, `DIRECT_RESPONSE_PROMPT`, and `GUARDRAIL_PROMPT` are not used by the current nodes. Keep them for reference, but do not wire them in — the guardrail node uses Bedrock Guardrails and the out-of-scope node returns a hardcoded message.

- **`langfuse._client.span.LangfuseSpan` import.** `context.py` imports `LangfuseSpan` from `langfuse._client.span` for the `trace` type hint. This is a private module path that can change across Langfuse versions. If the import breaks, use a `TYPE_CHECKING` guard or a string annotation (`"LangfuseSpan"`) — the field is already annotated as `Optional["LangfuseSpan"]`.

---

## 9. Definition of Done

This phase is complete when:

- [ ] `src/services/agents/__init__.py` exists as a package marker.
- [ ] `src/services/agents/models.py` defines `GuardrailScoring`, `GradeDocuments`, `SourceItem` (with `to_dict`), `ToolArtefact`, `RoutingDecision`, `GradingResult`, and `ReasoningStep`.
- [ ] `src/services/agents/state.py` defines `AgentState` as a `TypedDict` with all 13 fields, using `Annotated[list[AnyMessage], add_messages]` for `messages`.
- [ ] `src/services/agents/context.py` defines the `Context` dataclass with the four required client fields and the defaulted tuning fields (`model_name