# PHASE 4 — Schemas

> **Build guide for the from-scratch reconstruction of the arXiv Paper Curator (Agentic RAG).**
> This phase implements all Pydantic schemas in `src/schemas/` — the data contracts used by the API, services, and persistence layers.

---

## 1. Phase Objective

Implement every Pydantic schema module under `src/schemas/`. These schemas define the **data shapes** that flow through the system:

- **`api/`** — request/response models for the HTTP endpoints (`ask`, `health`, `search`).
- **`arxiv/`** — paper metadata schemas (used by the arXiv client and the repository).
- **`database/`** — `PostgreSQLSettings` (used by the database layer).
- **`embeddings/`** — Jina embeddings request/response models.
- **`indexing/`** — chunking metadata and text-chunk models.
- **`pdf_parser/`** — PDF parsing models (sections, figures, tables, parsed paper).
- **`ollama.py`** — structured output models for Ollama RAG responses.

**Why this phase comes at this point:** Schemas are the domain contracts consumed by repositories (PHASE 3), services, and routers. They depend only on Pydantic (and, for `database/config.py`, `pydantic-settings`). Defining them now ensures all later phases have stable, validated data shapes. Note that PHASE 3 already created `src/schemas/database/config.py` as a dependency — this phase documents it fully and adds the rest.

> **Known issue to handle:** `src/schemas/common/__init__.py` in the reference project has a **broken import** — it imports from `src.schemas.search.hybrid`, which does not exist. **Do not create this file.** Document it as a pitfall (see Section 8).

---

## 2. Prerequisites

- **PHASE 1 complete** — project skeleton, `uv sync`, `src/` package importable.
- **PHASE 2 complete** — `src/config.py` and `src/exceptions.py` exist.
- **PHASE 3 complete** — the database layer exists and `src/schemas/database/config.py` was created.
- **`pydantic>=2.11.3`** (locked `2.13.4`) and **`pydantic-settings>=2.8.1`** (locked `2.14.1`) installed.

---

## 3. Dependencies to Install

No new dependencies are added in this phase. Required packages (already in `pyproject.toml` from PHASE 1):

- `pydantic>=2.11.3` (locked `2.13.4`)
- `pydantic-settings>=2.8.1` (locked `2.14.1`)

Run `uv sync` to ensure they are installed.

---

## 4. Directory Structure to Create

```
<project-root>/src/schemas/
├── __init__.py              (this phase)
├── ollama.py                (this phase)
├── api/
│   ├── __init__.py          (empty)
│   ├── ask.py               (this phase)
│   ├── health.py            (this phase)
│   └── search.py            (this phase)
├── arxiv/
│   ├── __init__.py          (comment only)
│   └── paper.py             (this phase)
├── database/
│   ├── __init__.py          (empty — created in PHASE 3)
│   └── config.py            (created in PHASE 3; documented here)
├── embeddings/
│   ├── __init__.py          (empty)
│   └── jina.py              (this phase)
├── indexing/
│   ├── __init__.py          (empty)
│   └── models.py            (this phase)
├── pdf_parser/
│   ├── __init__.py          (comment only)
│   └── models.py            (this phase)
└── telegram/
    └── __init__.py          (empty — `__all__: list[str] = []`)
```

> **Do NOT create `src/schemas/common/`** — the reference file there has a broken import (see Section 8).

---

## 5. Step-by-step Implementation

### Step 1 — Create `src/schemas/api/ask.py`

**Full file path:** `<project-root>/src/schemas/api/ask.py`

Write the following content (this is the exact reference implementation):

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

**Explanation:**
- **`AskRequest`** is the request body for `/ask`, `/ask-agentic`, and `/ask-supervisor`. It validates `query` (1–1000 chars), `top_k` (1–10), `use_hybrid`, an optional `model`, and optional `categories` filter.
- **`AskResponse`** is the standard RAG response: `query`, `answer`, `sources` (PDF URLs), `chunks_used`, and `search_mode`.
- **`AgenticAskResponse`** extends `AskResponse` and **overrides `sources`** to `List[Any]` (rich source objects instead of plain URLs), adding `reasoning_steps`, `retrieval_attempts`, `rewritten_query`, `trace_id`, `guardrail_filter`, and `output_guardrail_filter`.
- **`FeedbackRequest`** / **`FeedbackResponse`** support the `/feedback` endpoint (Langfuse trace scoring).
- The `class Config: json_schema_extra` blocks provide OpenAPI example payloads.

### Step 2 — Create `src/schemas/api/health.py`

**Full file path:** `<project-root>/src/schemas/api/health.py`

Write the following content (this is the exact reference implementation):

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

**Explanation:** `ServiceStatus` models an individual dependency's health; `HealthResponse` aggregates the overall status, version, environment, service name, and a map of per-service statuses. Used by the `/api/v1/health` endpoint.

### Step 3 — Create `src/schemas/api/search.py`

**Full file path:** `<project-root>/src/schemas/api/search.py`

Write the following content (this is the exact reference implementation):

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

**Explanation:**
- **`SearchRequest`** is the (currently unused) keyword search request. It uses `from_` with `alias="from"` so the JSON field is `from` while the Python attribute is `from_`.
- **`HybridSearchRequest`** is the request for the `/api/v1/hybrid-search/` endpoint, supporting all search modes with `use_hybrid`, `min_score`, `categories`, and `latest_papers`. It sets `populate_by_name = True` so both the alias and the field name are accepted.
- **`SearchHit`** models an individual result, including chunk-specific fields (`chunk_text`, `chunk_id`, `section_name`) for unified search.
- **`SearchResponse`** wraps hits with `total`, `size`, `from_` (alias `from`), `search_mode`, and an optional `error`.

### Step 4 — Create `src/schemas/arxiv/paper.py`

**Full file path:** `<project-root>/src/schemas/arxiv/paper.py`

Write the following content (this is the exact reference implementation):

```python
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ArxivPaper(BaseModel):
    """Schema for arXiv API response data."""

    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(..., description="List of author names")
    abstract: str = Field(..., description="Paper abstract")
    categories: List[str] = Field(..., description="Paper categories")
    published_date: str = Field(..., description="Date published on arXiv (ISO format)")
    pdf_url: str = Field(..., description="URL to PDF")


class PaperBase(BaseModel):
    # Core arXiv metadata
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(..., description="List of author names")
    abstract: str = Field(..., description="Paper abstract")
    categories: List[str] = Field(..., description="Paper categories")
    published_date: datetime = Field(..., description="Date published on arXiv")
    pdf_url: str = Field(..., description="URL to PDF")


class PaperCreate(PaperBase):
    """Schema for creating a paper with optional parsed content."""

    # Parsed PDF content (optional - added when PDF is processed)
    raw_text: Optional[str] = Field(None, description="Full raw text extracted from PDF")
    sections: Optional[List[Dict[str, Any]]] = Field(None, description="List of sections with titles and content")
    references: Optional[List[Dict[str, Any]]] = Field(None, description="List of references if extracted")

    # PDF processing metadata (optional)
    parser_used: Optional[str] = Field(None, description="Which parser was used (DOCLING, etc.)")
    parser_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional parser metadata")
    pdf_processed: Optional[bool] = Field(False, description="Whether PDF was successfully processed")
    pdf_processing_date: Optional[datetime] = Field(None, description="When PDF was processed")


class PaperResponse(PaperBase):
    """Schema for paper API responses with all content."""

    id: UUID

    # Parsed PDF content (optional fields)
    raw_text: Optional[str] = Field(None, description="Full raw text extracted from PDF")
    sections: Optional[List[Dict[str, Any]]] = Field(None, description="List of sections with titles and content")
    references: Optional[List[Dict[str, Any]]] = Field(None, description="List of references if extracted")

    # PDF processing metadata
    parser_used: Optional[str] = Field(None, description="Which parser was used")
    parser_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional parser metadata")
    pdf_processed: bool = Field(False, description="Whether PDF was successfully processed")
    pdf_processing_date: Optional[datetime] = Field(None, description="When PDF was processed")

    # Timestamps
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PaperSearchResponse(BaseModel):
    papers: List[PaperResponse]
    total: int
```

**Explanation:**
- **`ArxivPaper`** models the raw arXiv API response (note `published_date` is a `str` here — ISO format from the API).
- **`PaperBase`** is the shared base with core metadata; note `published_date` is a `datetime` here (parsed).
- **`PaperCreate`** extends `PaperBase` with optional parsed PDF content and processing metadata. This is the schema used by `PaperRepository.create`/`upsert` (PHASE 3).
- **`PaperResponse`** extends `PaperBase` with the `id` (UUID), all parsed content, processing metadata, and timestamps. It sets `from_attributes = True` so it can be built directly from an ORM `Paper` object.
- **`PaperSearchResponse`** wraps a list of `PaperResponse` with a `total`.

### Step 5 — Create `src/schemas/database/config.py`

**Full file path:** `<project-root>/src/schemas/database/config.py`

This file was created in PHASE 3 (Step 0). It is documented here for completeness. The content is:

```python
from pydantic import Field
from pydantic_settings import BaseSettings


class PostgreSQLSettings(BaseSettings):
    """PostgreSQL configuration settings."""

    database_url: str = Field(
        default="postgresql://rag_user:rag_password@localhost:5432/rag_db", description="PostgreSQL database URL"
    )
    echo_sql: bool = Field(default=False, description="Enable SQL query logging")
    pool_size: int = Field(default=20, description="Database connection pool size")
    max_overflow: int = Field(default=0, description="Maximum pool overflow")

    class Config:
        env_prefix = "POSTGRES_"
```

**Explanation:** `PostgreSQLSettings` is a Pydantic Settings class with `env_prefix = "POSTGRES_"`. It is consumed by `make_database()` in PHASE 3. Ensure `src/schemas/database/__init__.py` exists as an empty file.

### Step 6 — Create `src/schemas/embeddings/jina.py`

**Full file path:** `<project-root>/src/schemas/embeddings/jina.py`

Write the following content (this is the exact reference implementation):

```python
from typing import Dict, List

from pydantic import BaseModel


class JinaEmbeddingRequest(BaseModel):
    """Request model for Jina embeddings API."""

    model: str = "jina-embeddings-v3"
    task: str = "retrieval.passage"  # or "retrieval.query" for queries
    dimensions: int = 1024
    late_chunking: bool = False
    embedding_type: str = "float"
    input: List[str]


class JinaEmbeddingResponse(BaseModel):
    """Response model from Jina embeddings API."""

    model: str
    object: str = "list"
    usage: Dict[str, int]
    data: List[Dict]
```

**Explanation:** `JinaEmbeddingRequest` models the request to the Jina embeddings API (`jina-embeddings-v3`, 1024-dim, `retrieval.passage` task by default). `JinaEmbeddingResponse` models the API response (`model`, `object`, `usage`, `data`).

### Step 7 — Create `src/schemas/indexing/models.py`

**Full file path:** `<project-root>/src/schemas/indexing/models.py`

Write the following content (this is the exact reference implementation):

```python
from typing import Optional

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk."""

    chunk_index: int
    start_char: int
    end_char: int
    word_count: int
    overlap_with_previous: int
    overlap_with_next: int
    section_title: Optional[str] = None


class TextChunk(BaseModel):
    """A chunk of text with metadata."""

    text: str
    metadata: ChunkMetadata
    arxiv_id: str
    paper_id: str
```

**Explanation:** `ChunkMetadata` captures chunk position/size/overlap info; `TextChunk` combines the text with its metadata and the owning paper (`arxiv_id`, `paper_id`). Used by the text chunker and hybrid indexer in later phases.

### Step 8 — Create `src/schemas/pdf_parser/models.py`

**Full file path:** `<project-root>/src/schemas/pdf_parser/models.py`

Write the following content (this is the exact reference implementation):

```python
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ParserType(str, Enum):
    """PDF parser types."""

    DOCLING = "docling"


class PaperSection(BaseModel):
    """Represents a section of a paper."""

    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    level: int = Field(default=1, description="Section hierarchy level")


class PaperFigure(BaseModel):
    """Represents a figure in a paper."""

    caption: str = Field(..., description="Figure caption")
    id: str = Field(..., description="Figure identifier")


class PaperTable(BaseModel):
    """Represents a table in a paper."""

    caption: str = Field(..., description="Table caption")
    id: str = Field(..., description="Table identifier")


class PdfContent(BaseModel):
    """PDF-specific content extracted by parsers like Docling."""

    sections: List[PaperSection] = Field(default_factory=list, description="Paper sections")
    figures: List[PaperFigure] = Field(default_factory=list, description="Figures")
    tables: List[PaperTable] = Field(default_factory=list, description="Tables")
    raw_text: str = Field(..., description="Full extracted text")
    references: List[str] = Field(default_factory=list, description="References")
    parser_used: ParserType = Field(..., description="Parser used for extraction")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Parser metadata")


class ArxivMetadata(BaseModel):
    """Paper metadata from arXiv API."""

    title: str = Field(..., description="Paper title from arXiv")
    authors: List[str] = Field(..., description="Authors from arXiv")
    abstract: str = Field(..., description="Abstract from arXiv")
    arxiv_id: str = Field(..., description="arXiv identifier")
    categories: List[str] = Field(default_factory=list, description="arXiv categories")
    published_date: str = Field(..., description="Publication date")
    pdf_url: str = Field(..., description="PDF download URL")


class ParsedPaper(BaseModel):
    """Complete paper data combining arXiv metadata and PDF content."""

    arxiv_metadata: ArxivMetadata = Field(..., description="Metadata from arXiv API")
    pdf_content: Optional[PdfContent] = Field(None, description="Content extracted from PDF")
```

**Explanation:**
- **`ParserType`** is a `str` Enum with `DOCLING = "docling"`.
- **`PaperSection`**, **`PaperFigure`**, **`PaperTable`** model extracted document elements.
- **`PdfContent`** aggregates sections, figures, tables, raw text, references, the parser used, and metadata.
- **`ArxivMetadata`** models the arXiv metadata portion of a parsed paper.
- **`ParsedPaper`** combines `arxiv_metadata` and optional `pdf_content` — the complete output of the PDF parsing pipeline.

### Step 9 — Create `src/schemas/ollama.py`

**Full file path:** `<project-root>/src/schemas/ollama.py`

Write the following content (this is the exact reference implementation):

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

**Explanation:** `RAGResponse` is the structured-output model for Ollama RAG queries, capturing `answer`, `sources`, `confidence`, and `citations`.

### Step 10 — Create the `__init__.py` files

Create the following package-marker files:

**`src/schemas/api/__init__.py`** — empty file.

**`src/schemas/arxiv/__init__.py`**:
```python
# ArXiv service schemas
```

**`src/schemas/database/__init__.py`** — empty file (created in PHASE 3).

**`src/schemas/embeddings/__init__.py`** — empty file.

**`src/schemas/indexing/__init__.py`** — empty file.

**`src/schemas/pdf_parser/__init__.py`**:
```python
# PDF Parser service schemas
```

**`src/schemas/telegram/__init__.py`**:
```python
__all__: list[str] = []
```

**`src/schemas/__init__.py`** (the top-level schemas package):
```python
from .api.health import HealthResponse
from .api.search import SearchHit, SearchRequest, SearchResponse
from .arxiv.paper import ArxivPaper, PaperCreate, PaperResponse, PaperSearchResponse
from .pdf_parser.models import PaperFigure, PaperSection, PaperTable, ParsedPaper, ParserType

__all__ = [
    "HealthResponse",
    "SearchRequest",
    "SearchHit",
    "SearchResponse",
    "ArxivPaper",
    "PaperCreate",
    "PaperResponse",
    "PaperSearchResponse",
    "ParsedPaper",
    "PaperSection",
    "PaperFigure",
    "PaperTable",
    "ParserType",
]
```

**Explanation:** The `__init__.py` files make each directory an importable package. The top-level `src/schemas/__init__.py` re-exports the most commonly used schemas for convenience. The `telegram/__init__.py` uses `__all__: list[str] = []` (an empty re-export list) — the Telegram schemas are defined elsewhere and this package is currently empty.

---

## 6. Configuration

The schemas phase introduces no new environment variables. The only settings-related schema is `PostgreSQLSettings` (`src/schemas/database/config.py`), which reads the `POSTGRES_*` env vars (documented in PHASE 3). All other schemas are pure Pydantic `BaseModel` data contracts with no env-var coupling.

---

## 7. Verification

Run these commands from the project root:

```bash
# 1. Confirm all schema modules import cleanly
uv run python -c "
import src.schemas.api.ask, src.schemas.api.health, src.schemas.api.search
import src.schemas.arxiv.paper, src.schemas.database.config
import src.schemas.embeddings.jina, src.schemas.indexing.models
import src.schemas.pdf_parser.models, src.schemas.ollama
import src.schemas
print('all schema modules import OK')
"

# 2. Validate AskRequest / AskResponse round-trip
uv run python -c "
from src.schemas.api.ask import AskRequest, AskResponse
req = AskRequest(query='What are transformers?', top_k=3, use_hybrid=True)
print('AskRequest valid:', req.query, req.top_k)
resp = AskResponse(query='q', answer='a', sources=['https://x'], chunks_used=3, search_mode='hybrid')
print('AskResponse valid:', resp.search_mode)
"

# 3. Validate AgenticAskResponse
uv run python -c "
from src.schemas.api.ask import AgenticAskResponse
r = AgenticAskResponse(query='q', answer='a', sources=[{'arxiv_id':'1706.03762'}], chunks_used=1, search_mode='hybrid', reasoning_steps=['step'], retrieval_attempts=1)
print('AgenticAskResponse valid:', r.sources, r.retrieval_attempts)
"

# 4. Validate PaperCreate (used by the repository)
uv run python -c "
from datetime import datetime
from src.schemas.arxiv.paper import PaperCreate
p = PaperCreate(arxiv_id='1706.03762', title='T', authors=['A'], abstract='abs', categories=['cs.AI'], published_date=datetime(2017,6,12), pdf_url='https://x')
print('PaperCreate valid:', p.arxiv_id, p.pdf_processed)
"

# 5. Validate HybridSearchRequest alias handling
uv run python -c "
from src.schemas.api.search import HybridSearchRequest
r = HybridSearchRequest(query='q', **{'from': 5})
print('alias from works:', r.from_)
"

# 6. Validate the top-level schemas package re-exports
uv run python -c "
from src.schemas import HealthResponse, PaperCreate, ParsedPaper
print('top-level re-exports OK')
"

# 7. Confirm the broken common package is NOT present
uv run python -c "
import importlib.util
print('common package exists:', importlib.util.find_spec('src.schemas.common') is not None)
"
```

**Expected results:**
- All schema modules import without error.
- `AskRequest`, `AskResponse`, `AgenticAskResponse`, `PaperCreate`, and `HybridSearchRequest` validate correctly.
- The `from` alias on `HybridSearchRequest` works (`r.from_ == 5`).
- The top-level `src.schemas` package re-exports work.
- `src.schemas.common` is **not** present (it was intentionally omitted due to the broken import).

---

## 8. Common Pitfalls

- **`src/schemas/common/__init__.py` has a broken import — DO NOT create it.** The reference file imports `from src.schemas.search.hybrid import ChunkResult, HybridSearchRequest, HybridSearchResponse`, but `src.schemas.search.hybrid` does **not exist** in the project. Creating this file will cause an `ImportError` whenever `src.schemas.common` is imported. **Omit the entire `common/` directory.** (The `HybridSearchRequest`/`HybridSearchResponse`/`ChunkResult` names it tries to import are not defined anywhere in the reference schemas.)
- **`AgenticAskResponse` overrides `sources` to `List[Any]`.** It inherits from `AskResponse` but changes the `sources` type from `List[str]` to `List[Any]`. Do not "fix" this to `List[str]` — the agentic endpoint returns rich source objects.
- **`from_` with `alias="from"`.** In `SearchRequest`, `HybridSearchRequest`, and `SearchResponse`, the Python field is `from_` but the JSON field is `from`. Set `populate_by_name = True` (as in `HybridSearchRequest`/`SearchResponse`) so both names are accepted; otherwise only the alias works on input.
- **`PaperCreate.published_date` is a `datetime`, but `ArxivPaper.published_date` is a `str`.** The raw arXiv API schema uses a string (ISO format); the persistence schema uses a parsed `datetime`. Do not conflate them.
- **`PaperResponse` needs `from_attributes = True`** so it can be constructed from an ORM `Paper` object. Omitting this breaks ORM→schema conversion.
- **`PostgreSQLSettings` uses `class Config: env_prefix = "POSTGRES_"`.** It is a `pydantic_settings.BaseSettings`, not a plain `BaseModel`. It reads `POSTGRES_DATABASE_URL`, `POSTGRES_ECHO_SQL`, `POSTGRES_POOL_SIZE`, and `POSTGRES_MAX_OVERFLOW` from the environment.
- **`SearchRequest` is currently unused (dead code).** The reference project defines it but the routers use `HybridSearchRequest` instead. Keep it for completeness, but do not expect it to be wired to an endpoint.
- **`src/schemas/__init__.py` re-exports a subset.** It does not re-export every schema (e.g., `ollama.py` and `database/config.py` are not in the top-level `__all__`). Import those from their specific modules.
- **Do not add `src/schemas/search/`.** The broken `common/__init__.py` references `src.schemas.search.hybrid`, but no such module exists in the reference. Do not invent it.

---

## 9. Definition of Done

- [ ] `src/schemas/api/ask.py` defines `AskRequest`, `AskResponse`, `AgenticAskResponse`, `FeedbackRequest`, and `FeedbackResponse`.
- [ ] `src/schemas/api/health.py` defines `ServiceStatus` and `HealthResponse`.
- [ ] `src/schemas/api/search.py` defines `SearchRequest`, `HybridSearchRequest`, `SearchHit`, and `SearchResponse`.
- [ ] `src/schemas/arxiv/paper.py` defines `ArxivPaper`, `PaperBase`, `PaperCreate`, `PaperResponse`, and `PaperSearchResponse`.
- [ ] `src/schemas/database/config.py` defines `PostgreSQLSettings` (created in PHASE 3).
- [ ] `src/schemas/embeddings/jina.py` defines `JinaEmbeddingRequest` and `JinaEmbeddingResponse`.
- [ ] `src/schemas/indexing/models.py` defines `ChunkMetadata` and `TextChunk`.
- [ ] `src/schemas/pdf_parser/models.py` defines `ParserType`, `PaperSection`, `PaperFigure`, `PaperTable`, `PdfContent`, `ArxivMetadata`, and `ParsedPaper`.
- [ ] `src/schemas/ollama.py` defines `RAGResponse`.
- [ ] All `__init__.py` package markers exist (`api`, `arxiv`, `database`, `embeddings`, `indexing`, `pdf_parser`, `telegram`, and the top-level `src/schemas/__init__.py`).
- [ ] The top-level `src/schemas/__init__.py` re-exports the common schemas.
- [ ] `src/schemas/common/` is **NOT** created (broken import avoided).
- [ ] `uv run python -c "import src.schemas"` and all individual schema modules import without error.
- [ ] `AskRequest`, `AskResponse`, `AgenticAskResponse`, `PaperCreate`, and `HybridSearchRequest` validate correctly (including the `from` alias).
- [ ] No broken imports exist across the schemas package.