# PHASE 8 — Indexing

> **Build guide for the from-scratch reconstruction of the arXiv Paper Curator (Agentic RAG).**
>
> This phase implements the indexing pipeline that prepares papers for hybrid search. It provides a [`TextChunker`](../src/services/indexing/text_chunker.py) that splits papers into overlapping, section-aware chunks, and a [`HybridIndexingService`](../src/services/indexing/hybrid_indexer.py) that orchestrates **chunk → embed → bulk index** into OpenSearch. It also defines the chunk schemas and the factory that wires everything together.

---

## 1. Phase Objective

By the end of this phase you will have:

- A [`TextChunker`](../src/services/indexing/text_chunker.py) that chunks papers using a hybrid section-based strategy (default 600 words per chunk, 100-word overlap).
- Pydantic schemas [`ChunkMetadata`](../src/schemas/indexing/models.py) and [`TextChunk`](../src/schemas/indexing/models.py).
- A [`HybridIndexingService`](../src/services/indexing/hybrid_indexer.py) that chunks a paper, generates embeddings for each chunk, and bulk-indexes them into OpenSearch.
- A factory [`make_hybrid_indexing_service`](../src/services/indexing/factory.py) that assembles the chunker, embeddings client, and OpenSearch client.
- Support for single-paper indexing, batch indexing, and reindexing.

---

## 2. Prerequisites

- Completion of **PHASE 6 (Embeddings)** so that the `JinaEmbeddingsClient` and `make_embeddings_client` exist.
- Completion of **PHASE 7 (OpenSearch)** so that the `OpenSearchClient`, `make_opensearch_client_fresh`, and the hybrid index (`arxiv-papers-chunks`) exist.
- Completion of **PHASE 2 (Configuration)** so that `ChunkingSettings` (chunk_size, overlap_size, min_chunk_size) exist in [`src/config.py`](../src/config.py).
- The `src/schemas/indexing/` and `src/schemas/pdf_parser/` package directories exist (from PHASE 4).
- A running OpenSearch cluster with the hybrid index created (via `setup_indices` from PHASE 7).
- Python 3.11+ and the project's virtual environment active.

---

## 3. Dependencies to Install

No new third-party dependencies are required for this phase. It builds on:

```bash
# Already installed in prior phases
pydantic
opensearch-py
httpx
```

> The indexing service composes the existing embeddings client (PHASE 6) and OpenSearch client (PHASE 7).

---

## 4. Directory Structure to Create

```
src/
├── schemas/
│   ├── indexing/
│   │   ├── __init__.py            # (empty package marker)
│   │   └── models.py              # ChunkMetadata, TextChunk
│   └── pdf_parser/
│       ├── __init__.py            # (empty package marker)
│       └── models.py              # PaperSection, PdfContent, ParsedPaper, etc.
└── services/
    └── indexing/
        ├── __init__.py            # (empty package marker)
        ├── factory.py             # make_hybrid_indexing_service
        ├── hybrid_indexer.py      # HybridIndexingService
        └── text_chunker.py        # TextChunker
```

---

## 5. Step-by-Step Implementation

### Step 1 — Define the chunk schemas

**Full file path:** `src/schemas/indexing/models.py`

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

**Explanation:** `ChunkMetadata` captures the position and overlap information for a chunk (index, character offsets, word count, overlaps, and optional section title). `TextChunk` bundles the chunk text with its metadata and the paper identifiers (`arxiv_id`, `paper_id`).

---

### Step 2 — Add the indexing schema package `__init__.py`

**Full file path:** `src/schemas/indexing/__init__.py`

```python
```

**Explanation:** This file is intentionally empty in the reference project. It marks the directory as a Python package.

---

### Step 3 — Define the PDF parser schemas (referenced by the chunker)

**Full file path:** `src/schemas/pdf_parser/models.py`

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

**Explanation:** These schemas model the output of the PDF parser (Docling) and arXiv metadata. `PdfContent` carries the structured `sections` (list of `PaperSection`) that the `TextChunker` uses for section-based chunking, plus `raw_text` for the fallback path. `ParsedPaper` combines arXiv metadata with optional PDF content.

---

### Step 4 — Add the PDF parser schema package `__init__.py`

**Full file path:** `src/schemas/pdf_parser/__init__.py`

```python
```

**Explanation:** This file is intentionally empty in the reference project. It marks the directory as a Python package.

---

### Step 5 — Implement the text chunker

**Full file path:** `src/services/indexing/text_chunker.py`

```python
import json
import logging
import re
from typing import Dict, List, Optional, Union

from src.schemas.indexing.models import ChunkMetadata, TextChunk

logger = logging.getLogger(__name__)


class TextChunker:
    """Service for chunking text into overlapping segments.

    Uses word-based chunking with configurable chunk size and overlap.
    Default: 600 words per chunk with 100 word overlap.
    """

    def __init__(self, chunk_size: int = 600, overlap_size: int = 100, min_chunk_size: int = 100):
        """Initialize text chunker.

        :param chunk_size: Target number of words per chunk
        :param overlap_size: Number of overlapping words between chunks
        :param min_chunk_size: Minimum words for a chunk to be valid
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size

        if overlap_size >= chunk_size:
            raise ValueError("Overlap size must be less than chunk size")

        logger.info(
            f"Text chunker initialized: chunk_size={chunk_size}, overlap_size={overlap_size}, min_chunk_size={min_chunk_size}"
        )

    def _split_into_words(self, text: str) -> List[str]:
        """Split text into words while preserving whitespace information.

        :param text: Input text
        :returns: List of words
        """
        # Split on whitespace while keeping the words
        words = re.findall(r"\S+", text)
        return words

    def _reconstruct_text(self, words: List[str]) -> str:
        """Reconstruct text from words.

        :param words: List of words
        :returns: Reconstructed text
        """
        return " ".join(words)

    def chunk_paper(
        self,
        title: str,
        abstract: str,
        full_text: str,
        arxiv_id: str,
        paper_id: str,
        sections: Optional[Union[Dict[str, str], str, list]] = None,
    ) -> List[TextChunk]:
        """Chunk a paper using hybrid section-based approach.

        Strategy:
        - For sections 100-800 words: Use as single chunk with title+abstract
        - For sections <100 words: Combine with adjacent sections
        - For sections >800 words: Split using traditional word-based chunking
        - Fallback to traditional chunking if no sections available

        :param title: Paper title
        :param abstract: Paper abstract
        :param full_text: Full text content
        :param arxiv_id: ArXiv ID of the paper
        :param paper_id: Database ID of the paper
        :param sections: Dictionary or JSON string of sections
        :returns: List of text chunks with metadata
        """
        # Try section-based chunking first
        if sections:
            try:
                section_chunks = self._chunk_by_sections(title, abstract, arxiv_id, paper_id, sections)
                if section_chunks:
                    logger.info(f"Created {len(section_chunks)} section-based chunks for {arxiv_id}")
                    return section_chunks
            except Exception as e:
                logger.warning(f"Section-based chunking failed for {arxiv_id}: {e}")

        # Fallback to traditional word-based chunking
        logger.info(f"Using traditional word-based chunking for {arxiv_id}")
        return self.chunk_text(full_text, arxiv_id, paper_id)

    def chunk_text(self, text: str, arxiv_id: str, paper_id: str) -> List[TextChunk]:
        """Chunk text into overlapping segments.

        :param text: Full text to chunk
        :param arxiv_id: ArXiv ID of the paper
        :param paper_id: Database ID of the paper
        :returns: List of text chunks with metadata
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for paper {arxiv_id}")
            return []

        # Split text into words
        words = self._split_into_words(text)

        if len(words) < self.min_chunk_size:
            logger.warning(f"Text for paper {arxiv_id} has only {len(words)} words, less than minimum {self.min_chunk_size}")
            # Return single chunk if text is too small
            if words:
                return [
                    TextChunk(
                        text=self._reconstruct_text(words),
                        metadata=ChunkMetadata(
                            chunk_index=0,
                            start_char=0,
                            end_char=len(text),
                            word_count=len(words),
                            overlap_with_previous=0,
                            overlap_with_next=0,
                        ),
                        arxiv_id=arxiv_id,
                        paper_id=paper_id,
                    )
                ]
            return []

        chunks = []
        chunk_index = 0
        current_position = 0

        while current_position < len(words):
            # Calculate chunk boundaries
            chunk_start = current_position
            chunk_end = min(current_position + self.chunk_size, len(words))

            # Extract chunk words
            chunk_words = words[chunk_start:chunk_end]
            chunk_text = self._reconstruct_text(chunk_words)

            # Calculate character offsets (approximate)
            start_char = len(" ".join(words[:chunk_start])) if chunk_start > 0 else 0
            end_char = len(" ".join(words[:chunk_end]))

            # Calculate overlaps
            overlap_with_previous = min(self.overlap_size, chunk_start) if chunk_start > 0 else 0
            overlap_with_next = self.overlap_size if chunk_end < len(words) else 0

            # Create chunk
            chunk = TextChunk(
                text=chunk_text,
                metadata=ChunkMetadata(
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    word_count=len(chunk_words),
                    overlap_with_previous=overlap_with_previous,
                    overlap_with_next=overlap_with_next,
                    section_title=None,  # Could be enhanced with section detection
                ),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
            )
            chunks.append(chunk)

            # Move to next chunk position (with overlap)
            current_position += self.chunk_size - self.overlap_size
            chunk_index += 1

            # Break if we've processed all words
            if chunk_end >= len(words):
                break

        logger.info(f"Chunked paper {arxiv_id}: {len(words)} words -> {len(chunks)} chunks")

        return chunks

    def _chunk_by_sections(
        self, title: str, abstract: str, arxiv_id: str, paper_id: str, sections: Union[Dict[str, str], str, list]
    ) -> List[TextChunk]:
        """Implement hybrid section-based chunking strategy.

        :param title: Paper title
        :param abstract: Paper abstract
        :param arxiv_id: ArXiv ID
        :param paper_id: Database ID
        :param sections: Sections data
        :returns: List of text chunks
        """
        # Parse sections data
        sections_dict = self._parse_sections(sections)
        if not sections_dict:
            return []

        # Filter and clean sections
        sections_dict = self._filter_sections(sections_dict, abstract)
        if not sections_dict:
            logger.warning(f"No meaningful sections found after filtering for {arxiv_id}")
            return []

        # Create header (title + abstract)
        header = f"{title}\n\nAbstract: {abstract}\n\n"

        # Process sections using hybrid strategy
        chunks = []
        small_sections = []  # Buffer for combining small sections

        section_items = list(sections_dict.items())

        for i, (section_title, section_content) in enumerate(section_items):
            content_str = str(section_content) if section_content else ""
            section_words = len(content_str.split())

            if section_words < 100:
                # Collect small sections to combine later
                small_sections.append((section_title, content_str, section_words))

                # If this is the last section or next section is large, process accumulated small sections
                if i == len(section_items) - 1 or len(str(section_items[i + 1][1]).split()) >= 100:
                    chunks.extend(self._create_combined_chunk(header, small_sections, chunks, arxiv_id, paper_id))
                    small_sections = []

            elif 100 <= section_words <= 800:
                # Perfect size - create single chunk
                chunk_text = f"{header}Section: {section_title}\n\n{content_str}"
                chunk = self._create_section_chunk(chunk_text, section_title, len(chunks), arxiv_id, paper_id)
                chunks.append(chunk)

            else:
                # Large section - split using traditional chunking
                section_text = f"Section: {section_title}\n\n{content_str}"
                full_section_text = f"{header}{section_text}"

                # Use traditional chunking but with section context
                section_chunks = self._split_large_section(
                    full_section_text, header, section_title, len(chunks), arxiv_id, paper_id
                )
                chunks.extend(section_chunks)

        return chunks

    def _parse_sections(self, sections: Union[Dict[str, str], str, list]) -> Dict[str, str]:
        """Parse sections data into a dictionary."""
        if isinstance(sections, dict):
            return sections
        elif isinstance(sections, list):
            # Handle list of sections directly
            result = {}
            for i, section in enumerate(sections):
                if isinstance(section, dict):
                    title = section.get("title", section.get("heading", f"Section {i + 1}"))
                    content = section.get("content", section.get("text", ""))
                    result[title] = content
                else:
                    result[f"Section {i + 1}"] = str(section)
            return result
        elif isinstance(sections, str):
            try:
                parsed = json.loads(sections)
                if isinstance(parsed, dict):
                    return parsed
                elif isinstance(parsed, list):
                    # Convert list to dict with enumerated keys
                    result = {}
                    for i, section in enumerate(parsed):
                        if isinstance(section, dict):
                            title = section.get("title", section.get("heading", f"Section {i + 1}"))
                            content = section.get("content", section.get("text", ""))
                            result[title] = content
                        else:
                            result[f"Section {i + 1}"] = str(section)
                    return result
            except json.JSONDecodeError:
                logger.warning("Failed to parse sections JSON")
        return {}

    def _filter_sections(self, sections_dict: Dict[str, str], abstract: str) -> Dict[str, str]:
        """Filter out unwanted sections and avoid duplication.

        :param sections_dict: Dictionary of sections
        :param abstract: Paper abstract for duplication check
        :returns: Filtered dictionary of sections
        """
        filtered = {}
        abstract_words = set(abstract.lower().split())

        for section_title, section_content in sections_dict.items():
            content_str = str(section_content).strip()

            # Skip empty sections
            if not content_str:
                continue

            # Skip metadata/header sections based on title
            if self._is_metadata_section(section_title):
                continue

            # Skip sections that are duplicates of the abstract
            if self._is_duplicate_abstract(content_str, abstract, abstract_words):
                logger.debug(f"Skipping duplicate abstract section: {section_title}")
                continue

            # Skip sections that are too small and contain only metadata
            if len(content_str.split()) < 20 and self._is_metadata_content(content_str):
                logger.debug(f"Skipping metadata section: {section_title}")
                continue

            filtered[section_title] = content_str

        return filtered

    def _is_metadata_section(self, section_title: str) -> bool:
        """Check if a section title indicates metadata/header content."""
        title_lower = section_title.lower().strip()

        metadata_indicators = [
            "content",
            "header",
            "authors",
            "author",
            "affiliation",
            "email",
            "arxiv",
            "preprint",
            "submitted",
            "received",
            "accepted",
        ]

        # Exact matches or very short titles that are likely metadata
        if title_lower in metadata_indicators or len(title_lower) < 5:
            return True

        # Check if title contains only metadata indicators
        for indicator in metadata_indicators:
            if indicator in title_lower and len(title_lower) < 20:
                return True

        return False

    def _is_duplicate_abstract(self, content: str, abstract: str, abstract_words: set) -> bool:
        """Check if section content is a duplicate of the abstract."""
        content_lower = content.lower().strip()
        abstract_lower = abstract.lower().strip()

        # Direct string match (allowing for minor formatting differences)
        if abstract_lower in content_lower or content_lower in abstract_lower:
            return True

        # Word overlap check - if >80% of words overlap, likely duplicate
        content_words = set(content_lower.split())

        if len(abstract_words) > 10:  # Only check for substantial abstracts
            overlap = len(abstract_words.intersection(content_words))
            overlap_ratio = overlap / len(abstract_words)

            if overlap_ratio > 0.8:
                return True

        return False

    def _is_metadata_content(self, content: str) -> bool:
        """Check if content contains only metadata (emails, arxiv IDs, etc.)."""
        content_lower = content.lower()

        # Check for common metadata patterns
        metadata_patterns = [
            "@",  # Email addresses
            "arxiv:",  # ArXiv IDs
            "university",
            "institute",
            "department",
            "college",
            "gmail.com",
            "edu",
            "ac.uk",
            "preprint",
        ]

        # If content is mostly metadata patterns
        word_count = len(content.split())
        if word_count < 30:  # Short content
            metadata_word_count = sum(1 for pattern in metadata_patterns if pattern in content_lower)
            if metadata_word_count >= 2:  # Contains multiple metadata indicators
                return True

        return False

    def _create_combined_chunk(
        self, header: str, small_sections: List, existing_chunks: List, arxiv_id: str, paper_id: str
    ) -> List[TextChunk]:
        """Create chunks by combining small sections."""
        if not small_sections:
            return []

        # Combine all small sections
        combined_content = []
        total_words = 0

        for section_title, content, word_count in small_sections:
            combined_content.append(f"Section: {section_title}\n\n{content}")
            total_words += word_count

        combined_text = f"{header}{'\n\n'.join(combined_content)}"

        # If still too small, combine with previous chunk if possible
        if total_words + len(header.split()) < 200 and existing_chunks:
            # Try to merge with previous chunk
            prev_chunk = existing_chunks[-1]
            merged_text = f"{prev_chunk.text}\n\n{'\n\n'.join(combined_content)}"

            # Update the previous chunk
            existing_chunks[-1] = TextChunk(
                text=merged_text,
                metadata=ChunkMetadata(
                    chunk_index=prev_chunk.metadata.chunk_index,
                    start_char=0,
                    end_char=len(merged_text),
                    word_count=len(merged_text.split()),
                    overlap_with_previous=0,
                    overlap_with_next=0,
                    section_title=f"{prev_chunk.metadata.section_title} + Combined",
                ),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
            )
            return []

        # Create new chunk with combined content
        sections_titles = [title for title, _, _ in small_sections]
        combined_title = " + ".join(sections_titles[:3])  # Limit title length
        if len(sections_titles) > 3:
            combined_title += f" + {len(sections_titles) - 3} more"

        chunk = self._create_section_chunk(combined_text, combined_title, len(existing_chunks), arxiv_id, paper_id)
        return [chunk]

    def _create_section_chunk(
        self, chunk_text: str, section_title: str, chunk_index: int, arxiv_id: str, paper_id: str
    ) -> TextChunk:
        """Create a single section-based chunk."""
        return TextChunk(
            text=chunk_text,
            metadata=ChunkMetadata(
                chunk_index=chunk_index,
                start_char=0,
                end_char=len(chunk_text),
                word_count=len(chunk_text.split()),
                overlap_with_previous=0,
                overlap_with_next=0,
                section_title=section_title,
            ),
            arxiv_id=arxiv_id,
            paper_id=paper_id,
        )

    def _split_large_section(
        self, full_section_text: str, header: str, section_title: str, base_chunk_index: int, arxiv_id: str, paper_id: str
    ) -> List[TextChunk]:
        """Split large sections using traditional word-based chunking."""
        # Remove header from section text for chunking, then add back to each chunk
        section_only = full_section_text[len(header):]

        # Use traditional chunking on section content
        traditional_chunks = self.chunk_text(section_only, arxiv_id, paper_id)

        # Add header to each chunk and update metadata
        enhanced_chunks = []
        for i, chunk in enumerate(traditional_chunks):
            enhanced_text = f"{header}{chunk.text}"

            enhanced_chunk = TextChunk(
                text=enhanced_text,
                metadata=ChunkMetadata(
                    chunk_index=base_chunk_index + i,
                    start_char=chunk.metadata.start_char,
                    end_char=chunk.metadata.end_char + len(header),
                    word_count=len(enhanced_text.split()),
                    overlap_with_previous=chunk.metadata.overlap_with_previous,
                    overlap_with_next=chunk.metadata.overlap_with_next,
                    section_title=f"{section_title} (Part {i + 1})",
                ),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
            )
            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks
```

**Explanation:** The `TextChunker` implements a hybrid section-based chunking strategy:
- **Sections 100–800 words** become a single chunk (prefixed with title + abstract header).
- **Sections <100 words** are buffered and combined with adjacent small sections (or merged into the previous chunk if still too small).
- **Sections >800 words** are split using traditional word-based chunking, with the header re-added to each part.
- **No sections** falls back to `chunk_text`, which slides a 600-word window with a 100-word overlap.
It also filters out metadata/header sections and abstract duplicates to avoid redundant chunks.

---

### Step 6 — Implement the hybrid indexing service

**Full file path:** `src/services/indexing/hybrid_indexer.py`

```python
import logging
from typing import Dict, List, Optional

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient

from .text_chunker import TextChunker

logger = logging.getLogger(__name__)


class HybridIndexingService:
    """Service for indexing papers with chunking and embeddings for hybrid search.

    Orchestrates the process of:
    1. Chunking papers into overlapping segments
    2. Generating embeddings for each chunk
    3. Indexing chunks with embeddings into OpenSearch
    """

    def __init__(self, chunker: TextChunker, embeddings_client: JinaEmbeddingsClient, opensearch_client: OpenSearchClient):
        """Initialize hybrid indexing service.

        :param chunker: Text chunking service
        :param embeddings_client: Embeddings generation client
        :param opensearch_client: OpenSearch client
        """
        self.chunker = chunker
        self.embeddings_client = embeddings_client
        self.opensearch_client = opensearch_client

        logger.info("Hybrid indexing service initialized")

    async def index_paper(self, paper_data: Dict) -> Dict[str, int]:
        """Index a single paper with chunking and embeddings.

        :param paper_data: Paper data from database
        :returns: Dictionary with indexing statistics
        """
        arxiv_id = paper_data.get("arxiv_id")
        paper_id = str(paper_data.get("id", ""))

        if not arxiv_id:
            logger.error("Paper missing arxiv_id")
            return {"chunks_created": 0, "chunks_indexed": 0, "embeddings_generated": 0, "errors": 1}

        try:
            # Step 1: Chunk the paper using hybrid section-based approach
            chunks = self.chunker.chunk_paper(
                title=paper_data.get("title", ""),
                abstract=paper_data.get("abstract", ""),
                full_text=paper_data.get("raw_text", paper_data.get("full_text", "")),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
                sections=paper_data.get("sections"),
            )

            if not chunks:
                logger.warning(f"No chunks created for paper {arxiv_id}")
                return {"chunks_created": 0, "chunks_indexed": 0, "embeddings_generated": 0, "errors": 0}

            logger.info(f"Created {len(chunks)} chunks for paper {arxiv_id}")

            # Step 2: Generate embeddings for chunks
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = await self.embeddings_client.embed_passages(
                texts=chunk_texts,
                batch_size=50,  # Process in batches
            )

            if len(embeddings) != len(chunks):
                logger.error(f"Embedding count mismatch: {len(embeddings)} != {len(chunks)}")
                return {"chunks_created": len(chunks), "chunks_indexed": 0, "embeddings_generated": len(embeddings), "errors": 1}

            # Step 3: Prepare chunks with embeddings for indexing
            chunks_with_embeddings = []

            for chunk, embedding in zip(chunks, embeddings):
                # Prepare chunk data for OpenSearch
                chunk_data = {
                    "arxiv_id": chunk.arxiv_id,
                    "paper_id": chunk.paper_id,
                    "chunk_index": chunk.metadata.chunk_index,
                    "chunk_text": chunk.text,
                    "chunk_word_count": chunk.metadata.word_count,
                    "start_char": chunk.metadata.start_char,
                    "end_char": chunk.metadata.end_char,
                    "section_title": chunk.metadata.section_title,
                    "embedding_model": "jina-embeddings-v3",
                    # Denormalized paper metadata for efficient search
                    "title": paper_data.get("title", ""),
                    "authors": ", ".join(paper_data.get("authors", []))
                    if isinstance(paper_data.get("authors"), list)
                    else paper_data.get("authors", ""),
                    "abstract": paper_data.get("abstract", ""),
                    "categories": paper_data.get("categories", []),
                    "published_date": paper_data.get("published_date"),
                }

                chunks_with_embeddings.append({"chunk_data": chunk_data, "embedding": embedding})

            # Step 4: Index chunks into OpenSearch
            results = self.opensearch_client.bulk_index_chunks(chunks_with_embeddings)

            logger.info(f"Indexed paper {arxiv_id}: {results['success']} chunks successful, {results['failed']} failed")

            return {
                "chunks_created": len(chunks),
                "chunks_indexed": results["success"],
                "embeddings_generated": len(embeddings),
                "errors": results["failed"],
            }

        except Exception as e:
            logger.error(f"Error indexing paper {arxiv_id}: {e}")
            return {"chunks_created": 0, "chunks_indexed": 0, "embeddings_generated": 0, "errors": 1}

    async def index_papers_batch(self, papers: List[Dict], replace_existing: bool = False) -> Dict[str, int]:
        """Index multiple papers in batch.

        :param papers: List of paper data
        :param replace_existing: If True, delete existing chunks before indexing
        :returns: Aggregated statistics
        """
        total_stats = {
            "papers_processed": 0,
            "total_chunks_created": 0,
            "total_chunks_indexed": 0,
            "total_embeddings_generated": 0,
            "total_errors": 0,
        }

        for paper in papers:
            arxiv_id = paper.get("arxiv_id")

            # Optionally delete existing chunks
            if replace_existing and arxiv_id:
                self.opensearch_client.delete_paper_chunks(arxiv_id)

            # Index the paper
            stats = await self.index_paper(paper)

            # Update totals
            total_stats["papers_processed"] += 1
            total_stats["total_chunks_created"] += stats["chunks_created"]
            total_stats["total_chunks_indexed"] += stats["chunks_indexed"]
            total_stats["total_embeddings_generated"] += stats["embeddings_generated"]
            total_stats["total_errors"] += stats["errors"]

        logger.info(
            f"Batch indexing complete: {total_stats['papers_processed']} papers, "
            f"{total_stats['total_chunks_indexed']} chunks indexed"
        )

        return total_stats

    async def reindex_paper(self, arxiv_id: str, paper_data: Dict) -> Dict[str, int]:
        """Reindex a paper by deleting old chunks and creating new ones.

        :param arxiv_id: ArXiv ID of the paper
        :param paper_data: Updated paper data
        :returns: Indexing statistics
        """
        # Delete existing chunks
        deleted = self.opensearch_client.delete_paper_chunks(arxiv_id)
        if deleted:
            logger.info(f"Deleted existing chunks for paper {arxiv_id}")

        # Index with new data
        return await self.index_paper(paper_data)
```

**Explanation:** `HybridIndexingService` orchestrates the full pipeline:
1. **Chunk** the paper via `chunker.chunk_paper` (section-based with fallback).
2. **Embed** each chunk via `embeddings_client.embed_passages` (batch size 50).
3. **Prepare** OpenSearch documents with denormalized paper metadata (title, authors, abstract, categories, published_date) plus `embedding_model`.
4. **Bulk index** via `opensearch_client.bulk_index_chunks`.
It returns per-paper statistics (`chunks_created`, `chunks_indexed`, `embeddings_generated`, `errors`). `index_papers_batch` aggregates across many papers (optionally deleting existing chunks first), and `reindex_paper` deletes then re-indexes a single paper.

---

### Step 7 — Add the indexing factory

**Full file path:** `src/services/indexing/factory.py`

```python
from typing import Optional

from src.config import Settings, get_settings
from src.services.embeddings.factory import make_embeddings_client
from src.services.opensearch.factory import make_opensearch_client_fresh

from .hybrid_indexer import HybridIndexingService
from .text_chunker import TextChunker


def make_hybrid_indexing_service(
    settings: Optional[Settings] = None, opensearch_host: Optional[str] = None
) -> HybridIndexingService:
    """Factory function to create hybrid indexing service.

    Creates a new service instance each time.

    :param settings: Optional settings instance
    :param opensearch_host: Optional OpenSearch host override
    :returns: HybridIndexingService instance
    """
    if settings is None:
        settings = get_settings()

    # Create dependencies using configuration
    chunker = TextChunker(
        chunk_size=settings.chunking.chunk_size,
        overlap_size=settings.chunking.overlap_size,
        min_chunk_size=settings.chunking.min_chunk_size,
    )
    embeddings_client = make_embeddings_client(settings)
    opensearch_client = make_opensearch_client_fresh(settings, host=opensearch_host)

    # Create indexing service
    return HybridIndexingService(
        chunker=chunker,
        embeddings_client=embeddings_client,
        opensearch_client=opensearch_client,
    )
```

**Explanation:** `make_hybrid_indexing_service` is the composition root for the indexing pipeline. It:
1. Falls back to `get_settings()` when no settings are passed.
2. Builds a `TextChunker` from `settings.chunking` (chunk size, overlap, min chunk size).
3. Reuses `make_embeddings_client` (Jina) and `make_opensearch_client_fresh` (a fresh client so it can accept a host override).
4. Returns a fully wired `HybridIndexingService`.

Note: unlike the OpenSearch factory, this factory deliberately creates a **fresh** OpenSearch client each call (no `lru_cache`) so that a custom `opensearch_host` can be supplied per invocation.

---

### Step 8 — Create the indexing package `__init__.py`

**Full file path:** `src/services/indexing/__init__.py`

```python
```

**Explanation:** This file is intentionally empty. It marks `src/services/indexing/` as a Python package so that the modules (`text_chunker`, `hybrid_indexer`, `factory`) can be imported with relative imports (e.g., `from .hybrid_indexer import HybridIndexingService`). No re-exports are required at this stage because the factory is the public entry point.

---

## 6. Configuration

The indexing pipeline is driven entirely by the `ChunkingSettings` block defined in [`src/config.py`](Agentic-RAG-project-agentops/src/config.py:70). These settings use the `CHUNKING__` environment prefix and the `__` nested delimiter, so they are configured as follows:

| Environment variable | Default | Description |
| --- | --- | --- |
| `CHUNKING__CHUNK_SIZE` | `600` | Target number of words per chunk. |
| `CHUNKING__OVERLAP_SIZE` | `100` | Number of overlapping words between adjacent chunks. |
| `CHUNKING__MIN_CHUNK_SIZE` | `100` | Minimum word count for a standalone chunk; smaller sections are combined. |
| `CHUNKING__SECTION_BASED` | `true` | When `true`, chunk by document sections; when `false`, fall back to plain sliding-window chunking. |

These map directly to the `TextChunker` constructor arguments in the factory:

```python
chunker = TextChunker(
    chunk_size=settings.chunking.chunk_size,
    overlap_size=settings.chunking.overlap_size,
    min_chunk_size=settings.chunking.min_chunk_size,
)
```

**Example `.env` snippet:**

```dotenv
CHUNKING__CHUNK_SIZE=600
CHUNKING__OVERLAP_SIZE=100
CHUNKING__MIN_CHUNK_SIZE=100
CHUNKING__SECTION_BASED=true
```

The indexing service also depends on the **OpenSearch** and **Embeddings** settings from earlier phases (the index name, vector dimension, and Jina API key), since `HybridIndexingService` writes chunks into the hybrid OpenSearch index and calls the Jina embeddings API.

---

## 7. Verification

Verify the indexing pipeline works end-to-end before wiring it into the ingestion flow.

### 7.1 Chunker smoke test

Run a quick Python snippet to confirm section-based chunking produces sensible chunks:

```bash
python -c "
from src.config import get_settings
from src.services.indexing.text_chunker import TextChunker

settings = get_settings()
chunker = TextChunker(
    chunk_size=settings.chunking.chunk_size,
    overlap_size=settings.chunking.overlap_size,
    min_chunk_size=settings.chunking.min_chunk_size,
)

text = 'Introduction\n\nThis paper introduces a novel method for retrieval augmented generation.\n\n' * 20
chunks = chunker.chunk_text(text, arxiv_id='2401.00001', paper_id='1')
print(f'Created {len(chunks)} chunks')
for c in chunks[:3]:
    print(f'  - {c.metadata.chunk_index}: {len(c.text.split())} words, section={c.metadata.section_title}')
"
```

**Expected:** A handful of chunks are created, each with a `chunk_index`, word count, and (where applicable) a `section_title`. No exceptions should be raised.

### 7.2 Indexing service test

With OpenSearch and the Jina API key configured, index a single paper:

```bash
python -c "
import asyncio
from src.config import get_settings
from src.services.indexing.factory import make_hybrid_indexing_service

async def main():
    service = make_hybrid_indexing_service(get_settings())
    paper = {
        'arxiv_id': '2401.00001',
        'paper_id': '1',
        'title': 'Test Paper',
        'authors': ['Alice', 'Bob'],
        'abstract': 'A short abstract about retrieval augmented generation.',
        'categories': ['cs.CL'],
        'published_date': '2024-01-01',
        'content': 'Introduction\n\nThis is the body of the paper with enough words to produce several chunks for indexing.\n\n' * 30,
    }
    result = await service.index_paper(paper)
    print('Indexing result:', result)

asyncio.run(main())
"
```

**Expected:** The result dict shows `chunks_created`, `chunks_indexed`, and `embeddings_generated` all greater than zero, with `errors` equal to zero.

### 7.3 Confirm chunks are searchable

Query the OpenSearch index to confirm the chunks were written:

```bash
python -c "
from src.config import get_settings
from src.services.opensearch.factory import make_opensearch_client

client = make_opensearch_client(get_settings())
stats = client.get_index_stats()
print('Index stats:', stats)
chunks = client.get_chunks_by_paper('2401.00001')
print(f'Found {len(chunks)} chunks for paper')
"
```

**Expected:** The index stats show a non-zero document count, and `get_chunks_by_paper` returns the chunks created in step 7.2.

---

## 8. Common Pitfalls

1. **Embedding dimension mismatch.** The `embedding` field in the OpenSearch mapping is a 1024-dimension `knn_vector`. If the Jina model returns a different dimension (e.g., because a different model name or `dimensions` override is used), indexing will fail with a mapper parsing exception. Keep the model at `jina-embeddings-v3` with the default 1024 dimensions.

2. **Strict dynamic mapping.** The hybrid index uses `dynamic: strict`, so any field not declared in the mapping causes a rejection. Ensure the document dict passed to `bulk_index_chunks` contains **only** the fields defined in `ARXIV_PAPERS_CHUNKS_MAPPING` (e.g., `chunk_id`, `arxiv_id`, `paper_id`, `chunk_index`, `chunk_text`, `chunk_word_count`, `start_char`, `end_char`, `title`, `authors`, `abstract`, `categories`, `published_date`, `section_title`, `embedding`, `embedding_model`, `created_at`, `updated_at`).

3. **Section-based chunking fallback.** If `section_based` is `true` but the parsed content has no usable sections, `chunk_paper` falls back to plain sliding-window chunking. Do not assume every paper produces section-tagged chunks — the fallback path is expected behavior for some PDFs.

4. **Small sections get combined.** Sections under `min_chunk_size` words are combined with adjacent content (via `_create_combined_chunk`) rather than dropped. This preserves information but means chunk boundaries do not always align with section boundaries.

5. **Batch size for embeddings.** `embed_passages` batches requests (default 100) to avoid overwhelming the Jina API. The indexing service uses a batch size of 50. If you hit rate limits, reduce the batch size rather than retrying the whole paper.

6. **Fresh OpenSearch client per call.** `make_hybrid_indexing_service` uses `make_opensearch_client_fresh`, not the cached `make_opensearch_client`. This is intentional so a custom host can be supplied, but it means each call opens a new connection. Do not call the factory in a hot loop.

7. **`reindex_paper` deletes first.** Re-indexing deletes existing chunks for the `arxiv_id` before writing new ones. If the delete fails (e.g., the index does not exist yet), the re-index will error — ensure `setup_indices` has been run first.

8. **Missing `paper_id` / `arxiv_id`.** Both fields are required in the chunk document and used for lookups (`get_chunks_by_paper`, `delete_paper_chunks`). Ensure the paper data passed to `index_paper` always includes both.

---

## 9. Definition of Done

- [ ] `src/services/indexing/text_chunker.py` exists and implements `TextChunker` with configurable `chunk_size`, `overlap_size`, and `min_chunk_size`.
- [ ] `TextChunker.chunk_paper` and `chunk_text` return `List[TextChunk]` with populated `ChunkMetadata` (chunk index, char offsets, word counts, overlap flags, optional section title).
- [ ] Section-based chunking is implemented with a plain sliding-window fallback, and metadata sections / duplicate abstracts are filtered out.
- [ ] `src/services/indexing/hybrid_indexer.py` exists and implements `HybridIndexingService` with `index_paper`, `index_papers_batch`, and `reindex_paper`.
- [ ] `HybridIndexingService` chunks → embeds → bulk-indexes into the hybrid OpenSearch index and returns per-paper statistics.
- [ ] `src/services/indexing/factory.py` exists and `make_hybrid_indexing_service` wires the chunker, embeddings client, and a fresh OpenSearch client from settings.
- [ ] `src/services/indexing/__init__.py` exists (empty package marker).
- [ ] `src/schemas/indexing/models.py` defines `ChunkMetadata` and `TextChunk`.
- [ ] `src/schemas/pdf_parser/models.py` defines `ParserType`, `PaperSection`, `PaperFigure`, `PaperTable`, `PdfContent`, `ArxivMetadata`, and `ParsedPaper`.
- [ ] `CHUNKING__*` environment variables are documented and honored by the factory.
- [ ] The chunker smoke test, indexing service test, and OpenSearch chunk lookup all pass.
- [ ] Indexed chunks are searchable via the hybrid OpenSearch index with correct 1024-dimension embeddings.
