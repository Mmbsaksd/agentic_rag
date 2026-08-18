# PHASE 10 — PDF Parsing

> **Build guide for the from-scratch reconstruction of the arXiv Paper Curator (Agentic RAG).**
>
> This phase implements the PDF parsing layer using **Docling**. It provides a [`DoclingParser`](../Agentic-RAG-project-agentops/src/services/pdf_parser/docling.py) that validates PDFs (empty, size, header, page-count checks) and extracts structured sections and raw text, a [`PDFParserService`](../Agentic-RAG-project-agentops/src/services/pdf_parser/parser.py) that wraps the Docling parser behind an async interface, and a factory [`make_pdf_parser_service`](../Agentic-RAG-project-agentops/src/services/pdf_parser/factory.py). It also defines the PDF parsing schemas (`ParserType`, `PaperSection`, `PaperFigure`, `PaperTable`, `PdfContent`, `ArxivMetadata`, `ParsedPaper`) in [`src/schemas/pdf_parser/models.py`](../Agentic-RAG-project-agentops/src/schemas/pdf_parser/models.py).

---

## 1. Phase Objective

By the end of this phase you will have:

- The PDF parsing schemas in [`src/schemas/pdf_parser/models.py`](../Agentic-RAG-project-agentops/src/schemas/pdf_parser/models.py): `ParserType`, `PaperSection`, `PaperFigure`, `PaperTable`, `PdfContent`, `ArxivMetadata`, and `ParsedPaper`.
- A [`DoclingParser`](../Agentic-RAG-project-agentops/src/services/pdf_parser/docling.py) that:
  - Validates a PDF before parsing (empty file, file-size limit, `%PDF-` header, page-count limit).
  - Extracts structured sections (title/section-header driven) and full raw text via Docling's `DocumentConverter`.
  - Returns a `PdfContent` object with `sections`, `raw_text`, `parser_used`, and metadata.
  - Raises typed exceptions (`PDFValidationError`, `PDFParsingException`) on failure, and gracefully returns `None` for size/page-limit skips.
- A [`PDFParserService`](../Agentic-RAG-project-agentops/src/services/pdf_parser/parser.py) that wraps `DoclingParser` behind an `async def parse_pdf` interface.
- A factory [`make_pdf_parser_service`](../Agentic-RAG-project-agentops/src/services/pdf_parser/factory.py) (cached with `@lru_cache(maxsize=1)`) that builds the service from `PDFParserSettings`.
- The `ParserType.DOCLING` enum value used to tag parsed content.

---

## 2. Prerequisites

- Completion of **PHASE 2 (Configuration)** so that `PDFParserSettings` exists in [`src/config.py`](../Agentic-RAG-project-agentops/src/config.py) with the `PDF_PARSER__` env prefix and defaults `max_pages=30`, `max_file_size_mb=20`, `do_ocr=False`, `do_table_structure=True`.
- Completion of **PHASE 4 (Exceptions)** so that `PDFParsingException` and `PDFValidationError` exist in [`src/exceptions.py`](../Agentic-RAG-project-agentops/src/exceptions.py).
- Completion of **PHASE 9 (arXiv Integration)** so that the PDF cache directory (`ARXIV__PDF_CACHE_DIR`) is populated with downloadable PDFs to test against.
- Python 3.11+ and the project's virtual environment active.
- A working internet connection on first run so Docling can download its model artifacts (or pre-cached models).

---

## 3. Dependencies to Install

This phase requires the following third-party packages:

```bash
uv add docling pypdfium2
```

> `docling` is the document-conversion engine (provides `DocumentConverter`, `PdfPipelineOptions`, `PdfFormatOption`, `InputFormat`). `pypdfium2` is used for the lightweight page-count validation (`import pypdfium2 as pdfium`). Both are already declared in the reference project's `pyproject.toml`.

If you are not using `uv`, install with pip:

```bash
pip install docling pypdfium2
```

> **Note:** `docling` pulls in a large dependency tree (including model artifacts). On first use it downloads models (e.g., layout, table-structure models) to a local cache. Ensure you have disk space and network access. `pydantic` (already installed) is used for the schemas.

---

## 4. Directory Structure to Create

```
src/
├── schemas/
│   └── pdf_parser/
│       ├── __init__.py            # (empty package marker)
│       └── models.py              # ParserType, PaperSection, PaperFigure, PaperTable, PdfContent, ArxivMetadata, ParsedPaper
└── services/
    └── pdf_parser/
        ├── docling.py             # DoclingParser
        ├── parser.py              # PDFParserService
        └── factory.py             # make_pdf_parser_service
```

> **Note on `__init__.py`:** In the reference project, `src/services/pdf_parser/` does **not** contain an `__init__.py` (only `docling.py`, `parser.py`, and `factory.py`). The `src/schemas/pdf_parser/` directory **does** contain an `__init__.py`. Create the schema package `__init__.py` as shown; the service directory may omit it (the parser uses a relative import `from .docling import DoclingParser`, which works with implicit namespace packages, but adding an empty `__init__.py` is acceptable and recommended for consistency).

---

## 5. Step-by-Step Implementation

### Step 1 — Define the PDF parsing schemas

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

**Explanation:** `ParserType` is a string enum with a single member `DOCLING = "docling"`. `PaperSection`, `PaperFigure`, and `PaperTable` model the structured elements of a paper. `PdfContent` is the primary output of the parser — it holds `sections`, `figures`, `tables`, `raw_text`, `references`, the `parser_used` tag, and free-form `metadata`. `ArxivMetadata` mirrors the arXiv API fields (note `published_date` is a `str` here, matching `ArxivPaper`). `ParsedPaper` combines arXiv metadata with optional PDF content — this is the object produced by the metadata fetcher in the next phase.

---

### Step 2 — Add the PDF parser schema package `__init__.py`

**Full file path:** `src/schemas/pdf_parser/__init__.py`

```python
```

**Explanation:** This file is intentionally empty in the reference project. It marks `src/schemas/pdf_parser/` as a Python package so the schema module can be imported as `from src.schemas.pdf_parser.models import PdfContent`.

---

### Step 3 — Implement the Docling parser

**Full file path:** `src/services/pdf_parser/docling.py`

```python
import logging
from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from src.exceptions import PDFParsingException, PDFValidationError
from src.schemas.pdf_parser.models import PaperFigure, PaperSection, PaperTable, ParserType, PdfContent

logger = logging.getLogger(__name__)


class DoclingParser:
    """Docling PDF parser for scientific document processing."""

    def __init__(self, max_pages: int, max_file_size_mb: int, do_ocr: bool = False, do_table_structure: bool = True):
        """Initialize DocumentConverter with optimized pipeline options.

        :param max_pages: Maximum number of pages to process
        :param max_file_size_mb: Maximum file size in MB
        :param do_ocr: Enable OCR for scanned PDFs (default: False, very slow)
        :param do_table_structure: Extract table structures (default: True)
        """
        # Configure pipeline options
        pipeline_options = PdfPipelineOptions(
            do_table_structure=do_table_structure,
            do_ocr=do_ocr,  # Usually disabled for speed
        )

        self._converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
        self._warmed_up = False
        self.max_pages = max_pages
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def _warm_up_models(self):
        """Pre-warm the models with a small dummy document to avoid cold start."""
        if not self._warmed_up:
            # This happens only once per DoclingParser instance
            self._warmed_up = True

    def _validate_pdf(self, pdf_path: Path) -> bool:
        """Comprehensive PDF validation including size and page limits.

        :param pdf_path: Path to PDF file
        :returns: True if PDF appears valid and within limits, False otherwise
        """
        try:
            # Check file exists and is not empty
            if pdf_path.stat().st_size == 0:
                logger.error(f"PDF file is empty: {pdf_path}")
                raise PDFValidationError(f"PDF file is empty: {pdf_path}")

            # Check file size limit
            file_size = pdf_path.stat().st_size
            if file_size > self.max_file_size_bytes:
                logger.warning(
                    f"PDF file size ({file_size / 1024 / 1024:.1f}MB) exceeds limit ({self.max_file_size_bytes / 1024 / 1024:.1f}MB), skipping processing"
                )
                raise PDFValidationError(
                    f"PDF file too large: {file_size / 1024 / 1024:.1f}MB > {self.max_file_size_bytes / 1024 / 1024:.1f}MB"
                )

            # Check if file starts with PDF header
            with open(pdf_path, "rb") as f:
                header = f.read(8)
                if not header.startswith(b"%PDF-"):
                    logger.error(f"File does not have PDF header: {pdf_path}")
                    raise PDFValidationError(f"File does not have PDF header: {pdf_path}")

            # Check page count limit
            pdf_doc = pdfium.PdfDocument(str(pdf_path))
            actual_pages = len(pdf_doc)
            pdf_doc.close()

            if actual_pages > self.max_pages:
                logger.warning(
                    f"PDF has {actual_pages} pages, exceeding limit of {self.max_pages} pages. Skipping processing to avoid performance issues."
                )
                raise PDFValidationError(f"PDF has too many pages: {actual_pages} > {self.max_pages}")

            return True

        except PDFValidationError:
            raise
        except Exception as e:
            logger.error(f"Error validating PDF {pdf_path}: {e}")
            raise PDFValidationError(f"Error validating PDF {pdf_path}: {e}")

    def parse_pdf(self, pdf_path: Path) -> Optional[PdfContent]:
        """Parse PDF using Docling parser.
        Limited to 20 pages to avoid memory issues with large papers.

        :param pdf_path: Path to PDF file
        :returns: PdfContent object or None if parsing failed
        """
        try:
            # Validate PDF first (includes size and page limits)
            self._validate_pdf(pdf_path)

            # Warm up models on first use
            self._warm_up_models()

            # Convert PDF using the modern API
            # Limit processing to avoid memory issues with large papers
            result = self._converter.convert(str(pdf_path), max_num_pages=self.max_pages, max_file_size=self.max_file_size_bytes)

            # Extract structured content
            doc = result.document

            # Extract sections from document structure
            sections = []
            current_section = {"title": "Content", "content": ""}

            for element in doc.texts:
                if hasattr(element, "label") and element.label in ["title", "section_header"]:
                    # Save previous section if it has content
                    if current_section["content"].strip():
                        sections.append(PaperSection(title=current_section["title"], content=current_section["content"].strip()))
                    # Start new section
                    current_section = {"title": element.text.strip(), "content": ""}
                else:
                    # Add content to current section
                    if hasattr(element, "text") and element.text:
                        current_section["content"] += element.text + "\n"

            # Add final section
            if current_section["content"].strip():
                sections.append(PaperSection(title=current_section["title"], content=current_section["content"].strip()))

            # Focus on what arXiv API doesn't provide: structured full text content only
            return PdfContent(
                sections=sections,
                figures=[],  # Removed: basic metadata not useful
                tables=[],  # Removed: basic metadata not useful
                raw_text=doc.export_to_text(),
                references=[],
                parser_used=ParserType.DOCLING,
                metadata={"source": "docling", "note": "Content extracted from PDF, metadata comes from arXiv API"},
            )

        except PDFValidationError as e:
            # Handle size/page limit validation errors gracefully by returning None
            error_msg = str(e).lower()
            if "too large" in error_msg or "too many pages" in error_msg:
                logger.info(f"Skipping PDF processing due to size/page limits: {e}")
                return None
            else:
                # Re-raise other validation errors (corrupted files, etc.)
                raise
        except Exception as e:
            logger.error(f"Failed to parse PDF with Docling: {e}")
            logger.error(f"PDF path: {pdf_path}")
            logger.error(f"PDF size: {pdf_path.stat().st_size} bytes")
            logger.error(f"Error type: {type(e).__name__}")

            # Add specific handling for common issues
            error_msg = str(e).lower()

            # Note: Page and size limit checks are now handled in _validate_pdf method

            if "not valid" in error_msg:
                logger.error("PDF appears to be corrupted or not a valid PDF file")
                raise PDFParsingException(f"PDF appears to be corrupted or invalid: {pdf_path}")
            elif "timeout" in error_msg:
                logger.error("PDF processing timed out - file may be too complex")
                raise PDFParsingException(f"PDF processing timed out: {pdf_path}")
            elif "memory" in error_msg or "ram" in error_msg:
                logger.error("Out of memory - PDF may be too large or complex")
                raise PDFParsingException(f"Out of memory processing PDF: {pdf_path}")
            elif "max_num_pages" in error_msg or "page" in error_msg:
                logger.error(f"PDF processing issue likely related to page limits (current limit: {self.max_pages} pages)")
                raise PDFParsingException(
                    f"PDF processing failed, possibly due to page limit ({self.max_pages} pages). Error: {e}"
                )
            else:
                raise PDFParsingException(f"Failed to parse PDF with Docling: {e}")
```

**Explanation:** The `DoclingParser` is the core parsing primitive. Key behaviors:

- **Pipeline configuration:** `PdfPipelineOptions(do_table_structure=..., do_ocr=...)` is passed to `DocumentConverter` via `PdfFormatOption`. OCR is disabled by default for speed.
- **Validation (`_validate_pdf`):** Checks the file is non-empty, within the size limit, starts with the `%PDF-` magic header, and has an acceptable page count (using `pypdfium2`). Any failure raises `PDFValidationError`.
- **Section extraction:** Iterates `doc.texts`, treating elements labeled `title` or `section_header` as section boundaries. Content accumulates into the current section until the next header, then is flushed into a `PaperSection`.
- **Graceful skips:** In `parse_pdf`, a `PDFValidationError` whose message contains `too large` or `too many pages` is converted to a `return None` (a soft skip), while other validation errors are re-raised.
- **Typed failures:** Other parse failures are mapped to `PDFParsingException` with specific messages for corrupted files, timeouts, out-of-memory, and page-limit issues.

---

### Step 4 — Implement the PDF parser service

**Full file path:** `src/services/pdf_parser/parser.py`

```python
import logging
from pathlib import Path
from typing import Optional

from src.exceptions import PDFParsingException, PDFValidationError
from src.schemas.pdf_parser.models import PdfContent

from .docling import DoclingParser

logger = logging.getLogger(__name__)


class PDFParserService:
    """Main PDF parsing service using Docling only."""

    def __init__(self, max_pages: int, max_file_size_mb: int, do_ocr: bool = False, do_table_structure: bool = True):
        """Initialize PDF parser service with configurable limits."""
        self.docling_parser = DoclingParser(
            max_pages=max_pages, max_file_size_mb=max_file_size_mb, do_ocr=do_ocr, do_table_structure=do_table_structure
        )

    async def parse_pdf(self, pdf_path: Path) -> Optional[PdfContent]:
        """Parse PDF using Docling parser only.

        :param pdf_path: Path to PDF file
        :returns: PdfContent object or None if parsing failed
        """
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            raise PDFValidationError(f"PDF file not found: {pdf_path}")

        try:
            result = self.docling_parser.parse_pdf(pdf_path)
            if result:
                logger.info(f"Parsed {pdf_path.name}")
                return result
            else:
                logger.error(f"Docling parsing returned no result for {pdf_path.name}")
                raise PDFParsingException(f"Docling parsing returned no result for {pdf_path.name}")

        except (PDFValidationError, PDFParsingException):
            raise
        except Exception as e:
            logger.error(f"Docling parsing error for {pdf_path.name}: {e}")
            raise PDFParsingException(f"Docling parsing error for {pdf_path.name}: {e}")
```

**Explanation:** `PDFParserService` is the async facade over `DoclingParser`. It checks the file exists (raising `PDFValidationError` if not), delegates to the Docling parser, and re-raises typed exceptions while wrapping any unexpected error in `PDFParsingException`. This is the service the metadata fetcher (next phase) will call via `await pdf_parser.parse_pdf(path)`.

---

### Step 5 — Add the PDF parser factory

**Full file path:** `src/services/pdf_parser/factory.py`

```python
from functools import lru_cache

from src.config import get_settings

from .parser import PDFParserService


@lru_cache(maxsize=1)
def make_pdf_parser_service() -> PDFParserService:
    """Factory function to create a PDF parser service instance.

    :returns: An instance of the PDF parser service
    :rtype: PDFParserService
    """
    settings = get_settings()

    # Create PDF parser service with explicit settings
    service = PDFParserService(
        max_pages=settings.pdf_parser.max_pages,
        max_file_size_mb=settings.pdf_parser.max_file_size_mb,
        do_ocr=settings.pdf_parser.do_ocr,
        do_table_structure=settings.pdf_parser.do_table_structure,
    )

    return service
```

**Explanation:** `make_pdf_parser_service` is the composition root for the PDF parser. It is decorated with `@lru_cache(maxsize=1)` so the (relatively expensive) `DocumentConverter` and its model artifacts are constructed only once per process. It reads the nested `settings.pdf_parser` block (a `PDFParserSettings` instance) and passes its values into the `PDFParserService` constructor.

---

## 6. Configuration

The PDF parser is driven entirely by the `PDFParserSettings` block defined in [`src/config.py`](../Agentic-RAG-project-agentops/src/config.py). These settings use the `PDF_PARSER__` environment prefix and the `__` nested delimiter:

| Environment variable | Default | Description |
| --- | --- | --- |
| `PDF_PARSER__MAX_PAGES` | `30` | Maximum number of pages to process per PDF. |
| `PDF_PARSER__MAX_FILE_SIZE_MB` | `20` | Maximum PDF file size in MB. |
| `PDF_PARSER__DO_OCR` | `False` | Enable OCR for scanned PDFs (slow; keep off). |
| `PDF_PARSER__DO_TABLE_STRUCTURE` | `True` | Extract table structures. |

**Example `.env` snippet:**

```dotenv
PDF_PARSER__MAX_PAGES=30
PDF_PARSER__MAX_FILE_SIZE_MB=20
PDF_PARSER__DO_OCR=false
PDF_PARSER__DO_TABLE_STRUCTURE=true
```

> The PDFs themselves come from the arXiv cache directory configured in PHASE 9 (`ARXIV__PDF_CACHE_DIR`, default `./data/arxiv_pdfs`). The parser does not download PDFs — it only reads files from disk.

---

## 7. Verification

Verify the PDF parsing layer works end-to-end before wiring it into the ingestion flow.

### 7.1 Schema smoke test

Confirm the schemas import and validate correctly:

```bash
python -c "
from src.schemas.pdf_parser.models import ParserType, PdfContent, PaperSection, ParsedPaper, ArxivMetadata

content = PdfContent(
    sections=[PaperSection(title='Introduction', content='Hello world.')],
    raw_text='Hello world.',
    parser_used=ParserType.DOCLING,
)
print('PdfContent OK:', content.parser_used, len(content.sections))
"
```

**Expected:** The `PdfContent` instance is created without validation errors and prints `PdfContent OK: docling 1`.

### 7.2 Parse a PDF from the cache

First ensure a PDF exists in the cache (from PHASE 9), then parse it:

```bash
python -c "
import asyncio
from pathlib import Path
from src.services.pdf_parser.factory import make_pdf_parser_service

async def main():
    service = make_pdf_parser_service()
    cache_dir = Path('./data/arxiv_pdfs')
    pdfs = list(cache_dir.glob('*.pdf'))
    if not pdfs:
        print('No PDFs in cache. Run PHASE 9 download first.')
        return
    path = pdfs[0]
    print(f'Parsing {path.name}...')
    result = await service.parse_pdf(path)
    if result:
        print(f'Sections: {len(result.sections)}')
        print(f'Raw text length: {len(result.raw_text)}')
        print(f'Parser: {result.parser_used}')
    else:
        print('Parse returned None (skipped due to limits)')

asyncio.run(main())
"
```

**Expected:** The first PDF in the cache is parsed. On first run, Docling downloads its model artifacts (this may take a while). The output shows a non-zero section count, a non-zero raw-text length, and `Parser: docling`.

### 7.3 Verify validation rejects a bad file

```bash
python -c "
import asyncio
from pathlib import Path
from src.services.pdf_parser.factory import make_pdf_parser_service
from src.exceptions import PDFValidationError

async def main():
    service = make_pdf_parser_service()
    bad = Path('./data/not_a_pdf.txt')
    bad.write_text('this is not a pdf')
    try:
        await service.parse_pdf(bad)
        print('ERROR: should have raised')
    except PDFValidationError as e:
        print(f'Correctly rejected: {e}')
    finally:
        bad.unlink()

asyncio.run(main())
"
```

**Expected:** The parser raises `PDFValidationError` because the file does not start with the `%PDF-` header.

### 7.4 Run the unit tests

The reference project includes unit tests for the PDF parser. Run them to confirm the validation and parsing logic behaves correctly:

```bash
python -m pytest tests/unit/services/test_pdf_parser.py -v
```

**Expected:** All tests in `test_pdf_parser.py` pass.

---

## 8. Common Pitfalls

- **First-run model download.** Docling downloads layout/table-structure models on first use. If the environment has no network access or insufficient disk space, `parse_pdf` will fail. Pre-warm the models (or run a single parse) during setup to avoid cold-start failures in production.

- **OCR is very slow.** Keep `PDF_PARSER__DO_OCR=false`. Enabling OCR on scanned PDFs dramatically increases parse time and memory. The reference project deliberately disables it.

- **Page and size limits.** `_validate_pdf` enforces `max_pages` (30) and `max_file_size_mb` (20). Papers exceeding these are **soft-skipped** (return `None`), not raised. Do not treat a `None` result as an error — the metadata fetcher counts these as skipped.

- **`pypdfium2` page-count check.** The page-count validation opens the PDF with `pdfium.PdfDocument`. If `pypdfium2` is not installed or the PDF is corrupt, this raises, which `_validate_pdf` wraps into `PDFValidationError`. Ensure `pypdfium2` is in dependencies.

- **`doc.texts` label attribute.** Section extraction relies on `element.label` being `title` or `section_header`. If Docling's model does not detect headers (e.g., for scanned or unusual layouts), all text collapses into a single `Content` section. This is expected behavior, not a bug.

- **`raw_text` vs `sections`.** `raw_text` is the full document text via `doc.export_to_text()`, while `sections` is the structured breakdown. Both are stored; downstream indexing (PHASE 8) consumes the structured content. Do not drop either.

- **Async wrapper is thin.** `PDFParserService.parse_pdf` is `async` but the underlying Docling call is blocking. This is intentional in the reference project — the concurrency control happens at the metadata-fetcher level (via `asyncio.Semaphore`), not here.

---

## 9. Definition of Done

This phase is complete when:

- [ ] `src/schemas/pdf_parser/models.py` defines `ParserType`, `PaperSection`, `PaperFigure`, `PaperTable`, `PdfContent`, `ArxivMetadata`, and `ParsedPaper` exactly as specified.
- [ ] `src/schemas/pdf_parser/__init__.py` exists (empty package marker).
- [ ] `src/services/pdf_parser/docling.py` implements the full `DoclingParser` with `_validate_pdf`, `_warm_up_models`, and `parse_pdf`.
- [ ] `src/services/pdf_parser/parser.py` implements `PDFParserService` with an `async def parse_pdf` interface.
- [ ] `src/services/pdf_parser/factory.py` implements `make_pdf_parser_service()` decorated with `@lru_cache(maxsize=1)`.
- [ ] `docling` and `pypdfium2` are added to the project dependencies.
- [ ] The `PDF_PARSER__*` environment variables are documented and honored by the parser.
- [ ] Verification steps 7.1–7.3 pass: schemas validate, a cached PDF parses into sections/raw text, and a non-PDF file is rejected with `PDFValidationError`.
- [ ] Unit tests in `tests/unit/services/test_pdf_parser.py` pass.
- [ ] The parser raises `PDFValidationError` / `PDFParsingException` on the appropriate failure paths and returns `None` for size/page-limit skips.
