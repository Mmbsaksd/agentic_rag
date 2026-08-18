# PHASE 11 — Metadata Fetcher

> **Build guide for the from-scratch reconstruction of the arXiv Paper Curator (Agentic RAG).**
>
> This phase implements the ingestion orchestration layer. It provides a [`MetadataFetcher`](../Agentic-RAG-project-agentops/src/services/metadata_fetcher.py) that fetches paper metadata from arXiv, downloads and parses PDFs concurrently (using `asyncio.Semaphore` for controlled concurrency), and stores the combined results to the database via `PaperRepository.upsert`. It also provides a factory [`make_metadata_fetcher`](../Agentic-RAG-project-agentops/src/services/metadata_fetcher.py) that wires the arXiv client, PDF parser, and settings together.

---

## 1. Phase Objective

By the end of this phase you will have:

- A [`MetadataFetcher`](../Agentic-RAG-project-agentops/src/services/metadata_fetcher.py) that:
  - Fetches paper metadata from arXiv via `ArxivClient.fetch_papers` (with optional date filtering).
  - Downloads and parses PDFs concurrently with separate download/parse semaphores.
  - Combines arXiv metadata and PDF content into `ParsedPaper` objects.
  - Serializes parsed content and stores papers to the database via `PaperRepository.upsert`.
  - Returns a results dictionary with statistics (`papers_fetched`, `pdfs_downloaded`, `pdfs_parsed`, `papers_stored`, `papers_indexed`, `errors`, `processing_time`).
- An overlapping download+parse pipeline (`_process_pdfs_batch` + `_download_and_parse_pipeline`) that starts parsing as soon as each download completes, while other downloads continue.
- A factory [`make_metadata_fetcher`](../Agentic-RAG-project-agentops/src/services/metadata_fetcher.py) that builds the fetcher from the arXiv client, PDF parser, and settings.
- Typed exception handling (`MetadataFetchingException`, `PipelineException`) for pipeline failures.

---

## 2. Prerequisites

- Completion of **PHASE 9 (arXiv Integration)** so that `ArxivClient` and `make_arxiv_client` exist.
- Completion of **PHASE 10 (PDF Parsing)** so that `PDFParserService` and `make_pdf_parser_service` exist.
- Completion of **PHASE 3 (Database Layer)** so that `PaperRepository` (with `upsert`) exists in [`src/repositories/paper.py`](../Agentic-RAG-project-agentops/src/repositories/paper.py) and `Session` is available.
- Completion of **PHASE 4 (Exceptions)** so that `MetadataFetchingException` and `PipelineException` exist in [`src/exceptions.py`](../Agentic-RAG-project-agentops/src/exceptions.py).
- Completion of **PHASE 4 (Schemas)** so that `ArxivPaper`, `PaperCreate`, `ArxivMetadata`, `ParsedPaper`, and `PdfContent` exist.
- Python 3.11+ and the project's virtual environment active.

---

## 3. Dependencies to Install

This phase requires the following third-party package:

```bash
uv add python-dateutil
```

> `python-dateutil` provides `dateutil.parser`, used to convert the arXiv `published_date` string into a `datetime` before database storage. `sqlalchemy` (already installed in PHASE 3) provides the `Session` type. No other new dependencies are required.

If you are not using `uv`, install with pip:

```bash
pip install python-dateutil
```

---

## 4. Directory Structure to Create

This phase adds a single new file to the existing service layer:

```
src/
└── services/
    └── metadata_fetcher.py        # MetadataFetcher + make_metadata_fetcher
```

> No new directories are required. The `MetadataFetcher` depends on services and schemas created in PHASES 9 and 10, and on `PaperRepository` from PHASE 3.

---

## 5. Step-by-Step Implementation

### Step 1 — Implement the metadata fetcher

**Full file path:** `src/services/metadata_fetcher.py`

```python
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dateutil import parser as date_parser
from sqlalchemy.orm import Session
from src.config import Settings
from src.exceptions import MetadataFetchingException, PipelineException
from src.repositories.paper import PaperRepository
from src.schemas.arxiv.paper import ArxivPaper, PaperCreate
from src.schemas.pdf_parser.models import ArxivMetadata, ParsedPaper, PdfContent
from src.services.arxiv.client import ArxivClient
from src.services.opensearch.client import OpenSearchClient
from src.services.pdf_parser.parser import PDFParserService

logger = logging.getLogger(__name__)


class MetadataFetcher:
    """Service for fetching arXiv papers with PDF processing and database storage."""

    def __init__(
        self,
        arxiv_client: ArxivClient,
        pdf_parser: PDFParserService,
        pdf_cache_dir: Optional[Path] = None,
        max_concurrent_downloads: int = 5,
        max_concurrent_parsing: int = 3,
        settings: Optional[Settings] = None,
    ):
        """Initialize metadata fetcher with services and settings.

        :param arxiv_client: Client for arXiv API operations
        :param pdf_parser: Service for parsing PDF documents
        :param opensearch_client: Optional OpenSearch client for indexing
        :param pdf_cache_dir: Directory for caching downloaded PDFs
        :param max_concurrent_downloads: Maximum concurrent PDF downloads
        :param max_concurrent_parsing: Maximum concurrent PDF parsing operations
        :param settings: Application settings instance
        :type arxiv_client: ArxivClient
        :type pdf_parser: PDFParserService
        :type opensearch_client: Optional[OpenSearchClient]
        :type pdf_cache_dir: Optional[Path]
        :type max_concurrent_downloads: int
        :type max_concurrent_parsing: int
        :type settings: Optional[Settings]
        """
        from src.config import get_settings

        self.arxiv_client = arxiv_client
        self.pdf_parser = pdf_parser
        self.pdf_cache_dir = pdf_cache_dir or self.arxiv_client.pdf_cache_dir
        self.max_concurrent_downloads = max_concurrent_downloads
        self.max_concurrent_parsing = max_concurrent_parsing
        self.settings = settings or get_settings()

    async def fetch_and_process_papers(
        self,
        max_results: Optional[int] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        process_pdfs: bool = True,
        store_to_db: bool = True,
        db_session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """Fetch papers from arXiv, process PDFs, and store to database.

        :param max_results: Maximum papers to fetch
        :param from_date: Filter papers from this date (YYYYMMDD)
        :param to_date: Filter papers to this date (YYYYMMDD)
        :param process_pdfs: Whether to download and parse PDFs
        :param store_to_db: Whether to store results in database
        :param db_session: Database session (required if store_to_db=True)
        :type max_results: Optional[int]
        :type from_date: Optional[str]
        :type to_date: Optional[str]
        :type process_pdfs: bool
        :type store_to_db: bool
        :type db_session: Optional[Session]
        :returns: Dictionary with processing results and statistics
        :rtype: Dict[str, Any]
        """

        results = {
            "papers_fetched": 0,
            "pdfs_downloaded": 0,
            "pdfs_parsed": 0,
            "papers_stored": 0,
            "papers_indexed": 0,
            "errors": [],
            "processing_time": 0,
        }

        start_time = datetime.now()

        try:
            # Step 1: Fetch paper metadata from arXiv
            papers = await self.arxiv_client.fetch_papers(
                max_results=max_results, from_date=from_date, to_date=to_date, sort_by="submittedDate", sort_order="descending"
            )

            results["papers_fetched"] = len(papers)

            if not papers:
                logger.warning("No papers found")
                return results

            # Step 2: Process PDFs if requested
            pdf_results = {}
            if process_pdfs:
                pdf_results = await self._process_pdfs_batch(papers)
                results["pdfs_downloaded"] = pdf_results["downloaded"]
                results["pdfs_parsed"] = pdf_results["parsed"]
                results["errors"].extend(pdf_results["errors"])

            # Step 3: Store to database if requested
            if store_to_db and db_session:
                logger.info("Step 3: Storing papers to database...")
                stored_count = self._store_papers_to_db(papers, pdf_results.get("parsed_papers", {}), db_session)
                results["papers_stored"] = stored_count
            elif store_to_db:
                logger.warning("Database storage requested but no session provided")
                results["errors"].append("Database session not provided for storage")

            # Calculate total processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            results["processing_time"] = processing_time

            # Simple logging summary
            logger.info(
                f"Pipeline completed in {processing_time:.1f}s: {results['papers_fetched']} papers, {results['pdfs_downloaded']} PDFs, {len(results['errors'])} errors"
            )

            if results["errors"]:
                logger.warning("Errors summary:")
                for i, error in enumerate(results["errors"][:5], 1):  # Show first 5 errors
                    logger.warning(f"  {i}. {error}")
                if len(results["errors"]) > 5:
                    logger.warning(f"  ... and {len(results['errors']) - 5} more errors")

            return results

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            results["errors"].append(f"Pipeline error: {str(e)}")
            raise PipelineException(f"Pipeline execution failed: {e}") from e

    async def _process_pdfs_batch(self, papers: List[ArxivPaper]) -> Dict[str, Any]:
        """
        Process PDFs for a batch of papers with async concurrency.

        Uses overlapping download+parse pipeline:
        - Downloads happen concurrently (up to max_concurrent_downloads)
        - As each download completes, parsing starts immediately
        - Multiple PDFs can be parsing while others are still downloading

        This is optimal for production workloads like 100 papers/day.

        Args:
            papers: List of ArxivPaper objects

        Returns:
            Dictionary with processing results and statistics
        """
        results = {
            "downloaded": 0,
            "parsed": 0,
            "parsed_papers": {},
            "errors": [],
            "download_failures": [],
            "parse_failures": [],
        }

        logger.info(f"Starting async pipeline for {len(papers)} PDFs...")
        logger.info(f"Concurrent downloads: {self.max_concurrent_downloads}")
        logger.info(f"Concurrent parsing: {self.max_concurrent_parsing}")

        # Create semaphores for controlled concurrency
        download_semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        parse_semaphore = asyncio.Semaphore(self.max_concurrent_parsing)

        # Start all download+parse pipelines concurrently
        pipeline_tasks = [self._download_and_parse_pipeline(paper, download_semaphore, parse_semaphore) for paper in papers]

        # Wait for all pipelines to complete
        pipeline_results = await asyncio.gather(*pipeline_tasks, return_exceptions=True)

        # Process results with detailed error tracking
        for paper, result in zip(papers, pipeline_results):
            if isinstance(result, Exception):
                error_msg = f"Pipeline error for {paper.arxiv_id}: {str(result)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
            elif result:
                # Check if result is a tuple before unpacking
                # Handle AirflowTaskTerminated and other non-tuple results
                if isinstance(result, tuple) and len(result) == 2:
                    # Result is tuple: (download_success, parsed_paper)
                    download_success, parsed_paper = result
                else:
                    # Result is not a tuple (could be AirflowTaskTerminated or other error)
                    error_msg = f"Pipeline error for {paper.arxiv_id}: Unexpected result type {type(result).__name__}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
                    continue

                if download_success:
                    results["downloaded"] += 1

                    if parsed_paper:
                        results["parsed"] += 1
                        results["parsed_papers"][paper.arxiv_id] = parsed_paper
                    else:
                        # Download succeeded but parsing failed
                        results["parse_failures"].append(paper.arxiv_id)
                else:
                    # Download failed
                    results["download_failures"].append(paper.arxiv_id)
            else:
                # No result returned (shouldn't happen but handle gracefully)
                results["download_failures"].append(paper.arxiv_id)

        # Simple processing summary
        logger.info(f"PDF processing: {results['downloaded']}/{len(papers)} downloaded, {results['parsed']} parsed")

        if results["download_failures"]:
            logger.warning(f"Download failures: {len(results['download_failures'])}")

        if results["parse_failures"]:
            logger.warning(f"Parse failures: {len(results['parse_failures'])}")

        # Add specific failure info to general errors list for backward compatibility
        if results["download_failures"]:
            results["errors"].extend([f"Download failed: {arxiv_id}" for arxiv_id in results["download_failures"]])
        if results["parse_failures"]:
            results["errors"].extend([f"PDF parse failed: {arxiv_id}" for arxiv_id in results["parse_failures"]])

        return results

    async def _download_and_parse_pipeline(
        self, paper: ArxivPaper, download_semaphore: asyncio.Semaphore, parse_semaphore: asyncio.Semaphore
    ) -> tuple:
        """
        Complete download+parse pipeline for a single paper with true parallelism.
        Downloads PDF, then immediately starts parsing while other downloads continue.

        Returns:
            Tuple of (download_success: bool, parsed_paper: Optional[ParsedPaper])
        """
        download_success = False
        parsed_paper = None

        try:
            # Step 1: Download PDF with download concurrency control
            async with download_semaphore:
                logger.debug(f"Starting download: {paper.arxiv_id}")
                pdf_path = await self.arxiv_client.download_pdf(paper, False)

                if pdf_path:
                    download_success = True
                    logger.debug(f"Download complete: {paper.arxiv_id}")
                else:
                    logger.error(f"Download failed: {paper.arxiv_id}")
                    return (False, None)

            # Step 2: Parse PDF with parse concurrency control (happens AFTER download completes)
            # This allows other downloads to continue while this PDF is being parsed
            async with parse_semaphore:
                logger.debug(f"Starting parse: {paper.arxiv_id}")
                pdf_content = await self.pdf_parser.parse_pdf(pdf_path)

                if pdf_content:
                    # Create ArxivMetadata from the paper
                    arxiv_metadata = ArxivMetadata(
                        title=paper.title,
                        authors=paper.authors,
                        abstract=paper.abstract,
                        arxiv_id=paper.arxiv_id,
                        categories=paper.categories,
                        published_date=paper.published_date,
                        pdf_url=paper.pdf_url,
                    )

                    # Combine into ParsedPaper
                    parsed_paper = ParsedPaper(arxiv_metadata=arxiv_metadata, pdf_content=pdf_content)
                    logger.debug(f"Parse complete: {paper.arxiv_id} - {len(pdf_content.raw_text)} chars extracted")
                else:
                    # PDF parsing failed, but this is not critical - we can continue with metadata only
                    logger.warning(f"PDF parsing failed for {paper.arxiv_id}, continuing with metadata only")

        except Exception as e:
            logger.error(f"Pipeline error for {paper.arxiv_id}: {e}")
            raise MetadataFetchingException(f"Pipeline error for {paper.arxiv_id}: {e}") from e

        return (download_success, parsed_paper)

    def _serialize_parsed_content(self, parsed_paper: ParsedPaper) -> Dict[str, Any]:
        """Serialize ParsedPaper content for database storage.

        :param parsed_paper: ParsedPaper object with PDF content
        :type parsed_paper: ParsedPaper
        :returns: Dictionary with serialized content for database storage
        :rtype: Dict[str, Any]
        """
        try:
            pdf_content = parsed_paper.pdf_content

            # Serialize sections
            sections = [{"title": section.title, "content": section.content} for section in pdf_content.sections]

            # Serialize references
            references = list(pdf_content.references)  #

            return {
                "raw_text": pdf_content.raw_text,
                "sections": sections,
                "references": references,
                "parser_used": pdf_content.parser_used.value if pdf_content.parser_used else None,
                "parser_metadata": pdf_content.metadata or {},
                "pdf_processed": True,
                "pdf_processing_date": datetime.now(),
            }
        except Exception as e:
            logger.error(f"Failed to serialize parsed content: {e}")
            return {"pdf_processed": False, "parser_metadata": {"error": str(e)}}

    def _store_papers_to_db(
        self,
        papers: List[ArxivPaper],
        parsed_papers: Dict[str, ParsedPaper],
        db_session: Session,
    ) -> int:
        """
        Store papers and parsed content to database with comprehensive content storage.

        Args:
            papers: List of ArxivPaper metadata
            parsed_papers: Dictionary of parsed PDF content by arxiv_id
            db_session: Database session

        Returns:
            Number of papers stored successfully
        """
        paper_repo = PaperRepository(db_session)
        stored_count = 0

        for paper in papers:
            try:
                # Get parsed content if available
                parsed_paper = parsed_papers.get(paper.arxiv_id)

                # Base paper data
                published_date = (
                    date_parser.parse(paper.published_date) if isinstance(paper.published_date, str) else paper.published_date
                )
                paper_data = {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract,
                    "categories": paper.categories,
                    "published_date": published_date,
                    "pdf_url": paper.pdf_url,
                }

                # Add parsed content if available
                if parsed_paper:
                    parsed_content = self._serialize_parsed_content(parsed_paper)
                    paper_data.update(parsed_content)
                    logger.debug(
                        f"Storing paper {paper.arxiv_id} with parsed content ({len(parsed_content.get('raw_text', '')) if parsed_content.get('raw_text') else 0} chars)"
                    )
                else:
                    # No parsed content - just store metadata
                    paper_data.update(
                        {"pdf_processed": False, "parser_metadata": {"note": "PDF processing not available or failed"}}
                    )
                    logger.debug(f"Storing paper {paper.arxiv_id} with metadata only")

                paper_create = PaperCreate(**paper_data)
                stored_paper = paper_repo.upsert(paper_create)

                if stored_paper:
                    stored_count += 1
                    content_info = "with parsed content" if parsed_paper else "metadata only"
                    logger.debug(f"Stored paper {paper.arxiv_id} to database ({content_info})")

            except Exception as e:
                logger.error(f"Failed to store paper {paper.arxiv_id}: {e}")

        # Commit all changes
        try:
            db_session.commit()
            logger.info(f"Committed {stored_count} papers to database with full content storage")
        except Exception as e:
            logger.error(f"Failed to commit papers to database: {e}")
            db_session.rollback()
            stored_count = 0

        return stored_count


def make_metadata_fetcher(
    arxiv_client: ArxivClient,
    pdf_parser: PDFParserService,
    pdf_cache_dir: Optional[Path] = None,
    settings: Optional[Settings] = None,
) -> MetadataFetcher:
    """Create MetadataFetcher instance with configuration settings.

    :param arxiv_client: Client for arXiv API operations
    :param pdf_parser: Service for parsing PDF documents
    :param pdf_cache_dir: Directory for caching downloaded PDFs
    :param settings: Application settings instance (uses default if None)
    :type arxiv_client: ArxivClient
    :type pdf_parser: PDFParserService
    :type pdf_cache_dir: Optional[Path]
    :type settings: Optional[Settings]
    :returns: Configured MetadataFetcher instance
    :rtype: MetadataFetcher
    """
    from src.config import get_settings

    if settings is None:
        settings = get_settings()

    return MetadataFetcher(
        arxiv_client=arxiv_client,
        pdf_parser=pdf_parser,
        pdf_cache_dir=pdf_cache_dir,
        max_concurrent_downloads=settings.arxiv.max_concurrent_downloads,
        max_concurrent_parsing=settings.arxiv.max_concurrent_parsing,
        settings=settings,
    )
```

**Explanation:** The `MetadataFetcher` is the ingestion orchestrator. Key behaviors:

- **`fetch_and_process_papers`** is the main entry point. It runs three steps: (1) fetch metadata from arXiv, (2) process PDFs concurrently, (3) store to the database. It returns a results dict with statistics and collects errors without aborting the whole pipeline (unless an unexpected top-level exception occurs, which is wrapped in `PipelineException`).
- **`_process_pdfs_batch`** creates two `asyncio.Semaphore` instances (`download_semaphore` and `parse_semaphore`) and launches one `_download_and_parse_pipeline` task per paper via `asyncio.gather(..., return_exceptions=True)`. It then tallies downloads, parses, and failures, and appends failure summaries to the general `errors` list.
- **`_download_and_parse_pipeline`** acquires the download semaphore, downloads the PDF, then (after releasing the download semaphore) acquires the parse semaphore and parses it. This overlapping design lets other downloads proceed while one PDF is being parsed. It returns a `(download_success, parsed_paper)` tuple. A failed parse still returns `(True, None)` so the paper can be stored with metadata only.
- **`_serialize_parsed_content`** converts a `ParsedPaper`'s `PdfContent` into a plain dict (`raw_text`, `sections`, `references`, `parser_used`, `parser_metadata`, `pdf_processed`, `pdf_processing_date`) for database storage.
- **`_store_papers_to_db`** builds a `PaperCreate` per paper (converting `published_date` from `str` to `datetime` via `dateutil.parser`), merges parsed content when available, calls `PaperRepository.upsert`, and commits once at the end.
- **`make_metadata_fetcher`** is the composition root, wiring the arXiv client, PDF parser, and concurrency settings (`settings.arxiv.max_concurrent_downloads` / `max_concurrent_parsing`) into a `MetadataFetcher`.

---

## 6. Configuration

The metadata fetcher does not introduce new settings of its own. It reuses the concurrency settings already defined in the `ArxivSettings` block (from PHASE 9):

| Environment variable | Default | Used for |
| --- | --- | --- |
| `ARXIV__MAX_CONCURRENT_DOWNLOADS` | `5` | `max_concurrent_downloads` — max simultaneous PDF downloads. |
| `ARXIV__MAX_CONCURRENT_PARSING` | `1` | `max_concurrent_parsing` — max simultaneous PDF parses. |

These are read by `make_metadata_fetcher` and passed into the `MetadataFetcher` constructor. The PDF cache directory defaults to `self.arxiv_client.pdf_cache_dir` (from `ARXIV__PDF_CACHE_DIR`).

> No `.env` changes are required beyond what PHASES 9 and 10 already configure. If you want tighter or looser concurrency, adjust `ARXIV__MAX_CONCURRENT_DOWNLOADS` and `ARXIV__MAX_CONCURRENT_PARSING`.

---

## 7. Verification

Verify the metadata fetcher works end-to-end before wiring it into the API or scheduling layer.

### 7.1 Fetch and process papers without database storage

This exercises the arXiv fetch and PDF processing pipeline without requiring a database session:

```bash
python -c "
import asyncio
from src.services.arxiv.factory import make_arxiv_client
from src.services.pdf_parser.factory import make_pdf_parser_service
from src.services.metadata_fetcher import make_metadata_fetcher

async def main():
    arxiv_client = make_arxiv_client()
    pdf_parser = make_pdf_parser_service()
    fetcher = make_metadata_fetcher(arxiv_client=arxiv_client, pdf_parser=pdf_parser)

    results = await fetcher.fetch_and_process_papers(
        max_results=3,
        process_pdfs=True,
        store_to_db=False,
    )
    print('Papers fetched:', results['papers_fetched'])
    print('PDFs downloaded:', results['pdfs_downloaded'])
    print('PDFs parsed:', results['pdfs_parsed'])
    print('Errors:', len(results['errors']))
    print('Processing time:', round(results['processing_time'], 1), 's')

asyncio.run(main())
"
```

**Expected:** A small number of papers are fetched, their PDFs are downloaded and parsed (subject to concurrency limits), and the results dict shows the counts. Errors may be non-zero if some PDFs fail to download/parse — this is expected and non-fatal.

### 7.2 Run the unit tests

The reference project includes unit tests for the metadata fetcher. Run them to confirm the pipeline logic behaves correctly:

```bash
python -m pytest tests/unit/services/test_metadata_fetcher.py -v
```

**Expected:** All tests in `test_metadata_fetcher.py` pass.

### 7.3 Full pipeline with database storage (optional)

If a database is available (from PHASE 3), verify the full fetch → parse → store flow:

```bash
python -c "
import asyncio
from src.services.arxiv.factory import make_arxiv_client
from src.services.pdf_parser.factory import make_pdf_parser_service
from src.services.metadata_fetcher import make_metadata_fetcher
from src.dependencies import get_db_session

async def main():
    arxiv_client = make_arxiv_client()
    pdf_parser = make_pdf_parser_service()
    fetcher = make_metadata_fetcher(arxiv_client=arxiv_client, pdf_parser=pdf_parser)

    # Use a session from the DI container
    session_gen = get_db_session()
    db_session = next(session_gen)
    try:
        results = await fetcher.fetch_and_process_papers(
            max_results=2,
            process_pdfs=True,
            store_to_db=True,
            db_session=db_session,
        )
        print('Papers stored:', results['papers_stored'])
    finally:
        db_session.close()

asyncio.run(main())
"
```

**Expected:** Papers (with parsed content where available) are upserted into the database, and `papers_stored` reflects the count.

---

## 8. Common Pitfalls

- **`db_session` is required for storage.** If `store_to_db=True` but `db_session` is `None`, the fetcher logs a warning and appends `"Database session not provided for storage"` to `errors` — it does not raise. Pass a real session when you expect persistence.

- **`published_date` type conversion.** `ArxivPaper.published_date` is a `str`; `_store_papers_to_db` converts it to `datetime` via `dateutil.parser.parse`. If `python-dateutil` is not installed, this import fails at module load. Ensure it is in dependencies.

- **Semaphore concurrency limits.** `max_concurrent_downloads` and `max_concurrent_parsing` come from `ARXIV__MAX_CONCURRENT_DOWNLOADS` (5) and `ARXIV__MAX_CONCURRENT_PARSING` (1). With parsing concurrency of 1, PDFs are parsed serially even though downloads overlap. Raise `ARXIV__MAX_CONCURRENT_PARSING` if you want more parallel parsing (at the cost of memory).

- **`asyncio.gather(..., return_exceptions=True)`.** Pipeline tasks return exceptions rather than raising, so `_process_pdfs_batch` must check `isinstance(result, Exception)` and handle non-tuple results (e.g., `AirflowTaskTerminated`). Do not assume every result is a 2-tuple.

- **Partial failures are non-fatal.** A failed download or parse does not abort the batch. The fetcher records the failure in `download_failures` / `parse_failures` and continues. Only an unexpected top-level exception raises `PipelineException`.

- **`_serialize_parsed_content` assumes `pdf_content` is not None.** It is only called when `parsed_paper` is truthy (i.e., has `pdf_content`). If you call it directly with a `ParsedPaper` lacking `pdf_content`, it will raise `AttributeError` and return a fallback dict with `pdf_processed: False`.

- **Commit is batched.** `_store_papers_to_db` commits once at the end. If any single paper fails to store, it is logged and skipped, but the successful ones are still committed together. On commit failure, it rolls back and returns `0`.

---

## 9. Definition of Done

This phase is complete when:

- [ ] `src/services/metadata_fetcher.py` defines the `MetadataFetcher` class with `fetch_and_process_papers`, `_process_pdfs_batch`, `_download_and_parse_pipeline`, `_serialize_parsed_content`, and `_store_papers_to_db`.
- [ ] `src/services/metadata_fetcher.py` defines the `make_metadata_fetcher` factory that wires the arXiv client, PDF parser, and concurrency settings.
- [ ] `python-dateutil` is added to the project dependencies.
- [ ] The fetcher uses `asyncio.Semaphore` for controlled download/parse concurrency and `asyncio.gather(..., return_exceptions=True)` for parallel pipelines.
- [ ] The fetcher returns the results dict with `papers_fetched`, `pdfs_downloaded`, `pdfs_parsed`, `papers_stored`, `papers_indexed`, `errors`, and `processing_time`.
- [ ] The fetcher converts `published_date` to `datetime` and stores papers via `PaperRepository.upsert`.
- [ ] Verification steps 7.1–7.2 pass: fetch+process runs without a DB session, and unit tests in `tests/unit/services/test_metadata_fetcher.py` pass.
- [ ] The fetcher raises `MetadataFetchingException` for per-paper pipeline errors and `PipelineException` for top-level pipeline failures.
