# PHASE 9 — arXiv Integration

> **Build guide for the from-scratch reconstruction of the arXiv Paper Curator (Agentic RAG).**
>
> This phase implements the arXiv API integration layer. It provides an [`ArxivClient`](../src/services/arxiv/client.py) that fetches paper metadata from the arXiv Atom API (with rate limiting, retry with exponential backoff, and XML parsing), downloads PDFs into a local cache (with retry), and a factory [`make_arxiv_client`](../src/services/arxiv/factory.py) that wires the client from configuration. It also defines the [`ArxivPaper`](../src/schemas/arxiv/paper.py) schema plus the paper create/response schemas used by the database layer.

---

## 1. Phase Objective

By the end of this phase you will have:

- An [`ArxivPaper`](../src/schemas/arxiv/paper.py) Pydantic schema representing a paper returned by the arXiv API.
- The `PaperBase`, `PaperCreate`, `PaperResponse`, and `PaperSearchResponse` schemas used for database persistence and API responses.
- An [`ArxivClient`](../src/services/arxiv/client.py) with:
  - `fetch_papers` — fetch papers for the configured category with optional date filtering, pagination, and sorting.
  - `fetch_papers_with_query` — fetch papers using a custom arXiv search query.
  - `fetch_paper_by_id` — fetch a single paper by its arXiv ID.
  - `download_pdf` — download a paper's PDF into a local cache directory (with caching and retry).
  - Internal XML parsing helpers (`_parse_response`, `_parse_single_entry`, `_get_text`, `_get_arxiv_id`, `_get_authors`, `_get_categories`, `_get_pdf_url`).
- A factory [`make_arxiv_client`](../src/services/arxiv/factory.py) that builds the client from `ArxivSettings`.
- Rate limiting (arXiv recommends 3 seconds between requests), retry with exponential backoff on HTTP 429/503, and typed exceptions for API, timeout, and parse errors.

---

## 2. Prerequisites

- Completion of **PHASE 2 (Configuration)** so that `ArxivSettings` exists in [`src/config.py`](../src/config.py) with the `ARXIV__` env prefix and the `namespaces` dictionary.
- Completion of **PHASE 4 (Exceptions)** so that `ArxivAPIException`, `ArxivAPITimeoutError`, `ArxivParseError`, `PDFDownloadException`, and `PDFDownloadTimeoutError` exist in [`src/exceptions.py`](../src/exceptions.py).
- Completion of **PHASE 4 (Schemas)** so that the `src/schemas/arxiv/` package directory exists.
- Python 3.11+ and the project's virtual environment active.
- No arXiv API key is required — the arXiv API is public and free to query.

---

## 3. Dependencies to Install

This phase requires the following third-party packages:

```bash
uv add httpx
```

> `httpx` is used for both the arXiv API metadata requests and the streaming PDF downloads. `pydantic` (already installed in prior phases) is used for the schemas. No other new dependencies are required.

If you are not using `uv`, install with pip:

```bash
pip install httpx
```

---

## 4. Directory Structure to Create

```
src/
├── schemas/
│   └── arxiv/
│       ├── __init__.py            # (empty package marker)
│       └── paper.py               # ArxivPaper, PaperBase, PaperCreate, PaperResponse, PaperSearchResponse
└── services/
    └── arxiv/
        ├── client.py              # ArxivClient
        └── factory.py             # make_arxiv_client
```

> **Note on `__init__.py`:** In the reference project, `src/services/arxiv/` does **not** contain an `__init__.py` file (only `client.py` and `factory.py`). The `src/schemas/arxiv/` directory **does** contain an `__init__.py`. Create the schema package `__init__.py` as shown; the service directory may omit it (the factory uses a relative import `from .client import ArxivClient`, which works regardless in modern Python with implicit namespace packages, but adding an empty `__init__.py` is also acceptable and recommended for consistency).

---

## 5. Step-by-Step Implementation

### Step 1 — Define the arXiv paper schemas

**Full file path:** `src/schemas/arxiv/paper.py`

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

**Explanation:** `ArxivPaper` is the schema produced directly by the arXiv API XML parser — note that its `published_date` is a plain `str` (the raw ISO string from the API). `PaperBase` is the persisted form where `published_date` is a `datetime`. `PaperCreate` extends `PaperBase` with optional parsed PDF content and processing metadata (used by `PaperRepository.upsert` in a later phase). `PaperResponse` adds the database `id` and timestamps and is used for API responses. `PaperSearchResponse` wraps a list of papers with a total count.

---

### Step 2 — Add the arXiv schema package `__init__.py`

**Full file path:** `src/schemas/arxiv/__init__.py`

```python
```

**Explanation:** This file is intentionally empty in the reference project. It marks `src/schemas/arxiv/` as a Python package so the schema module can be imported as `from src.schemas.arxiv.paper import ArxivPaper`.

---

### Step 3 — Implement the arXiv client

**Full file path:** `src/services/arxiv/client.py`

```python
import asyncio
import logging
import time
import xml.etree.ElementTree as ET
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote, urlencode

import httpx
from src.config import ArxivSettings
from src.exceptions import ArxivAPIException, ArxivAPITimeoutError, ArxivParseError, PDFDownloadException, PDFDownloadTimeoutError
from src.schemas.arxiv.paper import ArxivPaper

logger = logging.getLogger(__name__)


class ArxivClient:
    """Client for fetching papers from arXiv API."""

    def __init__(self, settings: ArxivSettings):
        self._settings = settings
        self._last_request_time: Optional[float] = None

    @cached_property
    def pdf_cache_dir(self) -> Path:
        """PDF cache directory."""
        cache_dir = Path(self._settings.pdf_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @property
    def base_url(self) -> str:
        return self._settings.base_url

    @property
    def namespaces(self) -> dict:
        return self._settings.namespaces

    @property
    def rate_limit_delay(self) -> float:
        return self._settings.rate_limit_delay

    @property
    def timeout_seconds(self) -> int:
        return self._settings.timeout_seconds

    @property
    def max_results(self) -> int:
        return self._settings.max_results

    @property
    def search_category(self) -> str:
        return self._settings.search_category

    async def fetch_papers(
        self,
        max_results: Optional[int] = None,
        start: int = 0,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[ArxivPaper]:
        """
        Fetch papers from arXiv for the configured category.

        Args:
            max_results: Maximum number of papers to fetch (uses settings default if None)
            start: Starting index for pagination
            sort_by: Sort criteria (submittedDate, lastUpdatedDate, relevance)
            sort_order: Sort order (ascending, descending)
            from_date: Filter papers submitted after this date (format: YYYYMMDD)
            to_date: Filter papers submitted before this date (format: YYYYMMDD)

        Returns:
            List of ArxivPaper objects for the configured category
        """
        if max_results is None:
            max_results = self.max_results

        # Build search query
        search_query = f"cat:{self.search_category}"

        # Add date filtering if provided
        if from_date or to_date:
            # Convert dates to arXiv format (YYYYMMDDHHMM) - use 0000 for start of day, 2359 for end
            date_from = f"{from_date}0000" if from_date else "*"
            date_to = f"{to_date}2359" if to_date else "*"
            # Use correct arXiv API syntax with + symbols
            search_query += f" AND submittedDate:[{date_from}+TO+{date_to}]"

        params = {
            "search_query": search_query,
            "start": start,
            "max_results": min(max_results, 2000),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        safe = ":+[]"  # Don't encode :, +, [, ] characters needed for arXiv queries
        url = f"{self.base_url}?{urlencode(params, quote_via=quote, safe=safe)}"

        # Retry loop with exponential backoff for 429 rate limits
        max_retries = 3
        base_wait = 5

        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching {max_results} {self.search_category} papers from arXiv (attempt {attempt + 1}/{max_retries})")

                # Add rate limiting delay between all requests (arXiv recommends 3 seconds)
                if self._last_request_time is not None:
                    time_since_last = time.time() - self._last_request_time
                    if time_since_last < self.rate_limit_delay:
                        sleep_time = self.rate_limit_delay - time_since_last
                        await asyncio.sleep(sleep_time)

                self._last_request_time = time.time()

                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    xml_data = response.text

                papers = self._parse_response(xml_data)
                logger.info(f"Fetched {len(papers)} papers")

                return papers

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                if status_code in (429, 503) and attempt < max_retries - 1:
                    wait_time = base_wait * (2 ** attempt)
                    logger.warning(f"arXiv API returned {status_code}. Waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                    await asyncio.sleep(wait_time)
                    continue
                logger.error(f"arXiv API HTTP error: {e}")
                raise ArxivAPIException(f"arXiv API returned error {status_code}: {e}")
            except httpx.TimeoutException as e:
                logger.error(f"arXiv API timeout: {e}")
                raise ArxivAPITimeoutError(f"arXiv API request timed out: {e}")
            except Exception as e:
                logger.error(f"Failed to fetch papers from arXiv: {e}")
                raise ArxivAPIException(f"Unexpected error fetching papers from arXiv: {e}")

        # Should never reach here, but satisfy type checker
        return []

    async def fetch_papers_with_query(
        self,
        search_query: str,
        max_results: Optional[int] = None,
        start: int = 0,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
    ) -> List[ArxivPaper]:
        """
        Fetch papers from arXiv using a custom search query.

        Args:
            search_query: Custom arXiv search query (e.g., "cat:cs.AI AND submittedDate:[20240101 TO 20241231]")
            max_results: Maximum number of papers to fetch (uses settings default if None)
            start: Starting index for pagination
            sort_by: Sort criteria (submittedDate, lastUpdatedDate, relevance)
            sort_order: Sort order (ascending, descending)

        Returns:
            List of ArxivPaper objects matching the search query

        Examples:
            # Papers from last 30 days
            "cat:cs.AI AND submittedDate:[20240101 TO *]"

            # Papers by specific author
            "au:LeCun AND cat:cs.AI"

            # Papers with specific keywords in title
            "ti:transformer AND cat:cs.AI"
        """
        if max_results is None:
            max_results = self.max_results

        params = {
            "search_query": search_query,
            "start": start,
            "max_results": min(max_results, 2000),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        safe = ":+[]*"  # Don't encode :, +, [, ], *, characters needed for arXiv queries
        url = f"{self.base_url}?{urlencode(params, quote_via=quote, safe=safe)}"

        # Retry loop with exponential backoff for 429 rate limits
        max_retries = 3
        base_wait = 5

        for attempt in range(max_retries):
            try:
                # Add rate limiting delay between all requests (arXiv recommends 3 seconds)
                if self._last_request_time is not None:
                    time_since_last = time.time() - self._last_request_time
                    if time_since_last < self.rate_limit_delay:
                        sleep_time = self.rate_limit_delay - time_since_last
                        await asyncio.sleep(sleep_time)

                self._last_request_time = time.time()

                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    xml_data = response.text

                papers = self._parse_response(xml_data)
                logger.info(f"Query returned {len(papers)} papers")

                return papers

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                if status_code in (429, 503) and attempt < max_retries - 1:
                    wait_time = base_wait * (2 ** attempt)
                    logger.warning(f"arXiv API returned {status_code}. Waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                    await asyncio.sleep(wait_time)
                    continue
                logger.error(f"arXiv API HTTP error: {e}")
                raise ArxivAPIException(f"arXiv API returned error {status_code}: {e}")
            except httpx.TimeoutException as e:
                logger.error(f"arXiv API timeout: {e}")
                raise ArxivAPITimeoutError(f"arXiv API request timed out: {e}")
            except Exception as e:
                logger.error(f"Failed to fetch papers from arXiv: {e}")
                raise ArxivAPIException(f"Unexpected error fetching papers from arXiv: {e}")

        # Should never reach here, but satisfy type checker
        return []

    async def fetch_paper_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        Fetch a specific paper by its arXiv ID.

        Args:
            arxiv_id: arXiv paper ID (e.g., "2507.17748v1" or "2507.17748")

        Returns:
            ArxivPaper object or None if not found
        """
        # Clean the arXiv ID (remove version if needed for search)
        clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
        params = {"id_list": clean_id, "max_results": 1}

        safe = ":+[]*"  # Don't encode :, +, [, ], *, characters needed for arXiv queries
        url = f"{self.base_url}?{urlencode(params, quote_via=quote, safe=safe)}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                xml_data = response.text

            papers = self._parse_response(xml_data)

            if papers:
                return papers[0]
            else:
                logger.warning(f"Paper {arxiv_id} not found")
                return None

        except httpx.TimeoutException as e:
            logger.error(f"arXiv API timeout for paper {arxiv_id}: {e}")
            raise ArxivAPITimeoutError(f"arXiv API request timed out for paper {arxiv_id}: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"arXiv API HTTP error for paper {arxiv_id}: {e}")
            raise ArxivAPIException(f"arXiv API returned error {e.response.status_code} for paper {arxiv_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to fetch paper {arxiv_id} from arXiv: {e}")
            raise ArxivAPIException(f"Unexpected error fetching paper {arxiv_id} from arXiv: {e}")

    def _parse_response(self, xml_data: str) -> List[ArxivPaper]:
        """
        Parse arXiv API XML response into ArxivPaper objects.

        Args:
            xml_data: Raw XML response from arXiv API

        Returns:
            List of parsed ArxivPaper objects
        """
        try:
            root = ET.fromstring(xml_data)
            entries = root.findall("atom:entry", self.namespaces)

            papers = []
            for entry in entries:
                paper = self._parse_single_entry(entry)
                if paper:
                    papers.append(paper)

            return papers

        except ET.ParseError as e:
            logger.error(f"Failed to parse arXiv XML response: {e}")
            raise ArxivParseError(f"Failed to parse arXiv XML response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing arXiv response: {e}")
            raise ArxivParseError(f"Unexpected error parsing arXiv response: {e}")

    def _parse_single_entry(self, entry: ET.Element) -> Optional[ArxivPaper]:
        """
        Parse a single entry from arXiv XML response.

        Args:
            entry: XML entry element

        Returns:
            ArxivPaper object or None if parsing fails
        """
        try:
            # Extract basic metadata
            arxiv_id = self._get_arxiv_id(entry)
            if not arxiv_id:
                return None

            title = self._get_text(entry, "atom:title", clean_newlines=True)
            authors = self._get_authors(entry)
            abstract = self._get_text(entry, "atom:summary", clean_newlines=True)
            published = self._get_text(entry, "atom:published")
            categories = self._get_categories(entry)
            pdf_url = self._get_pdf_url(entry)

            return ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                published_date=published,
                categories=categories,
                pdf_url=pdf_url,
            )

        except Exception as e:
            logger.error(f"Failed to parse entry: {e}")
            return None

    def _get_text(self, element: ET.Element, path: str, clean_newlines: bool = False) -> str:
        """
        Extract text from XML element safely.

        Args:
            element: Parent XML element
            path: XPath to find the text element
            clean_newlines: Whether to replace newlines with spaces

        Returns:
            Extracted text or empty string
        """
        elem = element.find(path, self.namespaces)
        if elem is None or elem.text is None:
            return ""

        text = elem.text.strip()
        return text.replace("\n", " ") if clean_newlines else text

    def _get_arxiv_id(self, entry: ET.Element) -> Optional[str]:
        """
        Extract arXiv ID from entry.

        Args:
            entry: XML entry element

        Returns:
            arXiv ID or None
        """
        id_elem = entry.find("atom:id", self.namespaces)
        if id_elem is None or id_elem.text is None:
            return None
        return id_elem.text.split("/")[-1]

    def _get_authors(self, entry: ET.Element) -> List[str]:
        """
        Extract author names from entry.

        Args:
            entry: XML entry element

        Returns:
            List of author names
        """
        authors = []
        for author in entry.findall("atom:author", self.namespaces):
            name = self._get_text(author, "atom:name")
            if name:
                authors.append(name)
        return authors

    def _get_categories(self, entry: ET.Element) -> List[str]:
        """
        Extract categories from entry.

        Args:
            entry: XML entry element

        Returns:
            List of category terms
        """
        categories = []
        for category in entry.findall("atom:category", self.namespaces):
            term = category.get("term")
            if term:
                categories.append(term)
        return categories

    def _get_pdf_url(self, entry: ET.Element) -> str:
        """
        Extract PDF URL from entry links.

        Args:
            entry: XML entry element

        Returns:
            PDF URL or empty string (always HTTPS)
        """
        for link in entry.findall("atom:link", self.namespaces):
            if link.get("type") == "application/pdf":
                url = link.get("href", "")
                # Convert HTTP to HTTPS for arXiv URLs
                if url.startswith("http://arxiv.org/"):
                    url = url.replace("http://arxiv.org/", "https://arxiv.org/")
                return url
        return ""

    async def download_pdf(self, paper: ArxivPaper, force_download: bool = False) -> Optional[Path]:
        """
        Download PDF for a given paper to local cache.

        Args:
            paper: ArxivPaper object containing PDF URL
            force_download: Force re-download even if file exists

        Returns:
            Path to downloaded PDF file or None if download failed
        """
        if not paper.pdf_url:
            logger.error(f"No PDF URL for paper {paper.arxiv_id}")
            return None

        pdf_path = self._get_pdf_path(paper.arxiv_id)

        # Return cached PDF if exists
        if pdf_path.exists() and not force_download:
            logger.info(f"Using cached PDF: {pdf_path.name}")
            return pdf_path

        # Download with retry
        if await self._download_with_retry(paper.pdf_url, pdf_path):
            return pdf_path
        else:
            return None

    def _get_pdf_path(self, arxiv_id: str) -> Path:
        """
        Get the local path for a PDF file.

        Args:
            arxiv_id: arXiv paper ID

        Returns:
            Path object for the PDF file
        """
        safe_filename = arxiv_id.replace("/", "_") + ".pdf"
        return self.pdf_cache_dir / safe_filename

    async def _download_with_retry(self, url: str, path: Path, max_retries: Optional[int] = None) -> bool:
        """Download a file with retry logic."""
        if max_retries is None:
            max_retries = self._settings.download_max_retries

        logger.info(f"Downloading PDF from {url}")

        # Respect rate limits
        await asyncio.sleep(self.rate_limit_delay)

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=float(self.timeout_seconds)) as client:
                    async with client.stream("GET", url) as response:
                        response.raise_for_status()
                        with open(path, "wb") as f:
                            async for chunk in response.aiter_bytes():
                                f.write(chunk)
                logger.info(f"Successfully downloaded to {path.name}")
                return True

            except httpx.TimeoutException as e:
                if attempt < max_retries - 1:
                    wait_time = self._settings.download_retry_delay_base * (attempt + 1)
                    logger.warning(f"PDF download timeout (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"PDF download failed after {max_retries} attempts due to timeout: {e}")
                    raise PDFDownloadTimeoutError(f"PDF download timed out after {max_retries} attempts: {e}")
            except httpx.HTTPError as e:
                if attempt < max_retries - 1:
                    wait_time = self._settings.download_retry_delay_base * (attempt + 1)  # Exponential backoff
                    logger.warning(f"Download failed (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    raise PDFDownloadException(f"PDF download failed after {max_retries} attempts: {e}")
            except Exception as e:
                logger.error(f"Unexpected download error: {e}")
                raise PDFDownloadException(f"Unexpected error during PDF download: {e}")

        # Clean up partial download
        if path.exists():
            path.unlink()

        return False
```

**Explanation:** The `ArxivClient` is the ingestion primitive for paper metadata. Key behaviors:

- **Rate limiting:** `_last_request_time` tracks the last request; before each request the client sleeps the remaining time to honor `rate_limit_delay` (default 3s, per arXiv's recommendation).
- **Retry with exponential backoff:** `fetch_papers` and `fetch_papers_with_query` retry up to 3 times on HTTP 429/503, sleeping `5 * 2**attempt` seconds between attempts.
- **URL encoding:** The `safe` string (`":+[]"` or `":+[]*"`) prevents `urlencode` from encoding the special characters arXiv's query syntax requires.
- **XML parsing:** `_parse_response` uses `xml.etree.ElementTree` with the `namespaces` dictionary from settings to find `atom:entry` elements. Each entry is parsed into an `ArxivPaper` via `_parse_single_entry`.
- **PDF download:** `download_pdf` checks the local cache first (returning the cached path unless `force_download`), then streams the PDF to disk with retry via `_download_with_retry`. Partial downloads are cleaned up on failure.
- **Typed exceptions:** API errors raise `ArxivAPIException`, timeouts raise `ArxivAPITimeoutError`, parse failures raise `ArxivParseError`, and download failures raise `PDFDownloadException` / `PDFDownloadTimeoutError`.

---

### Step 4 — Add the arXiv client factory

**Full file path:** `src/services/arxiv/factory.py`

```python
from src.config import get_settings

from .client import ArxivClient


def make_arxiv_client() -> ArxivClient:
    """Factory function to create an arXiv client instance.

    :returns: An instance of the arXiv client
    :rtype: ArxivClient
    """
    # Get settings from centralized config
    settings = get_settings()

    # Create arXiv client with explicit settings
    client = ArxivClient(settings=settings.arxiv)

    return client
```

**Explanation:** `make_arxiv_client` is the composition root for the arXiv client. It loads the application settings via `get_settings()`, passes the nested `settings.arxiv` block (an `ArxivSettings` instance) into the `ArxivClient` constructor, and returns the fully configured client. This factory is called from the application lifespan and from the metadata fetcher in a later phase.

---

## 6. Configuration

The arXiv client is driven entirely by the `ArxivSettings` block defined in [`src/config.py`](../src/config.py). These settings use the `ARXIV__` environment prefix and the `__` nested delimiter, so they are configured as follows:

| Environment variable | Default | Description |
| --- | --- | --- |
| `ARXIV__BASE_URL` | `https://export.arxiv.org/api/query` | arXiv API query endpoint. |
| `ARXIV__PDF_CACHE_DIR` | `./data/arxiv_pdfs` | Local directory where downloaded PDFs are cached. |
| `ARXIV__RATE_LIMIT_DELAY` | `3.0` | Seconds to wait between arXiv API requests (arXiv recommends 3s). |
| `ARXIV__TIMEOUT_SECONDS` | `60` | HTTP timeout for API and download requests. |
| `ARXIV__MAX_RESULTS` | `15` | Default number of papers to fetch per call. |
| `ARXIV__SEARCH_CATEGORY` | `cs.AI` | Default arXiv category to query (e.g., `cs.AI`). |
| `ARXIV__DOWNLOAD_MAX_RETRIES` | `3` | Number of PDF download retry attempts. |
| `ARXIV__DOWNLOAD_RETRY_DELAY_BASE` | `5.0` | Base delay (seconds) for download retry backoff. |
| `ARXIV__MAX_CONCURRENT_DOWNLOADS` | `5` | Max concurrent PDF downloads (used by the metadata fetcher in a later phase). |
| `ARXIV__MAX_CONCURRENT_PARSING` | `1` | Max concurrent PDF parsing operations (used by the metadata fetcher in a later phase). |

The `namespaces` dictionary is defined in code (not configurable via env) and contains the Atom, OpenSearch, and arXiv namespaces used for XML parsing:

```python
namespaces = {
    "atom": "http://www.w3.org/2005/Atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    "arxiv": "http://arxiv.org/schemas/atom",
}
```

**Example `.env` snippet:**

```dotenv
ARXIV__BASE_URL=https://export.arxiv.org/api/query
ARXIV__PDF_CACHE_DIR=./data/arxiv_pdfs
ARXIV__RATE_LIMIT_DELAY=3.0
ARXIV__TIMEOUT_SECONDS=60
ARXIV__MAX_RESULTS=15
ARXIV__SEARCH_CATEGORY=cs.AI
ARXIV__DOWNLOAD_MAX_RETRIES=3
ARXIV__DOWNLOAD_RETRY_DELAY_BASE=5.0
ARXIV__MAX_CONCURRENT_DOWNLOADS=5
ARXIV__MAX_CONCURRENT_PARSING=1
```

> The `pdf_cache_dir` validator in `ArxivSettings` creates the directory automatically if it does not exist (`os.makedirs(v, exist_ok=True)`).

---

## 7. Verification

Verify the arXiv integration works end-to-end before wiring it into the ingestion flow.

### 7.1 Schema smoke test

Confirm the schemas import and validate correctly:

```bash
python -c "
from src.schemas.arxiv.paper import ArxivPaper, PaperCreate
p = ArxivPaper(
    arxiv_id='2507.17748v1',
    title='Test Paper',
    authors=['Alice', 'Bob'],
    abstract='An abstract.',
    categories=['cs.AI'],
    published_date='2025-07-25T00:00:00Z',
    pdf_url='https://arxiv.org/pdf/2507.17748v1',
)
print('ArxivPaper OK:', p.arxiv_id)
"
```

**Expected:** The `ArxivPaper` instance is created without validation errors.

### 7.2 Fetch papers from the live arXiv API

```bash
python -c "
import asyncio
from src.services.arxiv.factory import make_arxiv_client

async def main():
    client = make_arxiv_client()
    papers = await client.fetch_papers(max_results=3)
    print(f'Fetched {len(papers)} papers')
    for p in papers:
        print(f'  - {p.arxiv_id}: {p.title[:60]}')
        print(f'    authors={len(p.authors)}, categories={p.categories}, pdf={p.pdf_url}')

asyncio.run(main())
"
```

**Expected:** A small number of papers are fetched from the live arXiv API, each with a populated `arxiv_id`, `title`, `authors`, `categories`, `published_date`, and `pdf_url`.

### 7.3 Fetch a paper by ID

```bash
python -c "
import asyncio
from src.services.arxiv.factory import make_arxiv_client

async def main():
    client = make_arxiv_client()
    paper = await client.fetch_paper_by_id('1706.03762')
    if paper:
        print(f'Found: {paper.title}')
    else:
        print('Paper not found')

asyncio.run(main())
"
```

**Expected:** The paper with arXiv ID `1706.03762` (Attention Is All You Need) is returned with its title.

### 7.4 Download a PDF to cache

```bash
python -c "
import asyncio
from src.services.arxiv.factory import make_arxiv_client

async def main():
    client = make_arxiv_client()
    paper = await client.fetch_paper_by_id('1706.03762')
    if paper:
        path = await client.download_pdf(paper)
        print(f'PDF cached at: {path}')
    else:
        print('Paper not found')

asyncio.run(main())
"
```

**Expected:** The PDF for the paper is downloaded into the configured `ARXIV__PDF_CACHE_DIR` (default `./data/arxiv_pdfs`) and the local file path is printed. Running the command a second time should print `Using cached PDF` and return immediately without re-downloading.

### 7.5 Run the unit tests

The reference project includes unit tests for the arXiv client. Run them to confirm the parsing and download logic behaves correctly:

```bash
python -m pytest tests/unit/services/test_arxiv_client.py -v
```

**Expected:** All tests in `test_arxiv_client.py` pass.

---

## 8. Common Pitfalls

- **Missing `namespaces` dictionary.** The XML parsing relies on `self.namespaces` (the `atom`, `opensearch`, and `arxiv` namespaces). If these are not defined in `ArxivSettings`, `entry.findall("atom:entry", ...)` returns an empty list and `fetch_papers` silently returns `[]`. Ensure the `namespaces` dict is present in `ArxivSettings`.

- **URL-encoding the arXiv query syntax.** `urlencode` by default encodes `:`, `+`, `[`, `]`, and `*`. The arXiv API requires these literal characters in `search_query` (e.g., `cat:cs.AI AND submittedDate:[20240101+TO+20241231]`). Always pass the `safe` argument (`":+[]"` or `":+[]*"`) to `urlencode`, otherwise the query returns no results or a 400 error.

- **Rate limiting / 429 responses.** arXiv enforces a 3-second minimum between requests. If you remove the `_last_request_time` rate limiting, you will hit HTTP 429 responses and get throttled. Keep the delay and the exponential-backoff retry loop (5s, 10s, 20s) for 429/503.

- **`published_date` type mismatch.** `ArxivPaper.published_date` is a `str` (raw ISO string from the API), but `PaperBase.published_date` is a `datetime`. Do not confuse the two — the conversion to `datetime` happens later in the metadata fetcher (via `dateutil.parser`), not in the arXiv client.

- **PDF cache directory not created.** The `pdf_cache_dir` is created lazily via the `@cached_property` `pdf_cache_dir` and the `ArxivSettings` field validator. If you bypass the factory and construct `ArxivClient` with a raw dict, the directory may not exist and `download_pdf` will fail with `FileNotFoundError`. Always go through `make_arxiv_client()`.

- **Partial downloads left on disk.** If a download fails mid-stream, `_download_with_retry` cleans up the partial file at the end. If you add your own download logic, remember to unlink partial files on failure so a corrupt PDF is not later treated as a valid cached file.

- **`fetch_paper_by_id` does not rate-limit.** Unlike `fetch_papers`, `fetch_paper_by_id` does not apply the `_last_request_time` delay. If you call it in a tight loop, you may hit rate limits. Add a small delay if you need to fetch many papers by ID.

---

## 9. Definition of Done

This phase is complete when:

- [ ] `src/schemas/arxiv/paper.py` defines `ArxivPaper`, `PaperBase`, `PaperCreate`, `PaperResponse`, and `PaperSearchResponse` exactly as specified, with `ArxivPaper.published_date` as `str` and `PaperBase.published_date` as `datetime`.
- [ ] `src/schemas/arxiv/__init__.py` exists (empty package marker).
- [ ] `src/services/arxiv/client.py` implements the full `ArxivClient` with `fetch_papers`, `fetch_papers_with_query`, `fetch_paper_by_id`, `download_pdf`, `_download_with_retry`, and all XML parsing helpers.
- [ ] `src/services/arxiv/factory.py` implements `make_arxiv_client()` that builds the client from `settings.arxiv`.
- [ ] `httpx` is added to the project dependencies.
- [ ] The `ARXIV__*` environment variables are documented and honored by the client.
- [ ] Verification steps 7.1–7.4 pass: schemas validate, live API fetch returns papers, `fetch_paper_by_id` returns a known paper, and `download_pdf` caches a PDF.
- [ ] Unit tests in `tests/unit/services/test_arxiv_client.py` pass.
- [ ] The client raises the typed exceptions (`ArxivAPIException`, `ArxivAPITimeoutError`, `ArxivParseError`, `PDFDownloadException`, `PDFDownloadTimeoutError`) on the appropriate failure paths.