# PHASE 2 — Configuration

> **Build guide for the from-scratch reconstruction of the arXiv Paper Curator (Agentic RAG).**
> This phase implements the centralized Pydantic Settings v2 configuration and the central exception hierarchy.

---

## 1. Phase Objective

Implement two foundational modules that every other layer depends on:

1. **`src/config.py`** — Pydantic Settings v2 configuration with a settings class hierarchy, per-domain `env_prefix` values, the `__` nested delimiter, and a `get_settings()` factory.
2. **`src/exceptions.py`** — the central, typed exception hierarchy used across all services.

**Why this phase comes at this point:** The dependency graph is `config → everything`. All factories, services, and routers read settings, so configuration must exist before any business logic. The exception hierarchy is likewise a cross-cutting contract that services raise and routers catch. Both are pure-Python modules with no dependencies on other application code, so they can be built immediately after the PHASE 1 skeleton.

---

## 2. Prerequisites

- **PHASE 1 complete** — the project skeleton exists, `uv sync` succeeds, and the `src/` package is importable.
- **`pydantic-settings` installed** (declared in PHASE 1's `pyproject.toml`).
- **No external services required** — this phase only defines configuration classes and exceptions.

---

## 3. Dependencies to Install

No new dependencies are added in this phase. The required packages were already declared in PHASE 1:

- `pydantic>=2.11.3` (locked `2.13.4`)
- `pydantic-settings>=2.8.1` (locked `2.14.1`)

Both are already in `pyproject.toml`. Run `uv sync` to ensure they are installed.

---

## 4. Directory Structure to Create

```
<project-root>/src/
├── __init__.py      (created in PHASE 1)
├── config.py        (this phase)
└── exceptions.py    (this phase)
```

Only two new files are created in this phase.

---

## 5. Step-by-step Implementation

### Step 1 — Create `src/config.py`

**Full file path:** `<project-root>/src/config.py`

Write the following content (this is the exact reference implementation):

```python
import os
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class BaseConfigSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )


class ArxivSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="ARXIV__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    base_url: str = "https://export.arxiv.org/api/query"
    pdf_cache_dir: str = "./data/arxiv_pdfs"
    rate_limit_delay: float = 3.0
    timeout_seconds: int = 60
    max_results: int = 15
    search_category: str = "cs.AI"
    download_max_retries: int = 3
    download_retry_delay_base: float = 5.0
    max_concurrent_downloads: int = 5
    max_concurrent_parsing: int = 1

    namespaces: dict = {
        "atom": "http://www.w3.org/2005/Atom",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    @field_validator("pdf_cache_dir")
    @classmethod
    def validate_cache_dir(cls, v: str) -> str:
        os.makedirs(v, exist_ok=True)
        return v


class PDFParserSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="PDF_PARSER__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    max_pages: int = 30
    max_file_size_mb: int = 20
    do_ocr: bool = False
    do_table_structure: bool = True


class ChunkingSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="CHUNKING__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    chunk_size: int = 600  # Target words per chunk
    overlap_size: int = 100  # Words to overlap between chunks
    min_chunk_size: int = 100  # Minimum words for a valid chunk
    section_based: bool = True  # Use section-based chunking when available


class OpenSearchSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="OPENSEARCH__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    host: str = "http://localhost:9200"
    index_name: str = "arxiv-papers"
    chunk_index_suffix: str = "chunks"  # Creates single hybrid index: {index_name}-{suffix}
    max_text_size: int = 1000000

    # Vector search settings
    vector_dimension: int = 1024  # Jina embeddings dimension
    vector_space_type: str = "cosinesimil"  # cosinesimil, l2, innerproduct

    # Hybrid search settings
    rrf_pipeline_name: str = "hybrid-rrf-pipeline"
    hybrid_search_size_multiplier: int = 2  # Get k*multiplier for better recall


class LangfuseSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="LANGFUSE__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    public_key: str = ""
    secret_key: str = ""
    host: str = "https://us.cloud.langfuse.com"
    enabled: bool = True
    flush_at: int = 15  # Number of events before flushing
    flush_interval: float = 1.0  # Seconds between flushes
    max_retries: int = 3
    timeout: int = 30
    debug: bool = False


class RedisSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="REDIS__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    url: str = "redis://localhost:6379"
    ttl_hours: int = 6


class TelegramSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="TELEGRAM__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    bot_token: str = ""
    enabled: bool = False


class MCPSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="MCP__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    enabled: bool = True
    path: str = "/mcp"


class BedrockSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="BEDROCK__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    aws_access_key_id: str = ""
    aws_secret_access_key: SecretStr = SecretStr("")
    aws_region: str = "us-east-1"
    model_id: str = "meta.llama3-1-70b-instruct-v1:0"
    guardrail_id: str = ""
    guardrail_version: str = "DRAFT"


class LogfireSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="LOGFIRE__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    enabled: bool = True
    token: str = ""
    service_name: str = "arxiv-rag"
    environment: str = "development"
    # "if-token-present" | "true" | "false"
    send_to_logfire: str = "if-token-present"


class Settings(BaseConfigSettings):
    app_version: str = "0.1.0"
    debug: bool = True
    environment: Literal["development", "staging", "production"] = "development"
    service_name: str = "rag-api"

    postgres_database_url: str = "postgresql://rag_user:rag_password@localhost:5432/rag_db"
    postgres_echo_sql: bool = False
    postgres_pool_size: int = 5
    postgres_max_overflow: int = 0

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_timeout: int = 300

    # LLM provider: "openai" or "bedrock"
    provider: str = "openai"

    # Jina AI embeddings configuration
    jina_api_key: str = ""

    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    pdf_parser: PDFParserSettings = Field(default_factory=PDFParserSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    logfire: LogfireSettings = Field(default_factory=LogfireSettings)
    bedrock: BedrockSettings = Field(default_factory=BedrockSettings)

    @field_validator("postgres_database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not (v.startswith("postgresql://") or v.startswith("postgresql+psycopg2://")):
            raise ValueError("Database URL must start with 'postgresql://' or 'postgresql+psycopg2://'")
        return v


def get_settings() -> Settings:
    return Settings()
```

**Explanation:**
- **`BaseConfigSettings`** is the shared base. Its `model_config` sets:
  - `env_file=[".env", str(ENV_FILE_PATH)]` — reads from the project-root `.env` (and `.env` relative to CWD).
  - `extra="ignore"` — unknown env vars are ignored rather than raising.
  - `frozen=True` — settings instances are immutable.
  - `env_nested_delimiter="__"` — the double-underscore delimiter that maps `ARXIV__MAX_RESULTS` to `ArxivSettings.max_results`.
  - `case_sensitive=False` — env var names are case-insensitive.
- **Per-domain settings classes** (`ArxivSettings`, `PDFParserSettings`, `ChunkingSettings`, `OpenSearchSettings`, `LangfuseSettings`, `RedisSettings`, `TelegramSettings`, `MCPSettings`, `BedrockSettings`, `LogfireSettings`) each override `env_prefix` with their domain prefix (e.g., `ARXIV__`). This means `ARXIV__MAX_RESULTS` populates `ArxivSettings.max_results`.
- **`Settings`** is the top-level class. It holds flat fields (e.g., `postgres_database_url`, `openai_api_key`, `provider`, `jina_api_key`) and composes the nested settings via `Field(default_factory=...)`.
- **`ArxivSettings.validate_cache_dir`** is a `field_validator` that creates the PDF cache directory on load.
- **`Settings.validate_database_url`** validates that the PostgreSQL URL uses an accepted scheme.
- **`get_settings()`** is the factory. The reference implementation simply returns `Settings()` (no `lru_cache` in the actual code — see Pitfalls). Every factory/service calls `get_settings()` to obtain validated configuration.

### Step 2 — Create `src/exceptions.py`

**Full file path:** `<project-root>/src/exceptions.py`

Write the following content (this is the exact reference implementation):

```python
class RepositoryException(Exception):
    """Base exception for repository-related errors."""


class PaperNotFound(RepositoryException):
    """Exception raised when paper data is not found."""


class PaperNotSaved(RepositoryException):
    """Exception raised when paper data is not saved."""


class ParsingException(Exception):
    """Base exception for parsing-related errors."""


# Phase 2: PDF parsing exceptions (implemented)
class PDFParsingException(ParsingException):
    """Base exception for PDF parsing-related errors."""


class PDFValidationError(PDFParsingException):
    """Exception raised when PDF file validation fails."""


class PDFDownloadException(Exception):
    """Base exception for PDF download-related errors."""


class PDFDownloadTimeoutError(PDFDownloadException):
    """Exception raised when PDF download times out."""


class PDFCacheException(Exception):
    """Exception raised for PDF cache-related errors."""


# Phase 3+: OpenSearch exceptions (placeholders for Phase 1)
class OpenSearchException(Exception):
    """Base exception for OpenSearch-related errors."""


# Phase 2+: ArXiv API exceptions
class ArxivAPIException(Exception):
    """Base exception for arXiv API-related errors."""


class ArxivAPITimeoutError(ArxivAPIException):
    """Exception raised when arXiv API request times out."""


class ArxivAPIRateLimitError(ArxivAPIException):
    """Exception raised when arXiv API rate limit is exceeded."""


class ArxivParseError(ArxivAPIException):
    """Exception raised when arXiv API response parsing fails."""


# Phase 2+: Metadata fetching exceptions
class MetadataFetchingException(Exception):
    """Base exception for metadata fetching pipeline errors."""


class PipelineException(MetadataFetchingException):
    """Exception raised during pipeline execution."""


class LLMException(Exception):
    """Base exception for LLM-related errors."""


class OllamaException(LLMException):
    """Exception raised for Ollama service errors."""


class OllamaConnectionError(OllamaException):
    """Exception raised when cannot connect to Ollama service."""


class OllamaTimeoutError(OllamaException):
    """Exception raised when Ollama service times out."""


class OpenAILLMException(LLMException):
    """Exception raised for OpenAI API errors."""


class OpenAIConnectionError(OpenAILLMException):
    """Exception raised when cannot reach the OpenAI API."""


class OpenAITimeoutError(OpenAILLMException):
    """Exception raised when the OpenAI API times out."""


class BedrockLLMException(LLMException):
    """Exception raised for AWS Bedrock API errors."""


class BedrockConnectionError(BedrockLLMException):
    """Exception raised when cannot reach the AWS Bedrock API."""


class BedrockTimeoutError(BedrockLLMException):
    """Exception raised when the AWS Bedrock API times out."""


class BedrockGuardrailsException(Exception):
    """Exception raised for AWS Bedrock Guardrails errors."""


# General application exceptions
class ConfigurationError(Exception):
    """Exception raised when configuration is invalid."""
```

**Explanation:**
- This is the **central exception taxonomy** of the system. Services **raise** these typed exceptions; routers **catch** them and map them to `HTTPException`.
- The hierarchy groups exceptions by domain: repository, parsing/PDF, OpenSearch, arXiv, metadata, LLM (with Ollama/OpenAI/Bedrock sub-branches), Bedrock guardrails, and general configuration.
- The reference project notes that `PaperNotFound` and `PaperNotSaved` are currently **unused** (dead code) — they are defined here for completeness but not raised anywhere in the current codebase.
- All exceptions inherit from `Exception` (or a domain base), so callers can catch either the specific type or the broad base.

---

## 6. Configuration

This phase *defines* the configuration system; the actual values are supplied via environment variables (documented in `.env.example` from PHASE 1). Key points:

- **Env file:** `.env` at the project root (git-ignored). The settings classes read `[".env", str(ENV_FILE_PATH)]`.
- **Nested delimiter:** `__` — e.g., `OPENSEARCH__HOST` → `Settings.opensearch.host`.
- **Env prefixes:** `ARXIV__`, `PDF_PARSER__`, `CHUNKING__`, `OPENSEARCH__`, `LANGFUSE__`, `REDIS__`, `TELEGRAM__`, `MCP__`, `BEDROCK__`, `LOGFIRE__`.
- **Flat top-level vars:** `DEBUG`, `ENVIRONMENT`, `PROVIDER`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_TIMEOUT`, `JINA_API_KEY`, `POSTGRES_DATABASE_URL`, `POSTGRES_ECHO_SQL`, `POSTGRES_POOL_SIZE`, `POSTGRES_MAX_OVERFLOW`.
- **Secrets:** `BedrockSettings.aws_secret_access_key` is a `SecretStr` (masked on repr). `openai_api_key` and `jina_api_key` are plain `str` in the reference.

---

## 7. Verification

Run these commands from the project root:

```bash
# 1. Confirm both modules import cleanly
uv run python -c "import src.config, src.exceptions; print('config + exceptions OK')"

# 2. Load settings with defaults (no .env required)
uv run python -c "from src.config import get_settings; s = get_settings(); print(s.service_name, s.provider, s.opensearch.host)"

# 3. Verify nested settings are populated
uv run python -c "from src.config import get_settings; s = get_settings(); print(s.arxiv.max_results, s.chunking.chunk_size, s.opensearch.vector_dimension)"

# 4. Verify the database URL validator rejects bad schemes
uv run python -c "from src.config import Settings; Settings(postgres_database_url='mysql://x')" 2>&1 || echo "validator correctly rejected bad URL"

# 5. Verify the exception hierarchy imports and raises
uv run python -c "
from src.exceptions import (RepositoryException, PaperNotFound, ParsingException,
    PDFParsingException, OpenSearchException, ArxivAPIException, LLMException,
    OpenAILLMException, BedrockLLMException, ConfigurationError)
for exc in (PaperNotFound, PDFParsingException, OpenSearchException, ArxivAPIException, OpenAILLMException, ConfigurationError):
    try:
        raise exc('test')
    except Exception as e:
        print(type(e).__name__, '->', e)
"
```

**Expected results:**
- Both modules import without error.
- `get_settings()` returns a `Settings` instance with default values populated.
- Nested settings (e.g., `s.arxiv.max_results`) are accessible.
- The database URL validator raises a `ValidationError` for a non-PostgreSQL scheme.
- Every exception type imports and raises cleanly.

---

## 8. Common Pitfalls

- **`get_settings()` has no `lru_cache` in the reference.** The task description mentions a `get_settings()` factory with `lru_cache`, but the actual reference implementation is simply `return Settings()`. If you add `@lru_cache`, be aware it deviates from the reference — but it is a safe optimization. **Match the reference** (`return Settings()`) unless you have a specific reason to cache.
- **`env_nested_delimiter="__"` is required** for the `ARXIV__X` → `ArxivSettings.x` mapping to work. Omitting it breaks all nested settings.
- **`extra="ignore"`** prevents unexpected env vars from raising. If you remove it, unrelated env vars (e.g., `PATH`) may cause validation errors.
- **`frozen=True`** makes settings immutable. Attempting to mutate a settings attribute at runtime raises an error — this is intentional.
- **`SecretStr` for Bedrock secret.** `aws_secret_access_key` is a `SecretStr`; access its value via `.get_secret_value()`.
- **The `namespaces` dict in `ArxivSettings`** is a plain `dict` field with a default — it is not overridable via env vars (dicts aren't parsed from env strings). This is expected.
- **`field_validator` on `pdf_cache_dir` creates the directory at load time.** Ensure the process has write permission to the target path.
- **Do not commit `.env`.** Only `.env.example` (names/placeholders) is committed.

---

## 9. Definition of Done

- [ ] `src/config.py` exists with `BaseConfigSettings` and all per-domain settings classes (`ArxivSettings`, `PDFParserSettings`, `ChunkingSettings`, `OpenSearchSettings`, `LangfuseSettings`, `RedisSettings`, `TelegramSettings`, `MCPSettings`, `BedrockSettings`, `LogfireSettings`).
- [ ] `Settings` composes all nested settings via `Field(default_factory=...)`.
- [ ] `get_settings()` factory exists and returns a `Settings` instance.
- [ ] `env_nested_delimiter="__"` and per-class `env_prefix` values are set correctly.
- [ ] `postgres_database_url` validator rejects non-PostgreSQL schemes.
- [ ] `src/exceptions.py` defines the full exception hierarchy (repository, parsing/PDF, OpenSearch, arXiv, metadata, LLM/Ollama/OpenAI/Bedrock, guardrails, configuration).
- [ ] `uv run python -c "import src.config, src.exceptions"` succeeds.
- [ ] All exception types import and raise cleanly.
- [ ] No secrets are committed.
