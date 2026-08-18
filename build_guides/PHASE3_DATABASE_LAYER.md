# PHASE 3 — Database Layer

> **Build guide for the from-scratch reconstruction of the arXiv Paper Curator (Agentic RAG).**
> This phase implements the persistence layer: the database interface contract, the PostgreSQL implementation, the factory, the `Paper` ORM model, and the `PaperRepository`.

---

## 1. Phase Objective

Implement the complete persistence layer that stores paper metadata in PostgreSQL:

1. **`src/db/interfaces/base.py`** — the `BaseDatabase` abstract contract (and the `BaseRepository` contract).
2. **`src/db/interfaces/postgresql.py`** — `PostgreSQLDatabase` with `declarative_base`, the `_force_ipv4_connect_arg` helper, and `startup`/`teardown`/`get_session`.
3. **`src/db/factory.py`** — `make_database()` factory that wires config → `PostgreSQLSettings` → `PostgreSQLDatabase`.
4. **`src/models/paper.py`** — the `Paper` SQLAlchemy ORM model.
5. **`src/repositories/paper.py`** — `PaperRepository` with `create`, `get_by_arxiv_id`, `upsert`, `get_processing_stats`, and related query methods.

**Why this phase comes at this point:** Persistence is required before repositories can be built, and repositories are required before the ingestion and RAG services. The database layer depends on PHASE 2's configuration (`get_settings()`) and on the `PostgreSQLSettings` schema (which is formally created in PHASE 4 — see the dependency note below). This phase establishes the persistence port contract (hexagonal lean) that later services consume.

> **Dependency note:** `src/db/factory.py` and `src/db/interfaces/postgresql.py` import `PostgreSQLSettings` from `src.schemas.database.config`. That schema is formally created in PHASE 4, but it is a tiny, self-contained Pydantic settings class. **Create it now** (Step 0) so the database layer imports cleanly; PHASE 4 will document it fully as part of the schemas phase.

---

## 2. Prerequisites

- **PHASE 1 complete** — project skeleton, `uv sync`, `src/` package importable.
- **PHASE 2 complete** — `src/config.py` (with `get_settings()`) and `src/exceptions.py` exist.
- **`sqlalchemy>=2.0.0`** (locked `2.0.50`) and **`psycopg2-binary>=2.9.10`** (locked `2.9.12`) installed (declared in PHASE 1).
- **A running PostgreSQL instance** for live verification (e.g., Neon serverless, or a local Postgres). The default connection string is `postgresql://rag_user:rag_password@localhost:5432/rag_db`.

---

## 3. Dependencies to Install

No new dependencies are added in this phase. Required packages (already in `pyproject.toml` from PHASE 1):

- `sqlalchemy>=2.0.0` (locked `2.0.50`)
- `psycopg2-binary>=2.9.10` (locked `2.9.12`)
- `pydantic>=2.11.3` (locked `2.13.4`)
- `pydantic-settings>=2.8.1` (locked `2.14.1`)

Run `uv sync` to ensure they are installed.

---

## 4. Directory Structure to Create

```
<project-root>/src/
├── __init__.py
├── config.py                 (PHASE 2)
├── exceptions.py             (PHASE 2)
├── db/
│   ├── __init__.py           (empty — created in PHASE 1)
│   ├── factory.py            (this phase)
│   └── interfaces/
│       ├── __init__.py       (empty — created in PHASE 1)
│       ├── base.py           (this phase)
│       └── postgresql.py     (this phase)
├── models/
│   ├── __init__.py           (this phase)
│   └── paper.py              (this phase)
├── repositories/
│   ├── __init__.py           (this phase)
│   └── paper.py              (this phase)
└── schemas/
    └── database/
        ├── __init__.py       (empty — this phase, needed for import)
        └── config.py         (this phase — PostgreSQLSettings; fully documented in PHASE 4)
```

---

## 5. Step-by-step Implementation

### Step 0 — Create `src/schemas/database/config.py` (dependency for the factory)

**Full file path:** `<project-root>/src/schemas/database/config.py`

Create the `src/schemas/database/__init__.py` as an **empty file**, then write `config.py`:

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

**Explanation:** `PostgreSQLSettings` is a Pydantic Settings class with `env_prefix = "POSTGRES_"`, so it can read `POSTGRES_DATABASE_URL`, `POSTGRES_ECHO_SQL`, `POSTGRES_POOL_SIZE`, and `POSTGRES_MAX_OVERFLOW` from the environment. The `make_database()` factory constructs this explicitly from the top-level `Settings` fields (see Step 3), so the env-prefix path is a fallback. This file is formally part of PHASE 4's schemas; it is created here because the database layer imports it.

### Step 1 — Create `src/db/interfaces/base.py`

**Full file path:** `<project-root>/src/db/interfaces/base.py`

Write the following content (this is the exact reference implementation):

```python
from abc import ABC, abstractmethod
from typing import Any, ContextManager, Dict, List, Optional

from sqlalchemy.orm import Session


class BaseDatabase(ABC):
    """Base class for database operations."""

    @abstractmethod
    def startup(self) -> None:
        """Initialize the database connection."""

    @abstractmethod
    def teardown(self) -> None:
        """Close the database connection."""

    @abstractmethod
    def get_session(self) -> ContextManager[Session]:
        """Get a database session."""


class BaseRepository(ABC):
    """Base repository pattern for data access."""

    def __init__(self, session: Session):
        self.session = session

    @abstractmethod
    def create(self, data: Dict[str, Any]) -> Any:
        """Create a new record."""

    @abstractmethod
    def get_by_id(self, record_id: Any) -> Optional[Any]:
        """Get a record by ID."""

    @abstractmethod
    def update(self, record_id: Any, data: Dict[str, Any]) -> Optional[Any]:
        """Update a record by ID."""

    @abstractmethod
    def delete(self, record_id: Any) -> bool:
        """Delete a record by ID."""

    @abstractmethod
    def list(self, limit: int = 100, offset: int = 0) -> List[Any]:
        """List records with pagination."""
```

**Explanation:**
- **`BaseDatabase`** is the abstract **persistence port** (hexagonal lean). It defines the contract every concrete database must implement: `startup()`, `teardown()`, and `get_session()` (a context manager yielding a SQLAlchemy `Session`).
- **`BaseRepository`** is an abstract data-access contract. **Known issue:** in the reference project, `PaperRepository` does **not** subclass `BaseRepository` — the contract is defined but unused. You may align `PaperRepository` with it or leave it as-is (matching the reference). This is documented as a pitfall.

### Step 2 — Create `src/db/interfaces/postgresql.py`

**Full file path:** `<project-root>/src/db/interfaces/postgresql.py`

Write the following content (this is the exact reference implementation):

```python
import logging
import re
import socket
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from src.db.interfaces.base import BaseDatabase
from src.schemas.database.config import PostgreSQLSettings

logger = logging.getLogger(__name__)


def _force_ipv4_connect_arg(url: str) -> dict:
    """Resolve the DB hostname to an IPv4 address.

    psycopg2 supports passing both `host` (used for SSL SNI) and `hostaddr`
    (the actual IP to connect to). When `hostaddr` is an IPv4 address, the
    TCP connection bypasses IPv6 entirely — fixing deployments where IPv6 is
    not routed but DNS still returns AAAA records.
    """
    match = re.search(r"@([^/:@?]+)", url)
    if not match:
        return {}
    host = match.group(1)
    try:
        results = socket.getaddrinfo(host, None, socket.AF_INET)
        if results:
            ipv4 = results[0][4][0]
            logger.info(f"Resolved {host} → IPv4 {ipv4} (IPv6 bypassed)")
            return {"hostaddr": ipv4}
    except OSError as e:
        logger.warning(f"IPv4 resolution failed for {host}: {e}")
    return {}


Base = declarative_base()


class PostgreSQLDatabase(BaseDatabase):
    """PostgreSQL database implementation."""

    def __init__(self, config: PostgreSQLSettings):
        self.config = config
        self.engine: Optional[Engine] = None
        self.session_factory: Optional[sessionmaker] = None

    def startup(self) -> None:
        """Initialize the database connection."""
        try:
            # Log connection attempt
            logger.info(
                f"Attempting to connect to PostgreSQL at: {self.config.database_url.split('@')[1] if '@' in self.config.database_url else 'localhost'}"
            )

            connect_args = {}
            url = self.config.database_url
            if "neon.tech" in url or "sslmode=require" in url:
                connect_args["sslmode"] = "require"
                # Force IPv4 — psycopg2 uses hostaddr for the TCP connection
                # and host for SSL SNI, so both IPv4 routing and SSL work correctly
                connect_args.update(_force_ipv4_connect_arg(url))

            self.engine = create_engine(
                url,
                echo=self.config.echo_sql,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=True,
                connect_args=connect_args,
            )

            self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

            # Test the connection
            assert self.engine is not None
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.info("Database connection test successful")

            # Check which tables exist before creating
            inspector = inspect(self.engine)
            existing_tables = inspector.get_table_names()

            # Create tables if they don't exist (idempotent operation)
            Base.metadata.create_all(bind=self.engine)

            # Check if any new tables were created
            updated_tables = inspector.get_table_names()
            new_tables = set(updated_tables) - set(existing_tables)

            if new_tables:
                logger.info(f"Created new tables: {', '.join(new_tables)}")
            else:
                logger.info("All tables already exist - no new tables created")

            logger.info("PostgreSQL database initialized successfully")
            assert self.engine is not None
            logger.info(f"Database: {self.engine.url.database}")
            logger.info(f"Total tables: {', '.join(updated_tables) if updated_tables else 'None'}")
            logger.info("Database connection established")

        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL database: {e}")
            raise

    def teardown(self) -> None:
        """Close the database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("PostgreSQL database connections closed")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session."""
        if not self.session_factory:
            raise RuntimeError("Database not initialized. Call startup() first.")

        session = self.session_factory()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
```

**Explanation:**
- **`_force_ipv4_connect_arg(url)`** resolves the DB hostname to an IPv4 address and returns a `{"hostaddr": ipv4}` connect arg. This fixes the **Neon IPv6 issue**: when DNS returns AAAA (IPv6) records but IPv6 isn't routed, psycopg2 can hang or fail. Passing `hostaddr` (the actual IP) alongside `host` (used for SSL SNI) bypasses IPv6 entirely.
- **`Base = declarative_base()`** is the shared declarative base that all ORM models (e.g., `Paper`) inherit from. `Base.metadata.create_all()` at startup auto-creates tables.
- **`PostgreSQLDatabase.startup()`**:
  - Builds `connect_args` — adds `sslmode=require` and the IPv4 fix when the URL is Neon or requests SSL.
  - Creates the SQLAlchemy `Engine` with `pool_size`, `max_overflow`, and `pool_pre_ping=True` (from config).
  - Creates a `sessionmaker` with `expire_on_commit=False`.
  - Tests the connection with `SELECT 1`.
  - Inspects existing tables, then calls `Base.metadata.create_all()` (idempotent) to create any missing tables.
- **`teardown()`** disposes the engine (closes pooled connections).
- **`get_session()`** is a context manager that yields a session, rolls back on exception, and always closes the session.

### Step 3 — Create `src/db/factory.py`

**Full file path:** `<project-root>/src/db/factory.py`

Write the following content (this is the exact reference implementation):

```python
from src.config import get_settings
from src.db.interfaces.base import BaseDatabase
from src.db.interfaces.postgresql import PostgreSQLDatabase
from src.schemas.database.config import PostgreSQLSettings


def make_database() -> BaseDatabase:
    """Factory function to create a database instance.

    :returns: An instance of the database
    :rtype: BaseDatabase
    """
    # Get settings from centralized config
    settings = get_settings()

    # Create PostgreSQL config from settings
    config = PostgreSQLSettings(
        database_url=settings.postgres_database_url,
        echo_sql=settings.postgres_echo_sql,
        pool_size=settings.postgres_pool_size,
        max_overflow=settings.postgres_max_overflow,
    )

    database = PostgreSQLDatabase(config=config)
    database.startup()
    return database
```

**Explanation:**
- **`make_database()`** is the factory that wires the centralized config into the database layer.
- It reads the top-level `Settings` fields (`postgres_database_url`, `postgres_echo_sql`, `postgres_pool_size`, `postgres_max_overflow`) and constructs a `PostgreSQLSettings` instance.
- It instantiates `PostgreSQLDatabase`, calls `startup()` (which connects and creates tables), and returns the ready database.
- This factory is called from the FastAPI lifespan in PHASE 14 (composition root).

### Step 4 — Create `src/models/paper.py`

**Full file path:** `<project-root>/src/models/paper.py`

Write the following content (this is the exact reference implementation):

```python
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID
from src.db.interfaces.postgresql import Base


class Paper(Base):
    __tablename__ = "papers"

    # Core arXiv metadata
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    arxiv_id = Column(String, unique=True, nullable=False, index=True)
    title = Column(String, nullable=False)
    authors = Column(JSON, nullable=False)
    abstract = Column(Text, nullable=False)
    categories = Column(JSON, nullable=False)
    published_date = Column(DateTime, nullable=False)
    pdf_url = Column(String, nullable=False)

    # Parsed PDF content (added for comprehensive storage)
    raw_text = Column(Text, nullable=True)
    sections = Column(JSON, nullable=True)
    references = Column(JSON, nullable=True)

    # PDF processing metadata
    parser_used = Column(String, nullable=True)
    parser_metadata = Column(JSON, nullable=True)
    pdf_processed = Column(Boolean, default=False, nullable=False)
    pdf_processing_date = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
```

**Explanation:**
- **`Paper`** is the SQLAlchemy ORM model mapped to the `papers` table. It inherits from `Base` (the `declarative_base()` defined in `postgresql.py`), so `Base.metadata.create_all()` will create this table.
- **Core arXiv metadata:** `id` (UUID PK, auto-generated), `arxiv_id` (unique, indexed), `title`, `authors` (JSON list), `abstract`, `categories` (JSON list), `published_date`, `pdf_url`.
- **Parsed PDF content:** `raw_text`, `sections` (JSON), `references` (JSON) — populated after PDF processing.
- **PDF processing metadata:** `parser_used`, `parser_metadata` (JSON), `pdf_processed` (bool, default `False`), `pdf_processing_date`.
- **Timestamps:** `created_at` and `updated_at` default to UTC now; `updated_at` also updates on row update.
- The `authors` and `categories` columns are `JSON` (not ARRAY) — the repository writes Python lists into them, which SQLAlchemy serializes to JSON.

### Step 5 — Create `src/models/__init__.py`

**Full file path:** `<project-root>/src/models/__init__.py`

Write the following content (this is the exact reference implementation):

```python
from .paper import Paper

__all__ = [
    "Paper",
]
```

**Explanation:** Re-exports `Paper` so consumers can `from src.models import Paper`.

### Step 6 — Create `src/repositories/paper.py`

**Full file path:** `<project-root>/src/repositories/paper.py`

Write the following content (this is the exact reference implementation):

```python
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from src.models.paper import Paper
from src.schemas.arxiv.paper import PaperCreate


class PaperRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, paper: PaperCreate) -> Paper:
        db_paper = Paper(**paper.model_dump())
        self.session.add(db_paper)
        self.session.commit()
        self.session.refresh(db_paper)
        return db_paper

    def get_by_arxiv_id(self, arxiv_id: str) -> Optional[Paper]:
        stmt = select(Paper).where(Paper.arxiv_id == arxiv_id)
        return self.session.scalar(stmt)

    def get_by_id(self, paper_id: UUID) -> Optional[Paper]:
        stmt = select(Paper).where(Paper.id == paper_id)
        return self.session.scalar(stmt)

    def get_all(self, limit: int = 100, offset: int = 0) -> List[Paper]:
        stmt = select(Paper).order_by(Paper.published_date.desc()).limit(limit).offset(offset)
        return list(self.session.scalars(stmt))

    def get_count(self) -> int:
        stmt = select(func.count(Paper.id))
        return self.session.scalar(stmt) or 0

    def get_processed_papers(self, limit: int = 100, offset: int = 0) -> List[Paper]:
        """Get papers that have been successfully processed with PDF content."""
        stmt = (
            select(Paper)
            .where(Paper.pdf_processed == True)
            .order_by(Paper.pdf_processing_date.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.session.scalars(stmt))

    def get_unprocessed_papers(self, limit: int = 100, offset: int = 0) -> List[Paper]:
        """Get papers that haven't been processed for PDF content yet."""
        stmt = select(Paper).where(Paper.pdf_processed == False).order_by(Paper.published_date.desc()).limit(limit).offset(offset)
        return list(self.session.scalars(stmt))

    def get_papers_with_raw_text(self, limit: int = 100, offset: int = 0) -> List[Paper]:
        """Get papers that have raw text content stored."""
        stmt = select(Paper).where(Paper.raw_text != None).order_by(Paper.pdf_processing_date.desc()).limit(limit).offset(offset)
        return list(self.session.scalars(stmt))

    def get_processing_stats(self) -> dict:
        """Get statistics about PDF processing status."""
        total_papers = self.get_count()

        # Count processed papers
        processed_stmt = select(func.count(Paper.id)).where(Paper.pdf_processed == True)
        processed_papers = self.session.scalar(processed_stmt) or 0

        # Count papers with text
        text_stmt = select(func.count(Paper.id)).where(Paper.raw_text != None)
        papers_with_text = self.session.scalar(text_stmt) or 0

        return {
            "total_papers": total_papers,
            "processed_papers": processed_papers,
            "papers_with_text": papers_with_text,
            "processing_rate": (processed_papers / total_papers * 100) if total_papers > 0 else 0,
            "text_extraction_rate": (papers_with_text / processed_papers * 100) if processed_papers > 0 else 0,
        }

    def update(self, paper: Paper) -> Paper:
        self.session.add(paper)
        self.session.commit()
        self.session.refresh(paper)
        return paper

    def upsert(self, paper_create: PaperCreate) -> Paper:
        # Check if paper already exists
        existing_paper = self.get_by_arxiv_id(paper_create.arxiv_id)
        if existing_paper:
            # Update existing paper with new content
            for key, value in paper_create.model_dump(exclude_unset=True).items():
                setattr(existing_paper, key, value)
            return self.update(existing_paper)
        else:
            # Create new paper
            return self.create(paper_create)
```

**Explanation:**
- **`PaperRepository`** centralizes all paper data access. It takes a SQLAlchemy `Session` in its constructor.
- **`create(paper: PaperCreate)`** builds a `Paper` from the `PaperCreate` schema (`paper.model_dump()`), adds, commits, and refreshes it.
- **`get_by_arxiv_id(arxiv_id)`** looks up a paper by its unique arXiv ID (used by `upsert`).
- **`get_by_id`, `get_all`, `get_count`** provide basic retrieval and counting.
- **`get_processed_papers`, `get_unprocessed_papers`, `get_papers_with_raw_text`** filter by PDF-processing state.
- **`get_processing_stats()`** returns a dict with total/processed/with-text counts and processing/text-extraction rates.
- **`update(paper)`** re-adds, commits, and refreshes an existing paper.
- **`upsert(paper_create)`** is the key ingestion method: if a paper with the same `arxiv_id` exists, it updates it with the new fields (`model_dump(exclude_unset=True)`); otherwise it creates a new row. This makes ingestion idempotent.
- **Transactions:** each method commits on success; the `get_session()` context manager rolls back on exception (from Step 2).

### Step 7 — Create `src/repositories/__init__.py`

**Full file path:** `<project-root>/src/repositories/__init__.py`

Write the following content (this is the exact reference implementation):

```python
from .paper import PaperRepository

__all__ = [
    "PaperRepository",
]
```

**Explanation:** Re-exports `PaperRepository` so consumers can `from src.repositories import PaperRepository`.

---

## 6. Configuration

The database layer reads the following settings (defined in PHASE 2's `src/config.py`):

| Setting field | Env var | Default | Purpose |
|---------------|---------|---------|---------|
| `postgres_database_url` | `POSTGRES_DATABASE_URL` | `postgresql://rag_user:rag_password@localhost:5432/rag_db` | PostgreSQL connection string |
| `postgres_echo_sql` | `POSTGRES_ECHO_SQL` | `False` | Log SQL statements |
| `postgres_pool_size` | `POSTGRES_POOL_SIZE` | `5` | Connection pool size |
| `postgres_max_overflow` | `POSTGRES_MAX_OVERFLOW` | `0` | Max pool overflow |

The `make_database()` factory maps these into a `PostgreSQLSettings` instance. For Neon, the connection string should include `?sslmode=require` (e.g., `postgresql+psycopg2://<user>:<password>@<host>.neon.tech/rag_db?sslmode=require`), which triggers the IPv4 fix.

---

## 7. Verification

Run these commands from the project root:

```bash
# 1. Confirm all modules import cleanly
uv run python -c "import src.db.factory, src.db.interfaces.base, src.db.interfaces.postgresql, src.models.paper, src.repositories.paper; print('db layer imports OK')"

# 2. Confirm the Paper model maps to the papers table
uv run python -c "
from src.db.interfaces.postgresql import Base
from src.models.paper import Paper
print('papers table:', Paper.__tablename__)
print('columns:', [c.name for c in Paper.__table__.columns])
"

# 3. Confirm the factory builds a database and creates tables (requires a running PostgreSQL)
uv run python -c "
from src.db.factory import make_database
db = make_database()
print('database started OK')
db.teardown()
print('database torn down OK')
"

# 4. Exercise the repository against a live database (requires PostgreSQL + a PaperCreate)
uv run python -c "
from src.db.factory import make_database
from src.repositories.paper import PaperRepository
from src.schemas.arxiv.paper import PaperCreate
from datetime import datetime

db = make_database()
with db.get_session() as session:
    repo = PaperRepository(session)
    paper = PaperCreate(
        arxiv_id='1706.03762',
        title='Attention Is All You Need',
        authors=['Vaswani', 'Shazeer'],
        abstract='We propose the Transformer...',
        categories=['cs.CL', 'cs.LG'],
        published_date=datetime(2017, 6, 12),
        pdf_url='https://arxiv.org/pdf/1706.03762.pdf',
    )
    saved = repo.upsert(paper)
    print('upserted paper id:', saved.id)
    fetched = repo.get_by_arxiv_id('1706.03762')
    print('fetched title:', fetched.title)
    print('stats:', repo.get_processing_stats())
db.teardown()
"
```

**Expected results:**
- All database-layer modules import without error.
- `Paper.__tablename__` is `papers` and its columns match the model definition.
- `make_database()` connects, creates tables, and `teardown()` closes cleanly.
- The repository `upsert`/`get_by_arxiv_id`/`get_processing_stats` round-trip works against a live database.

> **Note:** Verification steps 3 and 4 require a running PostgreSQL instance. If none is available, steps 1 and 2 (imports and model introspection) still validate the code.

---

## 8. Common Pitfalls

- **Neon IPv6 connection failure.** If connecting to Neon hangs or fails, the `_force_ipv4_connect_arg` helper resolves the hostname to an IPv4 address and passes it as `hostaddr`. Ensure the URL contains `neon.tech` or `sslmode=require` so the fix is applied. This is the reference project's documented fix for the "Neon IPv6 issue."
- **`BaseRepository` is not implemented by `PaperRepository`.** In the reference, `PaperRepository` does **not** subclass `BaseRepository`. If you make it subclass, you must implement all abstract methods (`create`, `get_by_id`, `update`, `delete`, `list`) with the `BaseRepository` signatures — which differ from `PaperRepository`'s current signatures. **Match the reference**: leave `PaperRepository` standalone.
- **`Paper` model must import `Base` from `postgresql.py`.** `Base` is defined in `src/db/interfaces/postgresql.py`. Importing it from there ensures `Base.metadata.create_all()` sees the `Paper` table. Do not define a second `declarative_base()` in the model.
- **`authors` and `categories` are JSON columns.** Write Python lists; SQLAlchemy serializes them. Do not use PostgreSQL ARRAY types unless you change the model.
- **`get_session()` requires `startup()` first.** Calling `get_session()` before `startup()` raises `RuntimeError("Database not initialized. Call startup() first.")`.
- **`pool_pre_ping=True`** is set on the engine — this is important for serverless/managed Postgres where idle connections may be dropped.
- **`expire_on_commit=False`** on the sessionmaker keeps attribute access working after commit.
- **The `PaperCreate` schema** (from `src.schemas.arxiv.paper`) is required by the repository. It is formally created in PHASE 4; create it now (or in PHASE 4 before running repository code) to avoid import errors.

---

## 9. Definition of Done

- [ ] `src/db/interfaces/base.py` defines `BaseDatabase` (abstract `startup`/`teardown`/`get_session`) and `BaseRepository`.
- [ ] `src/db/interfaces/postgresql.py` defines `_force_ipv4_connect_arg`, `Base = declarative_base()`, and `PostgreSQLDatabase` with `startup`/`teardown`/`get_session`.
- [ ] `startup()` creates the engine with `pool_size`, `max_overflow`, `pool_pre_ping=True`, tests the connection, and calls `Base.metadata.create_all()`.
- [ ] `src/db/factory.py` defines `make_database()` that wires `get_settings()` → `PostgreSQLSettings` → `PostgreSQLDatabase.startup()`.
- [ ] `src/models/paper.py` defines the `Paper` ORM model mapped to the `papers` table with all columns (core metadata, parsed content, processing metadata, timestamps).
- [ ] `src/models/__init__.py` re-exports `Paper`.
- [ ] `src/repositories/paper.py` defines `PaperRepository` with `create`, `get_by_arxiv_id`, `get_by_id`, `get_all`, `get_count`, `get_processed_papers`, `get_unprocessed_papers`, `get_papers_with_raw_text`, `get_processing_stats`, `update`, and `upsert`.
- [ ] `src/repositories/__init__.py` re-exports `PaperRepository`.
- [ ] `src/schemas/database/config.py` defines `PostgreSQLSettings` (dependency for the factory).
- [ ] `uv run python -c "import src.db.factory, src.models.paper, src.repositories.paper"` succeeds.
- [ ] Against a live PostgreSQL, `make_database()` connects, creates the `papers` table, and `upsert`/`get_by_arxiv_id`/`get_processing_stats` round-trip correctly.
