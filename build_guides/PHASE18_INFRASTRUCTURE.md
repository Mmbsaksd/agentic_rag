# PHASE 18 — Infrastructure: Docker, Compose, Airflow & Kubernetes

> Build guide for containerizing the application and standing up the full infrastructure: a multi-stage Dockerfile, a Docker Compose stack (API, OpenSearch, Dashboards, Airflow), the Airflow daily ingestion DAG, and the Kubernetes deployment manifests (EKS cluster, API Deployment/HPA/Service, OpenSearch StatefulSet, Secrets, and Grafana Cloud monitoring).

---

## 1. Phase Objective

In PHASE 17 you composed the full application into a single bootable FastAPI app. In this phase you will make it deployable and operational by:

1. Creating the **multi-stage `Dockerfile`** (uv build + slim runtime) with CPU-only PyTorch and Docling system dependencies.
2. Creating the **`compose.yml`** stack with 4 services: `api`, `opensearch`, `opensearch-dashboards`, and `airflow`.
3. Creating the **`entrypoint.sh`** that migrates the Airflow DB, syncs permissions, creates the admin user, and starts webserver + scheduler.
4. Creating the **Airflow DAG** (`arxiv_paper_ingestion.py`) and the modular `arxiv_ingestion` package (`common`, `setup`, `fetching`, `indexing`, `reporting`).
5. Creating the **Airflow `Dockerfile`** and its own `entrypoint.sh`.
6. Creating the **Kubernetes manifests**: EKS `cluster.yaml`, API `deployment.yaml`/`hpa.yaml`/`service.yaml`, OpenSearch `statefulset.yaml`, and the `secret-template.yaml`.
7. Creating the **Grafana Cloud** `values.yaml` for cluster metrics, logs, and cost monitoring.

By the end of this phase the application runs as a containerized stack locally via Docker Compose and is deployable to AWS EKS with autoscaling, persistent storage, and observability.

---

## 2. Prerequisites

- PHASE 17 completed: the application composes and boots via `uvicorn src.main:app`.
- PHASE 2 completed: `Settings` with all the `*__*` nested config sections referenced by the Secret template.
- PHASE 8 completed: the hybrid indexing service (`make_hybrid_indexing_service`) and OpenSearch hybrid index setup.
- PHASE 9 completed: the arXiv client with `fetch_papers_with_query` and `search_category`.
- PHASE 10 completed: the Docling PDF parser service.
- PHASE 11 completed: the metadata fetcher (`make_metadata_fetcher`).
- PHASE 3 completed: `make_database` and the `Paper` model.
- Local tooling: Docker, Docker Compose, `uv`, and (for Kubernetes) `eksctl`, `awscli`, and `kubectl`.

---

## 3. Dependencies to Install

No new Python dependencies are required for this phase — the infrastructure files reference packages already installed in earlier phases. The container images pull their own dependencies:

- **API image**: built from `pyproject.toml`/`uv.lock` (already contains all app dependencies).
- **Airflow image**: installs `apache-airflow[postgres]==2.10.3` (pinned via the official Airflow constraints file), `psycopg2-binary`, CPU-only `torch`/`torchvision`/`torchaudio`, and the project's `requirements-airflow.txt`.

Local CLI tooling for Kubernetes:

```bash
# macOS
brew install eksctl awscli kubectl
# Windows (chocolatey)
choco install eksctl awscli kubernetes-cli
```

---

## 4. Directory Structure to Create

```
.
├── Dockerfile                        # multi-stage API image (NEW)
├── compose.yml                       # 4-service stack (NEW)
├── entrypoint.sh                     # Airflow entrypoint (NEW)
├── requirements-airflow.txt          # Airflow extra deps (NEW)
├── airflow/
│   ├── Dockerfile                    # Airflow image (NEW)
│   ├── entrypoint.sh                 # Airflow entrypoint (NEW)
│   └── dags/
│       ├── arxiv_paper_ingestion.py  # daily DAG (NEW)
│       └── arxiv_ingestion/
│           ├── __init__.py           # package marker (NEW)
│           ├── common.py             # get_cached_services (NEW)
│           ├── setup.py              # setup_environment (NEW)
│           ├── fetching.py           # fetch_daily_papers (NEW)
│           ├── indexing.py           # index_papers_hybrid, verify_hybrid_index (NEW)
│           └── reporting.py          # generate_daily_report (NEW)
├── opensearch_dashboards/
│   └── opensearch_dashboards.yml     # dashboards config (NEW)
└── deployment/
    ├── eks/
    │   └── cluster.yaml              # eksctl ClusterConfig (NEW)
    ├── k8s/
    │   ├── api/
    │   │   ├── deployment.yaml       # rag-api Deployment (NEW)
    │   │   ├── hpa.yaml              # HorizontalPodAutoscaler (NEW)
    │   │   └── service.yaml          # LoadBalancer Service (NEW)
    │   ├── opensearch/
    │   │   └── statefulset.yaml      # OpenSearch StatefulSet (NEW)
    │   └── secrets/
    │       └── secret-template.yaml  # rag-app-secrets template (NEW)
    └── grafana/
        └── values.yaml               # Grafana Cloud k8s-monitoring (NEW)
```

---

## 5. Step-by-Step Implementation

### Step 1 — `Dockerfile` (API image)

Create the multi-stage Dockerfile. The `base` stage uses the official `uv` image to sync dependencies from `uv.lock` (which resolves CPU-only torch wheels), and the `final` stage is a slim Python runtime with Docling's system libraries.

Create [`Dockerfile`](Agentic-RAG-project-agentops/Dockerfile):

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

WORKDIR /app

# Copy configuration files
COPY pyproject.toml uv.lock ./

# UV_COMPILE_BYTECODE for generating .pyc files -> faster application startup.
# UV_LINK_MODE=copy to silence warnings about not being able to use hard links
# since the cache and sync target are on separate file systems.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Install dependencies.
# uv.lock is configured to resolve CPU-only torch wheels from PyTorch's CPU index,
# so no CUDA/NVIDIA libraries are pulled in (~2GB saved vs default PyPI CUDA wheels).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=/app/uv.lock \
    --mount=type=bind,source=pyproject.toml,target=/app/pyproject.toml \
    uv sync --frozen --no-dev

# Copy source code
COPY src /app/src

FROM python:3.12.8-slim AS final

EXPOSE 8000

# PYTHONUNBUFFERED=1 to disable output buffering
ENV PYTHONUNBUFFERED=1
ARG VERSION=0.1.0
ENV APP_VERSION=$VERSION

WORKDIR /app

# Install runtime system dependencies for Docling PDF parsing
# Docling's poppler bindings require X11 client libraries at runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libxcb1 \
        libx11-6 \
        libpq-dev \
        poppler-utils \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the base stage
COPY --from=base /app /app

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Explanation:** The `base` stage copies `pyproject.toml` and `uv.lock`, sets `UV_COMPILE_BYTECODE` and `UV_LINK_MODE`, and runs `uv sync --frozen --no-dev` with cache and bind mounts. The `final` stage starts from `python:3.12.8-slim`, installs Docling's runtime system libraries (`libxcb1`, `libx11-6`, `libpq-dev`, `poppler-utils`, `tesseract-ocr`, `libgl1`, `libglib2.0-0`), copies the entire `/app` (including the virtualenv) from `base`, adds `.venv/bin` to `PATH`, and runs uvicorn with 4 workers.

---

### Step 2 — `compose.yml`

Create the Docker Compose stack with 4 services. The `api` service depends on a healthy OpenSearch, the OpenSearch service runs single-node with security disabled, Dashboards provides a search UI, and Airflow runs the daily ingestion DAG.

Create [`compose.yml`](Agentic-RAG-project-agentops/compose.yml):

```yaml
name: agentic-rag-project

services:
  # FastAPI application
  api:
    build: .
    container_name: rag-api
    ports:
      - "8000:8000"
    depends_on:
      opensearch:
        condition: service_healthy
    healthcheck:
      test: [ "CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')\"" ]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    env_file:
      - .env
    environment:
      # Override OpenSearch host for container networking
      - OPENSEARCH__HOST=http://opensearch:9200
    dns:
      - 8.8.8.8
      - 1.1.1.1
    networks:
      - rag-network

  # OpenSearch — hybrid BM25 + vector search
  opensearch:
    image: opensearchproject/opensearch:2.19.5
    container_name: rag-opensearch
    environment:
      - discovery.type=single-node
      - OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m
      - DISABLE_SECURITY_PLUGIN=true
      - bootstrap.memory_lock=true
    ports:
      - "9200:9200"
      - "9600:9600"
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - opensearch_data:/usr/share/opensearch/data
    healthcheck:
      test: [ "CMD-SHELL", "curl -f http://localhost:9200/_cluster/health || exit 1" ]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
    restart: unless-stopped
    networks:
      - rag-network

  # OpenSearch Dashboards — search UI
  opensearch-dashboards:
    image: opensearchproject/opensearch-dashboards:2.19.5
    container_name: rag-dashboards
    ports:
      - "5601:5601"
    environment:
      - OPENSEARCH_HOSTS=http://opensearch:9200
      - DISABLE_SECURITY_DASHBOARDS_PLUGIN=true
    volumes:
      - ./opensearch_dashboards/opensearch_dashboards.yml:/usr/share/opensearch-dashboards/config/opensearch_dashboards.yml:ro
    depends_on:
      - opensearch
    healthcheck:
      test: [ "CMD-SHELL", "curl -f http://localhost:5601/api/status || exit 1" ]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
    networks:
      - rag-network

  # Airflow — daily arXiv ingestion DAG
  airflow:
    build:
      context: .
      dockerfile: airflow/Dockerfile
    container_name: rag-airflow
    depends_on:
      opensearch:
        condition: service_healthy
    env_file:
      - .env
    environment:
      - AIRFLOW_HOME=/opt/airflow
      - PYTHONPATH=/opt/airflow/src
      - OPENSEARCH__HOST=http://opensearch:9200
      # Suppress FutureWarnings emitted by Airflow's own internal config
      # migration code (core/sql_alchemy_conn → database/sql_alchemy_conn).
      # Our env var AIRFLOW__DATABASE__SQL_ALCHEMY_CONN is already correct.
      - PYTHONWARNINGS=ignore::FutureWarning:airflow,ignore::DeprecationWarning:airflow
      # UPDATE_FAB_PERMS=False: skip the slow perm sync inside gunicorn workers.
      # entrypoint.sh runs `airflow sync-perm` once before the webserver starts,
      # so permissions are already in Neon when gunicorn boots — no sync needed.
      - AIRFLOW__WEBSERVER__UPDATE_FAB_PERMS=False
      # Enable Basic Auth for the REST API (default is session/cookie auth which
      # only works in a browser). Without this, GET /api/v1/dags returns 401
      # even with correct credentials.
      - AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - airflow_logs:/opt/airflow/logs
      - ./airflow/plugins:/opt/airflow/plugins
      - ./src:/opt/airflow/src
    ports:
      - "8080:8080"
    dns:
      - 8.8.8.8
      - 1.1.1.1
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:8080/health" ]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 300s
    networks:
      - rag-network

volumes:
  opensearch_data:
  airflow_logs:

networks:
  rag-network:
    driver: bridge
```

**Explanation:** The `api` service builds from the root `Dockerfile`, maps port 8000, waits for a healthy OpenSearch, reads `.env`, and overrides `OPENSEARCH__HOST` to the container-network hostname `opensearch`. The `opensearch` service runs `2.19.5` single-node with security disabled, memlock unlocked, and a persistent volume. `opensearch-dashboards` connects to OpenSearch and mounts a read-only config. The `airflow` service builds from `airflow/Dockerfile`, bind-mounts `airflow/dags`, `airflow/plugins`, and `src`, enables Basic Auth for the REST API, and exposes port 8080. All services share the `rag-network` bridge.

---

### Step 3 — `entrypoint.sh`

Create the Airflow entrypoint script. It removes stale PID files, forces IPv4 for Neon Postgres, migrates the Airflow DB, syncs FAB permissions, creates the admin user, and starts the webserver (background) and scheduler (foreground).

Create [`entrypoint.sh`](Agentic-RAG-project-agentops/entrypoint.sh):

```bash
#!/bin/bash
set -e

# Remove stale PID files from any previous container run
rm -f ${AIRFLOW_HOME}/airflow-webserver.pid
rm -f ${AIRFLOW_HOME}/airflow-scheduler.pid

# Force IPv4 for Neon Postgres to avoid IPv6 routing failures inside Docker.
# psycopg2 supports hostaddr= (TCP IP) independent of host= (SSL SNI).
if [ -n "$AIRFLOW__DATABASE__SQL_ALCHEMY_CONN" ]; then
    NEON_HOST=$(python3 -c "
from urllib.parse import urlparse
import os
url = os.environ.get('AIRFLOW__DATABASE__SQL_ALCHEMY_CONN', '')
parsed = urlparse(url)
print(parsed.hostname or '')
" 2>/dev/null)

    if [ -n "$NEON_HOST" ]; then
        NEON_IPV4=$(python3 -c "
import socket, sys
try:
    results = socket.getaddrinfo('$NEON_HOST', None, socket.AF_INET)
    print(results[0][4][0])
except Exception as e:
    print('', end='')
" 2>/dev/null)

        if [ -n "$NEON_IPV4" ]; then
            echo "Airflow DB: forcing IPv4 $NEON_HOST -> $NEON_IPV4"
            export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="${AIRFLOW__DATABASE__SQL_ALCHEMY_CONN}&hostaddr=${NEON_IPV4}"
        fi
    fi
fi

# Initialise / migrate Airflow metadata database
echo "Initializing Airflow database..."
airflow db migrate

# Sync FAB permissions FIRST so roles (Admin, Viewer, etc.) exist before user create.
echo "Syncing Airflow FAB permissions..."
airflow sync-perm

# Create admin user (idempotent — skips silently if already exists)
echo "Creating admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin || echo "Admin user already exists"

# Start webserver in background (no --daemon to keep it as a child process),
# then run scheduler in foreground so Docker tracks the container's main process.
echo "Starting Airflow webserver and scheduler..."
airflow webserver --port 8080 &
airflow scheduler
```

**Explanation:** The script first removes stale PID files. It then resolves the Neon Postgres hostname to an IPv4 address and appends `hostaddr=` to the SQLAlchemy connection string to avoid IPv6 routing failures inside Docker. It runs `airflow db migrate`, then `airflow sync-perm` (so FAB roles exist before user creation), then creates the `admin` user idempotently. Finally it starts the webserver in the background and the scheduler in the foreground so Docker tracks the container's main process.

---

### Step 4 — Airflow DAG package: `common.py`

Create the shared service cache used by all DAG tasks. `get_cached_services` uses `@lru_cache` so services are initialized once per scheduler process.

Create [`common.py`](Agentic-RAG-project-agentops/airflow/dags/arxiv_ingestion/common.py):

```python
import logging
import sys
from functools import lru_cache
from typing import Any, Tuple

sys.path.insert(0, "/opt/airflow")

from src.db.factory import make_database
from src.services.arxiv.factory import make_arxiv_client
from src.services.metadata_fetcher import make_metadata_fetcher
from src.services.opensearch.factory import make_opensearch_client
from src.services.pdf_parser.factory import make_pdf_parser_service

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_cached_services() -> Tuple[Any, Any, Any, Any, Any]:
    """Get cached service instances using lru_cache for automatic memoization.

    :returns: Tuple of (arxiv_client, pdf_parser, database, metadata_fetcher, opensearch_client)
    """
    logger.info("Initializing services (cached with lru_cache)")

    # Initialize core services
    arxiv_client = make_arxiv_client()
    pdf_parser = make_pdf_parser_service()
    database = make_database()
    opensearch_client = make_opensearch_client()

    # Create metadata fetcher with dependencies
    metadata_fetcher = make_metadata_fetcher(arxiv_client, pdf_parser)

    logger.info("All services initialized and cached with lru_cache")
    return arxiv_client, pdf_parser, database, metadata_fetcher, opensearch_client
```

**Explanation:** `get_cached_services` inserts `/opt/airflow` into `sys.path` (so `src.*` imports resolve), then lazily constructs and caches the arXiv client, PDF parser, database, OpenSearch client, and metadata fetcher. The `@lru_cache(maxsize=1)` decorator ensures the services are created only once per scheduler process.

---

### Step 5 — Airflow DAG package: `setup.py`

Create the environment setup task that verifies the database and OpenSearch, and creates the hybrid index and RRF pipeline.

Create [`setup.py`](Agentic-RAG-project-agentops/airflow/dags/arxiv_ingestion/setup.py):

```python
import logging

from sqlalchemy import text

from .common import get_cached_services

logger = logging.getLogger(__name__)


def setup_environment():
    """Setup environment and verify dependencies.

    Creates hybrid search index with RRF pipeline.
    """
    logger.info("Setting up environment for arXiv paper ingestion")

    try:
        arxiv_client, _pdf_parser, database, _metadata_fetcher, opensearch_client = get_cached_services()

        with database.get_session() as session:
            session.execute(text("SELECT 1"))
            logger.info("Database connection verified")

        try:
            health = opensearch_client.client.cluster.health()
            if health["status"] in ["green", "yellow", "red"]:
                logger.info(f"OpenSearch hybrid client connected (cluster status: {health['status']})")
            else:
                raise Exception(f"OpenSearch cluster unhealthy: {health['status']}")
        except Exception as e:
            raise Exception(f"OpenSearch hybrid client connection failed: {e}")

        setup_results = opensearch_client.setup_indices(force=False)
        if setup_results.get("hybrid_index"):
            logger.info("Hybrid search index created with vector support")
        else:
            logger.info("Hybrid search index already exists")

        if setup_results.get("rrf_pipeline"):
            logger.info("RRF pipeline created successfully")
        else:
            logger.info("RRF pipeline already exists")

        logger.info("Hybrid search setup completed")

        logger.info(f"arXiv client ready: {arxiv_client.base_url}")
        logger.info("PDF parser service ready (Docling models cached)")

        return {"status": "success", "message": "Environment setup completed"}

    except Exception as e:
        error_msg = f"Environment setup failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)
```

**Explanation:** `setup_environment` verifies the database with a `SELECT 1`, checks the OpenSearch cluster health, and calls `setup_indices(force=False)` to create the hybrid index and RRF pipeline if they don't already exist. It returns a success dict, or raises on any failure so the DAG task fails visibly.

---

### Step 6 — Airflow DAG package: `fetching.py`

Create the paper fetching task. It uses a hardcoded keyword search (`transformer`) and monkey-patches `fetch_papers` so the metadata fetcher uses the keyword query instead of the default date-based query.

Create [`fetching.py`](Agentic-RAG-project-agentops/airflow/dags/arxiv_ingestion/fetching.py):

```python
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from .common import get_cached_services

logger = logging.getLogger(__name__)

# Hardcoded keyword search for sure-shot results during DAG ingestion.
# Searches for "transformer" in paper titles within the configured category.
HARDCODED_SEARCH_KEYWORD = "transformer"
HARDCODED_MAX_RESULTS = 2


async def run_paper_ingestion_pipeline(
    target_date: str,
    process_pdfs: bool = True,
) -> dict:
    """Async wrapper for the paper ingestion pipeline.

    :param target_date: Date to fetch papers for (YYYYMMDD format) — ignored;
                       we use a hardcoded keyword search instead.
    :param process_pdfs: Whether to download and process PDFs
    :returns: Dictionary with ingestion statistics
    """
    arxiv_client, _, database, metadata_fetcher, _ = get_cached_services()

    # Build hardcoded search query: e.g. "cat:cs.AI AND ti:transformer"
    search_query = f"cat:{arxiv_client.search_category} AND ti:{HARDCODED_SEARCH_KEYWORD}"
    logger.info(f"Using hardcoded keyword search: {search_query} (max_results={HARDCODED_MAX_RESULTS})")

    # Monkey-patch fetch_papers so MetadataFetcher uses our keyword query
    # instead of the default date-based query.
    original_fetch_papers = arxiv_client.fetch_papers

    async def _patched_fetch_papers(*args, **kwargs):
        return await arxiv_client.fetch_papers_with_query(
            search_query=search_query,
            max_results=HARDCODED_MAX_RESULTS,
            sort_by="relevance",
            sort_order="descending",
        )

    arxiv_client.fetch_papers = _patched_fetch_papers
    try:
        with database.get_session() as session:
            return await metadata_fetcher.fetch_and_process_papers(
                max_results=HARDCODED_MAX_RESULTS,
                from_date=None,
                to_date=None,
                process_pdfs=process_pdfs,
                store_to_db=True,
                db_session=session,
            )
    finally:
        arxiv_client.fetch_papers = original_fetch_papers


def fetch_daily_papers(**context):
    """Fetch daily papers from arXiv and store in PostgreSQL.

    This task:
    1. Fetches papers from arXiv API using a hardcoded keyword search
    2. Downloads and processes PDFs using Docling
    3. Stores metadata and parsed content in PostgreSQL

    Note: OpenSearch indexing is handled by a separate dedicated task
    """
    logger.info("Starting daily paper fetching task (keyword search mode)")

    target_date = datetime.now().strftime("%Y%m%d")
    logger.info(f"Run date (for reference): {target_date}")

    results = asyncio.run(
        run_paper_ingestion_pipeline(
            target_date=target_date,
            process_pdfs=True,
        )
    )

    logger.info(f"Daily fetch complete: {results['papers_fetched']} papers fetched")

    results["date"] = target_date
    ti = context.get("ti")
    if ti:
        ti.xcom_push(key="fetch_results", value=results)

    return results
```

**Explanation:** `run_paper_ingestion_pipeline` builds a keyword query like `cat:cs.AI AND ti:transformer`, monkey-patches `arxiv_client.fetch_papers` with a wrapper that calls `fetch_papers_with_query` (relevance-sorted), then runs `metadata_fetcher.fetch_and_process_papers` with PDF processing and DB storage enabled. The patch is restored in a `finally` block. `fetch_daily_papers` is the synchronous Airflow callable that runs the async pipeline via `asyncio.run` and pushes the results to XCom under the key `fetch_results`.

---

### Step 7 — Airflow DAG package: `indexing.py`

Create the hybrid indexing task that chunks papers, generates embeddings, and indexes them into OpenSearch, plus a verification task.

Create [`indexing.py`](Agentic-RAG-project-agentops/airflow/dags/arxiv_ingestion/indexing.py):

```python
import asyncio
import logging
from datetime import datetime, timedelta, timezone

from src.db.factory import make_database
from src.services.indexing.factory import make_hybrid_indexing_service
from src.services.opensearch.factory import make_opensearch_client_fresh

logger = logging.getLogger(__name__)


async def _index_papers_with_chunks(papers):
    """Async helper to index papers with chunking and embeddings."""
    indexing_service = make_hybrid_indexing_service()

    papers_data = []
    for paper in papers:
        if hasattr(paper, "__dict__"):
            paper_dict = {
                "id": str(paper.id),
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": paper.authors,
                "abstract": paper.abstract,
                "categories": paper.categories,
                "published_date": paper.published_date,
                "raw_text": paper.raw_text,
                "sections": paper.sections,
            }
        else:
            paper_dict = paper
        papers_data.append(paper_dict)

    stats = await indexing_service.index_papers_batch(papers=papers_data, replace_existing=True)

    return stats


def index_papers_hybrid(**context):
    """Index papers with chunking and vector embeddings for hybrid search.

    This task:
    1. Fetches recently processed papers from PostgreSQL
    2. Chunks them into overlapping segments (600 words, 100 overlap)
    3. Generates embeddings using Jina AI
    4. Indexes chunks with embeddings into OpenSearch
    """
    try:
        database = make_database()

        ti = context.get("ti")

        fetch_results = None
        if ti:
            fetch_results = ti.xcom_pull(task_ids="fetch_daily_papers", key="fetch_results")

        with database.get_session() as session:
            from src.models.paper import Paper

            if fetch_results and fetch_results.get("papers_stored", 0) > 0:
                from sqlalchemy import desc

                papers = session.query(Paper).order_by(desc(Paper.created_at)).limit(fetch_results["papers_stored"]).all()
            else:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=1)
                papers = session.query(Paper).filter(Paper.created_at >= cutoff_date).all()

            if not papers:
                logger.info("No papers to index for hybrid search")
                return {"papers_indexed": 0, "chunks_created": 0}

            logger.info(f"Indexing {len(papers)} papers for hybrid search")

            stats = asyncio.run(_index_papers_with_chunks(papers))

            logger.info(
                f"Hybrid indexing complete: {stats['papers_processed']} papers, "
                f"{stats['total_chunks_created']} chunks created, "
                f"{stats['total_chunks_indexed']} chunks indexed"
            )

            if ti:
                ti.xcom_push(key="hybrid_index_stats", value=stats)

            return stats

    except Exception as e:
        logger.error(f"Failed to index papers for hybrid search: {e}")
        raise


def verify_hybrid_index(**context):
    """Verify hybrid index health and get statistics."""
    try:
        opensearch_client = make_opensearch_client_fresh()

        stats = opensearch_client.client.indices.stats(index=opensearch_client.index_name)

        count = opensearch_client.client.count(index=opensearch_client.index_name)

        paper_count_query = {"aggs": {"unique_papers": {"cardinality": {"field": "arxiv_id"}}}, "size": 0}

        paper_count_response = opensearch_client.client.search(index=opensearch_client.index_name, body=paper_count_query)

        unique_papers = paper_count_response["aggregations"]["unique_papers"]["value"]

        result = {
            "index_name": opensearch_client.index_name,
            "total_chunks": count["count"],
            "unique_papers": unique_papers,
            "avg_chunks_per_paper": (count["count"] / unique_papers if unique_papers > 0 else 0),
            "index_size_mb": stats["indices"][opensearch_client.index_name]["total"]["store"]["size_in_bytes"] / (1024 * 1024),
        }

        logger.info(
            f"Hybrid index stats: {result['total_chunks']} chunks, "
            f"{result['unique_papers']} papers, "
            f"{result['avg_chunks_per_paper']:.1f} chunks/paper"
        )

        return result

    except Exception as e:
        logger.error(f"Failed to verify hybrid index: {e}")
        raise
```

**Explanation:** `index_papers_hybrid` pulls the fetch results from XCom, queries the most recently stored papers (or papers from the last day), converts each `Paper` ORM object into a dict, and calls `indexing_service.index_papers_batch(...)` with `replace_existing=True`. It pushes the stats to XCom under `hybrid_index_stats`. `verify_hybrid_index` uses a fresh OpenSearch client to compute total chunks, unique papers (via a cardinality aggregation on `arxiv_id`), average chunks per paper, and index size in MB.

---

### Step 8 — Airflow DAG package: `reporting.py`

Create the daily report task that aggregates statistics from all previous tasks.

Create [`reporting.py`](Agentic-RAG-project-agentops/airflow/dags/arxiv_ingestion/reporting.py):

```python
import json
import logging
from datetime import datetime

from .common import get_cached_services

logger = logging.getLogger(__name__)


def generate_daily_report(**context):
    """Generate a daily report of the ingestion pipeline results.

    Collects statistics from all previous tasks and generates a summary report.
    """
    logger.info("Generating daily ingestion report")

    ti = context.get("ti")
    if not ti:
        logger.warning("No task instance available, generating basic report")
        return {"status": "basic_report", "message": "No task instance for XCom data"}

    fetch_stats = ti.xcom_pull(task_ids="fetch_daily_papers", key="fetch_results") or {}
    hybrid_stats = ti.xcom_pull(task_ids="index_papers_hybrid", key="hybrid_index_stats") or {}

    report = {
        "execution_date": context.get("execution_date", datetime.now()).isoformat(),
        "fetch_statistics": {
            "papers_fetched": fetch_stats.get("papers_fetched", 0),
            "papers_stored": fetch_stats.get("papers_stored", 0),
            "target_date": fetch_stats.get("date", "unknown"),
        },
        "indexing_statistics": {
            "papers_processed": hybrid_stats.get("papers_processed", 0),
            "chunks_created": hybrid_stats.get("total_chunks_created", 0),
            "chunks_indexed": hybrid_stats.get("total_chunks_indexed", 0),
            "embeddings_generated": hybrid_stats.get("total_embeddings_generated", 0),
        },
        "pipeline_status": "success" if fetch_stats and hybrid_stats else "partial",
    }

    try:
        _arxiv_client, _pdf_parser, database, _metadata_fetcher, opensearch_client = get_cached_services()

        with database.get_session() as session:
            from sqlalchemy import func
            from src.models.paper import Paper

            total_papers = session.query(func.count(Paper.id)).scalar()
            report["database_statistics"] = {"total_papers": total_papers}

        if opensearch_client.health_check():
            try:
                stats_response = opensearch_client.client.indices.stats(index=opensearch_client.index_name)

                count_response = opensearch_client.client.count(index=opensearch_client.index_name)

                index_stats = stats_response["indices"][opensearch_client.index_name]["total"]

                report["opensearch_statistics"] = {
                    "index_name": opensearch_client.index_name,
                    "document_count": count_response["count"],
                    "index_size_mb": round(index_stats["store"]["size_in_bytes"] / (1024 * 1024), 2),
                }
            except Exception as stats_error:
                logger.error(f"Failed to get OpenSearch statistics: {stats_error}")
                report["opensearch_statistics"] = {"index_name": opensearch_client.index_name, "error": str(stats_error)}
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        report["error"] = str(e)

    logger.info("Daily Ingestion Report:")
    logger.info(json.dumps(report, indent=2))

    ti.xcom_push(key="daily_report", value=report)

    return report
```

**Explanation:** `generate_daily_report` pulls the fetch and hybrid-index stats from XCom, builds a summary report with fetch, indexing, and pipeline-status sections, then queries the database for the total paper count and OpenSearch for document count and index size. It logs the report as JSON and pushes it to XCom under the key `daily_report`. Any failure to gather statistics is captured in the `report["error"]` field rather than failing the whole task.

---

### Step 9 — Airflow DAG: `arxiv_paper_ingestion.py`

Create the DAG that orchestrates the modular tasks. It runs Monday–Friday at 6 AM UTC and chains: `setup_environment → fetch_daily_papers → index_papers_hybrid → generate_daily_report → cleanup_temp_files`.

Create [`arxiv_paper_ingestion.py`](Agentic-RAG-project-agentops/airflow/dags/arxiv_paper_ingestion.py):

```python
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from arxiv_ingestion.fetching import fetch_daily_papers
from arxiv_ingestion.indexing import index_papers_hybrid, verify_hybrid_index
from arxiv_ingestion.reporting import generate_daily_report

# Import task functions from modular structure
from arxiv_ingestion.setup import setup_environment

# Default DAG arguments
default_args = {
    "owner": "arxiv-curator",
    "depends_on_past": False,
    "start_date": datetime(2025, 8, 8),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=30),
    "catchup": False,
}

# Create the DAG
dag = DAG(
    "arxiv_paper_ingestion",
    default_args=default_args,
    description="Daily arXiv CS.AI paper pipeline: fetch → store to PostgreSQL → chunk & embed → hybrid OpenSearch indexing",
    schedule="0 6 * * 1-5",  # Monday-Friday at 6 AM UTC
    max_active_runs=1,
    catchup=False,
    tags=["arxiv", "papers", "ingestion", "hybrid-search", "embeddings", "chunks"],
)

# Task definitions
setup_task = PythonOperator(
    task_id="setup_environment",
    python_callable=setup_environment,
    dag=dag,
)

fetch_task = PythonOperator(
    task_id="fetch_daily_papers",
    python_callable=fetch_daily_papers,
    dag=dag,
)

# Hybrid search indexing task (replaces old OpenSearch task)
index_hybrid_task = PythonOperator(
    task_id="index_papers_hybrid",
    python_callable=index_papers_hybrid,
    dag=dag,
)

report_task = PythonOperator(
    task_id="generate_daily_report",
    python_callable=generate_daily_report,
    dag=dag,
)

cleanup_task = BashOperator(
    task_id="cleanup_temp_files",
    bash_command="""
    echo "Cleaning up temporary files..."
    # Remove PDFs older than 30 days to manage disk space
    find /tmp -name "*.pdf" -type f -mtime +30 -delete 2>/dev/null || true
    echo "Cleanup completed"
    """,
    dag=dag,
)

# Task dependencies
# Simplified pipeline: setup -> fetch -> hybrid index -> report -> cleanup
setup_task >> fetch_task >> index_hybrid_task >> report_task >> cleanup_task
```

**Explanation:** The DAG imports the task callables from the modular `arxiv_ingestion` package, sets `default_args` with 2 retries and a 30-minute retry delay, and schedules on `0 6 * * 1-5`. The task chain is `setup_environment → fetch_daily_papers → index_papers_hybrid → generate_daily_report → cleanup_temp_files`. The `cleanup_temp_files` task is a `BashOperator` that deletes PDFs older than 30 days from `/tmp`.

---

### Step 10 — Airflow `Dockerfile`

Create the Airflow image. It installs Airflow 2.10.3 with the official constraints file, CPU-only PyTorch from the CPU index, and the project's `requirements-airflow.txt`, then bakes in the application source.

Create [`airflow/Dockerfile`](Agentic-RAG-project-agentops/airflow/Dockerfile):

```dockerfile
FROM python:3.12-slim

# ── uv ────────────────────────────────────────────────────────────────────────
# Copy the uv binary from the official image — no pip install needed for uv itself.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ── env ───────────────────────────────────────────────────────────────────────
ENV AIRFLOW_HOME=/opt/airflow \
    AIRFLOW_VERSION=2.10.3 \
    PYTHON_VERSION=3.12

ENV CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# UV settings for Docker:
#   UV_LINK_MODE=copy   — Docker layers live on different filesystems; copy
#                          prevents "can't hardlink across filesystems" errors.
#   UV_COMPILE_BYTECODE — pre-compile .pyc files for faster container startup.
#   UV_PYTHON_DOWNLOADS — don't let uv download its own Python; we use the
#                          one from python:3.12-slim.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

# ── system deps ───────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libpq-dev \
        poppler-utils \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
        libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# ── airflow user ──────────────────────────────────────────────────────────────
RUN groupadd -r -g 50000 airflow && \
    useradd -r -u 50000 -g airflow -d ${AIRFLOW_HOME} -s /bin/bash airflow

RUN mkdir -p ${AIRFLOW_HOME}/dags ${AIRFLOW_HOME}/logs ${AIRFLOW_HOME}/plugins && \
    chown -R 50000:50000 ${AIRFLOW_HOME} && \
    chmod -R 755 ${AIRFLOW_HOME}

# ── Airflow core ──────────────────────────────────────────────────────────────
# Uses the official Airflow constraint file to pin every transitive dep.
# --mount=type=cache keeps the uv download cache warm across rebuilds.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        "apache-airflow[postgres]==${AIRFLOW_VERSION}" \
        --constraint "${CONSTRAINT_URL}" \
        psycopg2-binary

# ── CPU-only PyTorch ──────────────────────────────────────────────────────────
# Install torch from the CPU wheel index BEFORE requirements-airflow.txt so
# pip/uv's resolver sees torch already satisfied and never fetches the
# ~4 GB CUDA variant that PyPI serves on Linux by default.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu

# ── project requirements ──────────────────────────────────────────────────────
COPY requirements-airflow.txt /tmp/requirements-airflow.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        -r /tmp/requirements-airflow.txt

# ── application source ────────────────────────────────────────────────────────
# In Docker Compose, src/ is bind-mounted at runtime (./src:/opt/airflow/src).
# In Kubernetes, bind mounts from the host are not available — source code must
# be baked into the image at build time.
#
# The CD pipeline builds this image with the REPO ROOT as the build context:
#   docker build -f airflow/Dockerfile .
# This allows COPY src to find the src/ directory from the repo root.
#
# DAG files are also copied here as a fallback; the primary source in K8s is
# the airflow-dags ConfigMap (mounted at /opt/airflow/dags by the Deployment).
COPY src /opt/airflow/src
COPY airflow/dags /opt/airflow/dags

# ── entrypoint ────────────────────────────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER airflow
WORKDIR ${AIRFLOW_HOME}

EXPOSE 8080

CMD ["/entrypoint.sh"]
```

**Explanation:** The Airflow image copies the `uv` binary from the official image, installs Airflow 2.10.3 with the Postgres extra pinned by the official constraints file, then installs CPU-only PyTorch from the CPU wheel index *before* `requirements-airflow.txt` so the resolver never fetches the ~4 GB CUDA variant. It creates the `airflow` user (UID 50000), bakes in `src` and `airflow/dags`, copies the entrypoint, and runs it as the `airflow` user.

---

### Step 11 — Airflow `entrypoint.sh`

Create the Airflow entrypoint script. It is identical to the root `entrypoint.sh` and handles stale PID cleanup, Neon IPv4 forcing, DB migration, permission sync, admin user creation, and webserver/scheduler startup.

Create [`airflow/entrypoint.sh`](Agentic-RAG-project-agentops/airflow/entrypoint.sh):

```bash
#!/bin/bash
set -e

# Remove stale PID files from any previous container run
rm -f ${AIRFLOW_HOME}/airflow-webserver.pid
rm -f ${AIRFLOW_HOME}/airflow-scheduler.pid

# Force IPv4 for Neon Postgres to avoid IPv6 routing failures inside Docker.
# psycopg2 supports hostaddr= (TCP IP) independent of host= (SSL SNI).
if [ -n "$AIRFLOW__DATABASE__SQL_ALCHEMY_CONN" ]; then
    NEON_HOST=$(python3 -c "
from urllib.parse import urlparse
import os
url = os.environ.get('AIRFLOW__DATABASE__SQL_ALCHEMY_CONN', '')
parsed = urlparse(url)
print(parsed.hostname or '')
" 2>/dev/null)

    if [ -n "$NEON_HOST" ]; then
        NEON_IPV4=$(python3 -c "
import socket, sys
try:
    results = socket.getaddrinfo('$NEON_HOST', None, socket.AF_INET)
    print(results[0][4][0])
except Exception as e:
    print('', end='')
" 2>/dev/null)

        if [ -n "$NEON_IPV4" ]; then
            echo "Airflow DB: forcing IPv4 $NEON_HOST -> $NEON_IPV4"
            export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="${AIRFLOW__DATABASE__SQL_ALCHEMY_CONN}&hostaddr=${NEON_IPV4}"
        fi
    fi
fi

# Initialise / migrate Airflow metadata database
echo "Initializing Airflow database..."
airflow db migrate

# Sync FAB permissions FIRST so roles (Admin, Viewer, etc.) exist before user create.
echo "Syncing Airflow FAB permissions..."
airflow sync-perm

# Create admin user (idempotent — skips silently if already exists)
echo "Creating admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin || echo "Admin user already exists"

# Start webserver in background (no --daemon to keep it as a child process),
# then run scheduler in foreground so Docker tracks the container's main process.
echo "Starting Airflow webserver and scheduler..."
airflow webserver --port 8080 &
airflow scheduler
```

**Explanation:** This is the same script as the root `entrypoint.sh` (Step 3). It is copied into the Airflow image at `/entrypoint.sh` and set as the container `CMD`. It removes stale PID files, forces IPv4 for Neon, migrates the Airflow metadata DB, syncs FAB permissions, creates the `admin` user idempotently, and starts the webserver in the background with the scheduler in the foreground.

---

### Step 12 — EKS cluster: `cluster.yaml`

Create the eksctl `ClusterConfig` for the EKS cluster. It provisions a managed node group with OIDC enabled (required for IRSA) and CloudWatch control-plane logging.

Create [`deployment/eks/cluster.yaml`](Agentic-RAG-project-agentops/deployment/eks/cluster.yaml):

```yaml
# ============================================================
# eksctl Cluster Configuration for Agentic RAG
# Usage: eksctl create cluster -f deployment/eks/cluster.yaml
#
# What this creates:
#   - EKS control plane (managed by AWS, ~$73/month)
#   - One managed node group (2-4 x m5.xlarge)
#   - OIDC provider (required for IRSA — IAM Roles for Service Accounts)
#   - VPC, subnets, security groups via CloudFormation (automatic)
#
# Prerequisites:
#   brew install eksctl awscli kubectl
#   aws configure  (set region to us-east-1)
#
# After cluster creation, run:
#   eksctl create iamserviceaccount \
#     --cluster agentic-rag-cluster --namespace production --name rag-api-sa \
#     --attach-policy-arn arn:aws:iam::ACCOUNT_ID:policy/AgenticRAGBedrockPolicy \
#     --approve
# ============================================================
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  # Name used in kubectl contexts and all eksctl commands
  name: agentic-rag-cluster
  region: us-east-1
  version: "1.31"

# withOIDC enables the OIDC identity provider on the cluster.
# This is REQUIRED for IRSA (IAM Roles for Service Accounts) — the mechanism
# that lets pods assume IAM roles without embedding static AWS credentials.
iam:
  withOIDC: true

# Managed node groups: EC2 instances that EKS provisions and manages.
# EKS handles OS patching, AMI updates, and node replacement automatically.
managedNodeGroups:
  - name: rag-workers
    # m5.xlarge = 4 vCPU, 16 GB RAM
    # OpenSearch requires at least 2 GB heap (-Xms1g -Xmx1g) + OS + other pods.
    # m5.xlarge is the safe minimum for running OpenSearch + API on the same node.
    # To save cost during development: switch to t3.large (2 vCPU, 8 GB, ~$60/month)
    # but reduce OpenSearch heap to -Xms512m -Xmx512m.
    instanceType: m5.xlarge

    minSize: 2        # Keep 2 nodes for pod anti-affinity spread
    maxSize: 4        # HPA can add API pods; cluster autoscaler adds nodes
    desiredCapacity: 2

    # EBS volume per node. 50 GB is enough for container images + ephemeral data.
    # OpenSearch data lives on a separate PVC (see statefulset.yaml).
    volumeSize: 50

    # Nodes in private subnets — they don't have public IPs.
    # Traffic flows: Internet → ELB (public) → Node (private) → Pod.
    privateNetworking: true

    labels:
      role: worker
      environment: production

# Control plane logging. These CloudWatch log groups help debug cluster-level issues.
cloudWatch:
  clusterLogging:
    enableTypes:
      - api           # API server request logs
      - audit         # Kubernetes audit trail (who did what)
      - authenticator # IAM authentication logs
```

**Explanation:** The `ClusterConfig` creates an EKS cluster named `agentic-rag-cluster` in `us-east-1` at version 1.31. `iam.withOIDC: true` enables IRSA. The `rag-workers` managed node group uses `m5.xlarge` instances (4 vCPU / 16 GB) with a min of 2 and max of 4, private networking, and a 50 GB EBS volume per node. CloudWatch logging is enabled for `api`, `audit`, and `authenticator` log types.

---

### Step 13 — Kubernetes: API `deployment.yaml`

Create the `rag-api` Deployment. It uses IRSA (`rag-api-sa`), injects all secrets via `envFrom.secretRef`, waits for OpenSearch in an init container, and configures rolling updates, health probes, resource limits, and pod anti-affinity.

Create [`deployment/k8s/api/deployment.yaml`](Agentic-RAG-project-agentops/deployment/k8s/api/deployment.yaml):

```yaml
# ============================================================
# RAG API Deployment (FastAPI + LangGraph + Bedrock)
#
# Key design decisions:
#
# 1. serviceAccountName: rag-api-sa
#    IRSA (IAM Roles for Service Accounts) — the pod assumes an AWS IAM role
#    that has Bedrock permissions. AWS injects temporary credentials via a
#    projected token; no static keys are embedded in the pod.
#
# 2. envFrom.secretRef: rag-app-secrets
#    Injects ALL 25+ environment variables from the K8s Secret at once.
#    This is equivalent to docker compose's env_file: .env
#    The Secret key names match exactly what src/config.py (pydantic-settings) expects.
#
# 3. RollingUpdate with maxUnavailable: 0
#    Zero-downtime deployments. The new pod must pass its readinessProbe
#    before the old pod is terminated. Always 2+ pods serving traffic.
#
# 4. readinessProbe: GET /api/v1/health
#    This endpoint (src/routers/ping.py) checks Postgres, OpenSearch, and the LLM.
#    The pod only receives traffic once ALL its dependencies are healthy.
#
# 5. podAntiAffinity
#    Spreads the 2 replicas across different nodes. If one node fails,
#    the other replica continues serving traffic.
#
# Prerequisites:
#   - rag-app-secrets Secret must exist
#   - rag-api-sa ServiceAccount must exist (created by eksctl iamserviceaccount)
#   - OpenSearch must be ready (init container waits for it)
# ============================================================
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  namespace: production
  labels:
    app: rag-api
    app.kubernetes.io/part-of: agentic-rag
spec:
  replicas: 2   # HPA (hpa.yaml) scales this between 2 and 6
  selector:
    matchLabels:
      app: rag-api

  # ── Rolling Update Strategy ──────────────────────────────────────────────
  # maxUnavailable: 0  — Never terminate a pod until the replacement is ready
  # maxSurge: 1        — Allow 1 extra pod temporarily during rollout
  # Result: during a deploy, pods go 2→3→2 (never below 2)
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0
      maxSurge: 1

  template:
    metadata:
      labels:
        app: rag-api
        app.kubernetes.io/part-of: agentic-rag
    spec:
      # ── IRSA Service Account ─────────────────────────────────────────────
      # Created by: eksctl create iamserviceaccount --name rag-api-sa ...
      # The SA has an annotation: eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT:role/...
      # When this pod starts, AWS SDK automatically picks up the IRSA token.
      serviceAccountName: rag-api-sa

      # ── Init Container ────────────────────────────────────────────────────
      # Wait for OpenSearch to respond before starting the API.
      # The API's lifespan handler checks OpenSearch on startup and logs a warning
      # if it's not ready, but the init container prevents the pod from crash-looping.
      initContainers:
        - name: wait-for-opensearch
          image: busybox:1.36
          command:
            - sh
            - -c
            - |
              echo "Waiting for OpenSearch at http://opensearch:9200..."
              until wget -qO- http://opensearch:9200/_cluster/health; do
                echo "OpenSearch not ready, retrying in 5s..."
                sleep 5
              done
              echo "OpenSearch is ready. Starting RAG API."

      containers:
        - name: rag-api
          image: 685057748560.dkr.ecr.us-east-1.amazonaws.com/agentic-rag/api:latest
          imagePullPolicy: Always
          ports:
            - name: http
              containerPort: 8000

          # ── Inject All Secrets as Environment Variables ──────────────────
          # envFrom.secretRef injects the ENTIRE rag-app-secrets Secret as env vars.
          # Every key in the Secret becomes an environment variable in the container.
          # This is equivalent to: docker compose env_file: .env
          envFrom:
            - secretRef:
                name: rag-app-secrets

          # ── Override Specific Variables ──────────────────────────────────
          # OPENSEARCH__HOST: use the K8s Service DNS name ("opensearch").
          # This overrides the Secret value and matches how compose.yml works:
          #   environment: OPENSEARCH__HOST=http://opensearch:9200
          env:
            - name: OPENSEARCH__HOST
              value: "http://opensearch:9200"

          resources:
            requests:
              memory: "6Gi"    # uvicorn + LangGraph + boto3 baseline
              cpu: "500m"      # 0.5 CPU cores guaranteed
            limits:
              memory: "8Gi"    # PyTorch + LangGraph peaks at ~3.2GB; 8Gi prevents OOMKill under load
              cpu: "2000m"     # 2 CPU cores max (Bedrock calls are CPU-light but async)

          # ── Health Checks ─────────────────────────────────────────────────
          # /api/v1/health checks: Postgres connection, OpenSearch cluster, LLM API
          # See: src/routers/ping.py
          readinessProbe:
            httpGet:
              path: /api/v1/health
              port: 8000
            initialDelaySeconds: 40   # uvicorn + service init (DB + OpenSearch + SA) ~30s
            periodSeconds: 15
            timeoutSeconds: 10
            failureThreshold: 5

          livenessProbe:
            httpGet:
              path: /api/v1/health
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 30
            timeoutSeconds: 10
            failureThreshold: 3

      # ── Anti-Affinity ──────────────────────────────────────────────────────
      # "preferred" (not "required") — still schedules if only one node available,
      # but strongly prefers to place the 2 replicas on different nodes.
      # This is the Kubernetes equivalent of Docker Swarm's spread placement.
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                    - key: app
                      operator: In
                      values:
                        - rag-api
                topologyKey: kubernetes.io/hostname
```

**Explanation:** The Deployment runs 2 replicas of `rag-api` in the `production` namespace. It uses the `rag-api-sa` service account (IRSA), injects all secrets via `envFrom.secretRef`, and overrides `OPENSEARCH__HOST` to the K8s Service DNS name. A `wait-for-opensearch` init container polls OpenSearch before the API starts. Rolling updates use `maxUnavailable: 0` / `maxSurge: 1` for zero downtime. Readiness and liveness probes hit `/api/v1/health`. Resources request 6 GiB / 500m and limit 8 GiB / 2000m. Pod anti-affinity spreads replicas across nodes.

---

### Step 14 — Kubernetes: API `hpa.yaml`

Create the HorizontalPodAutoscaler that scales the API between 2 and 6 replicas based on CPU (70%) and memory (80%) utilization, with a 300-second scale-down stabilization window.

Create [`deployment/k8s/api/hpa.yaml`](Agentic-RAG-project-agentops/deployment/k8s/api/hpa.yaml):

```yaml
# ============================================================
# Horizontal Pod Autoscaler (HPA) for RAG API
#
# HPA automatically adjusts the number of API replicas based on
# observed CPU and memory utilization. This handles traffic spikes
# without over-provisioning at all times.
#
# How it works:
#   1. Metrics Server collects CPU/memory from pods every 15s
#   2. HPA compares actual utilization vs target (every 15s by default)
#   3. desired_replicas = ceil(current_replicas × (actual / target))
#   4. HPA adjusts the Deployment's replica count
#
# IMPORTANT: Install Metrics Server first:
#   kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
#
# Check HPA status:
#   kubectl get hpa -n production
#   # NAME          REFERENCE           TARGETS         MINPODS   MAXPODS   REPLICAS
#   # rag-api-hpa   Deployment/rag-api   42%/70%         2         6         2
# ============================================================
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
  namespace: production
  labels:
    app: rag-api
    app.kubernetes.io/part-of: agentic-rag
spec:
  # Target: the Deployment to scale
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api

  minReplicas: 2   # Always keep 2 pods (HA — survives one node failure)
  maxReplicas: 6   # Cap at 6 to control AWS costs

  metrics:
    # Scale up when average CPU across all API pods exceeds 70%
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70

    # Scale up when average memory exceeds 80%
    # LangGraph loads ML models into memory — watch this metric closely
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80

  # ── Scale-Down Behavior ────────────────────────────────────────────────
  # Bedrock/LangGraph calls spike suddenly and drop quickly.
  # Without a stabilization window, the HPA would thrash (add/remove pods rapidly).
  # 300s = wait 5 minutes of consistently low load before scaling down.
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Pods
          value: 1             # Remove at most 1 pod per scale-down event
          periodSeconds: 60    # At most once per minute
    scaleUp:
      stabilizationWindowSeconds: 0   # Scale up immediately (no delay)
      policies:
        - type: Pods
          value: 2             # Add up to 2 pods at a time
          periodSeconds: 60
```

**Explanation:** The HPA targets the `rag-api` Deployment with `minReplicas: 2` and `maxReplicas: 6`. It scales on both CPU (70% average utilization) and memory (80%). The `behavior` block adds a 300-second scale-down stabilization window (removing at most 1 pod per minute) to prevent thrashing, while scale-up is immediate (up to 2 pods per minute).

---

### Step 15 — Kubernetes: API `service.yaml`

Create the LoadBalancer Service that exposes the API externally. AWS provisions a Classic ELB that forwards port 80 to the pods' port 8000.

Create [`deployment/k8s/api/service.yaml`](Agentic-RAG-project-agentops/deployment/k8s/api/service.yaml):

```yaml
# ============================================================
# RAG API Service
#
# Type: LoadBalancer — AWS automatically provisions a Classic ELB.
# External clients connect to port 80 on the ELB hostname.
# The ELB forwards to port 8000 on the API pods.
#
# To find the URL after deployment:
#   kubectl get service rag-api -n production
#   Look for EXTERNAL-IP — the ELB hostname (takes ~2 min to provision)
#
# Example:
#   export API=abc123.us-east-1.elb.amazonaws.com
#   curl http://$API/api/v1/health | jq .
#   curl -X POST http://$API/api/v1/ask-agentic \
#     -H "Content-Type: application/json" \
#     -d '{"query": "What is vector policy?"}'
# ============================================================
apiVersion: v1
kind: Service
metadata:
  name: rag-api
  namespace: production
  labels:
    app: rag-api
    app.kubernetes.io/part-of: agentic-rag
  annotations:
    # Keep ALB connections alive for 5 minutes.
    # LangGraph agentic responses can take 30-60s for multi-step retrieval.
    service.beta.kubernetes.io/aws-load-balancer-connection-idle-timeout: "300"
    # ALB health check path — ELB pings this to decide if pods are healthy
    service.beta.kubernetes.io/aws-load-balancer-healthcheck-path: "/api/v1/health"
spec:
  type: LoadBalancer
  selector:
    app: rag-api   # Routes to pods with this label
  ports:
    - name: http
      port: 80           # External port — clients connect here
      targetPort: 8000   # Container port — uvicorn listens here
      protocol: TCP
```

**Explanation:** The Service is `type: LoadBalancer`, selecting pods with the `app: rag-api` label. It maps external port 80 to container port 8000. The annotations set a 300-second connection idle timeout (for long agentic responses) and point the ELB health check at `/api/v1/health`.

---

### Step 16 — Kubernetes: OpenSearch `statefulset.yaml`

Create the OpenSearch StatefulSet. It uses a headless service, a privileged init container to set `vm.max_map_count`, and a `volumeClaimTemplates` entry for persistent storage.

Create [`deployment/k8s/opensearch/statefulset.yaml`](Agentic-RAG-project-agentops/deployment/k8s/opensearch/statefulset.yaml):

```yaml
# ============================================================
# OpenSearch StatefulSet
#
# WHY StatefulSet (not Deployment)?
#   StatefulSets give each pod a STABLE identity: opensearch-0, opensearch-1, ...
#   This means:
#     1. Ordered, graceful startup/shutdown (important for cluster bootstrapping)
#     2. Stable DNS: opensearch-0.opensearch-headless.production.svc.cluster.local
#     3. volumeClaimTemplates: each pod automatically gets its own PVC
#        (opensearch-data-opensearch-0) that persists across pod restarts
#
#   A Deployment would require a manually created PVC with a claimName, and
#   the pod could land on any node — potentially one that can't mount the EBS
#   volume attached in a different AZ. StatefulSet handles this correctly.
#
# Usage:
#   kubectl apply -f deployment/k8s/opensearch/service.yaml   # service first
#   kubectl apply -f deployment/k8s/opensearch/statefulset.yaml
#   kubectl rollout status statefulset/opensearch -n production
# ============================================================
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: opensearch
  namespace: production
  labels:
    app: opensearch
    app.kubernetes.io/part-of: agentic-rag
spec:
  # serviceName must match the headless Service (opensearch-headless).
  # The StatefulSet uses this to register DNS entries for each pod.
  serviceName: opensearch-headless
  replicas: 1  # Single-node for simplicity. Scale to 3 for production HA.

  selector:
    matchLabels:
      app: opensearch

  template:
    metadata:
      labels:
        app: opensearch
        app.kubernetes.io/part-of: agentic-rag
    spec:
      # fsGroup ensures EBS volumes are chowned to GID 1000 (opensearch) on mount.
      # Without this, the volume is owned by root and OpenSearch (runAsUser: 1000) cannot write.
      securityContext:
        fsGroup: 1000

      # ── Init Container ─────────────────────────────────────────────────────
      # OpenSearch requires vm.max_map_count >= 262144 for Lucene memory-mapped files.
      # Regular containers cannot set kernel parameters, but a privileged init
      # container running as root can write to /proc/sys before the main container starts.
      initContainers:
        - name: fix-kernel-settings
          image: busybox:1.36
          securityContext:
            privileged: true   # Needed to write to /proc/sys
          command:
            - sh
            - -c
            - |
              # Increase virtual memory map count — required by Lucene / OpenSearch
              sysctl -w vm.max_map_count=262144
              echo "vm.max_map_count set to: $(cat /proc/sys/vm/max_map_count)"

      # ── Main Container ─────────────────────────────────────────────────────
      containers:
        - name: opensearch
          image: opensearchproject/opensearch:2.19.5
          ports:
            - name: http
              containerPort: 9200   # REST API — used by rag-api and airflow
            - name: transport
              containerPort: 9300   # Cluster-internal transport (not exposed externally)
            - name: performance
              containerPort: 9600   # Performance Analyzer

          securityContext:
            allowPrivilegeEscalation: false
            runAsNonRoot: true
            runAsUser: 1000   # opensearch user inside the container

          env:
            # Single-node mode: do not wait for other members before starting
            - name: discovery.type
              value: "single-node"
            # JVM heap: set -Xms and -Xmx to the SAME value to prevent heap resizing.
            # 1g is safe for m5.xlarge (16 GB node). Raise to 2g if you have high query load.
            # Rule of thumb: heap should not exceed 50% of available node RAM.
            - name: OPENSEARCH_JAVA_OPTS
              value: "-Xms1g -Xmx1g"
            # Disable the security plugin — simplifies setup for learning.
            # In a real production deployment, enable TLS + authentication.
            - name: DISABLE_SECURITY_PLUGIN
              value: "true"
            # Lock memory pages to prevent OS swapping — critical for search performance.
            # Combined with resource limits below, this is equivalent to
            # ulimits.memlock: soft/hard = -1 in Docker Compose.
            - name: bootstrap.memory_lock
              value: "true"

          # ── Resource Requests and Limits ────────────────────────────────────
          # requests: guaranteed resources — Kubernetes will not schedule this pod
          #           on a node that can't provide these.
          # limits:   hard cap — pod is OOMKilled if it exceeds memory limit.
          resources:
            requests:
              memory: "2Gi"    # 1g heap + 1g for OS/JVM overhead
              cpu: "500m"      # 0.5 CPU cores guaranteed
            limits:
              memory: "3Gi"    # Kill if exceeds 3 GB (protects the node)
              cpu: "2000m"     # 2 CPU cores max

          volumeMounts:
            # Mount the persistent volume at OpenSearch's data directory.
            # This is where all indices, segments, and WAL files are stored.
            - name: opensearch-data
              mountPath: /usr/share/opensearch/data

          # ── Health Checks ────────────────────────────────────────────────────
          # livenessProbe: Kubernetes restarts the container if this fails.
          # OpenSearch takes ~60-90s to start — set initialDelaySeconds accordingly.
          livenessProbe:
            httpGet:
              path: /_cluster/health
              port: 9200
            initialDelaySeconds: 90
            periodSeconds: 30
            timeoutSeconds: 10
            failureThreshold: 5

          # readinessProbe: Kubernetes removes the pod from Service endpoints if this fails.
          # No traffic is sent to the pod until /_cluster/health returns yellow or green.
          # "yellow" means all primary shards are active (good enough for single-node).
          readinessProbe:
            httpGet:
              path: /_cluster/health?wait_for_status=yellow&timeout=30s
              port: 9200
            initialDelaySeconds: 60
            periodSeconds: 15
            timeoutSeconds: 30
            failureThreshold: 10

  # ── Volume Claim Templates ────────────────────────────────────────────────
  # This is the key StatefulSet feature: each replica automatically gets its own PVC.
  # For replicas=1, this creates PVC "opensearch-data-opensearch-0".
  # The PVC persists even if the pod is deleted — your index data survives pod restarts.
  # To delete the data: kubectl delete pvc opensearch-data-opensearch-0 -n production
  volumeClaimTemplates:
    - metadata:
        name: opensearch-data
        labels:
          app: opensearch
      spec:
        accessModes:
          - ReadWriteOnce   # Only one node can mount this EBS volume at a time (correct for single-node)
        storageClassName: gp3   # AWS EBS gp3 via EBS CSI driver (required on EKS 1.23+)
        resources:
          requests:
            storage: 20Gi   # 20 GB for arXiv papers index + vectors + BM25 segments
```

**Explanation:** The StatefulSet runs a single OpenSearch node with a stable identity. The `fix-kernel-settings` privileged init container sets `vm.max_map_count=262144` (required by Lucene). The main container runs as UID 1000 with `fsGroup: 1000` so the EBS volume is writable. It uses `discovery.type=single-node`, a 1 GB JVM heap, security disabled, and memory locking. Liveness and readiness probes hit `/_cluster/health`. The `volumeClaimTemplates` entry provisions a 20 GiB `gp3` PVC named `opensearch-data-opensearch-0` that persists across pod restarts.

---

### Step 17 — Kubernetes: `secret-template.yaml`

Create the Secret template. It is a reference template (never commit real values) that defines all 25+ environment variables the application expects, matching the `*__*` nested config keys in `Settings`.

Create [`deployment/k8s/secrets/secret-template.yaml`](Agentic-RAG-project-agentops/deployment/k8s/secrets/secret-template.yaml):

```yaml
# ============================================================
# Kubernetes Secret Template — DO NOT COMMIT WITH REAL VALUES
#
# This file is a REFERENCE TEMPLATE. Never store real secrets here.
# Kubernetes Secrets are only base64-encoded, not encrypted in etcd by default.
#
# HOW TO CREATE THE REAL SECRET:
#
#   Option A — kubectl (simplest for students):
#     kubectl create secret generic rag-app-secrets \
#       --namespace production \
#       --from-literal=POSTGRES_DATABASE_URL="postgresql+psycopg2://user:pass@host.neon.tech/db" \
#       --from-literal=REDIS__URL="rediss://default:token@host.upstash.io:6379" \
#       ... (repeat for every key below) \
#       --dry-run=client -o yaml | kubectl apply -f -
#
#   The --dry-run=client | apply pattern is idempotent:
#   it creates the secret if it doesn't exist, or updates it if it does.
#
#   Option B — CI/CD (automated, see .github/workflows/cd.yml):
#     GitHub Actions reads secrets from repository Secrets and creates the
#     K8s Secret automatically on every deployment.
#
# WHY envFrom.secretRef (not individual env.valueFrom)?
#   The app has 25+ env vars. Listing each individually would make deployment
#   YAMLs very long. envFrom.secretRef injects the ENTIRE Secret as environment
#   variables — one line instead of 25.
# ============================================================
apiVersion: v1
kind: Secret
metadata:
  name: rag-app-secrets
  namespace: production
  labels:
    app.kubernetes.io/part-of: agentic-rag
# Opaque = generic secret (as opposed to kubernetes.io/tls, kubernetes.io/dockerconfigjson)
type: Opaque
# stringData: Kubernetes base64-encodes these values automatically.
# Replace PLACEHOLDER with real values only when creating via kubectl.
stringData:
  # ── Application ─────────────────────────────────────────────────────────
  DEBUG: "false"
  ENVIRONMENT: "production"
  # "bedrock" uses AWS Bedrock for LLM; "openai" uses OpenAI API
  PROVIDER: "bedrock"

  # ── OpenSearch (K8s ClusterIP service name — no placeholder needed) ─────
  # "opensearch" is the Kubernetes Service name defined in k8s/opensearch/service.yaml
  # K8s DNS resolves "opensearch" to the ClusterIP automatically within the namespace
  OPENSEARCH__HOST: "http://opensearch:9200"
  OPENSEARCH__INDEX_NAME: "arxiv-papers"
  OPENSEARCH__CHUNK_INDEX_SUFFIX: "chunks"
  OPENSEARCH__VECTOR_DIMENSION: "1024"

  # ── PostgreSQL (Neon serverless — external, not deployed in K8s) ─────────
  POSTGRES_DATABASE_URL: "PLACEHOLDER"

  # ── Redis (Upstash — external, not deployed in K8s) ──────────────────────
  REDIS__URL: "PLACEHOLDER"
  REDIS__TTL_HOURS: "6"

  # ── AWS Bedrock LLM ──────────────────────────────────────────────────────
  # These credentials are also available via IRSA (IAM role on service account),
  # but the app currently reads them explicitly via pydantic-settings.
  BEDROCK__AWS_ACCESS_KEY_ID: "PLACEHOLDER"
  BEDROCK__AWS_SECRET_ACCESS_KEY: "PLACEHOLDER"
  BEDROCK__AWS_REGION: "us-east-1"
  # Full ARN of the inference profile (e.g. arn:aws:bedrock:us-east-1:ACCOUNT:inference-profile/...)
  BEDROCK__MODEL_ID: "PLACEHOLDER"
  # From: uv run python scripts/create_bedrock_guardrail.py
  BEDROCK__GUARDRAIL_ID: "PLACEHOLDER"
  BEDROCK__GUARDRAIL_VERSION: "DRAFT"

  # ── Langfuse (observability tracing — https://us.cloud.langfuse.com) ─────
  LANGFUSE__PUBLIC_KEY: "PLACEHOLDER"
  LANGFUSE__SECRET_KEY: "PLACEHOLDER"
  LANGFUSE__HOST: "https://us.cloud.langfuse.com"
  LANGFUSE__ENABLED: "true"
  LANGFUSE__FLUSH_AT: "15"
  LANGFUSE__FLUSH_INTERVAL: "1.0"

  # ── Logfire (structured logging — https://logfire.pydantic.dev) ──────────
  LOGFIRE__TOKEN: "PLACEHOLDER"
  LOGFIRE__ENABLED: "true"
  LOGFIRE__SERVICE_NAME: "arxiv-rag"
  LOGFIRE__ENVIRONMENT: "production"

  # ── Jina AI embeddings (https://jina.ai) ─────────────────────────────────
  JINA_API_KEY: "PLACEHOLDER"

  # ── OpenAI (fallback provider when PROVIDER=openai) ──────────────────────
  OPENAI_API_KEY: "PLACEHOLDER"
  OPENAI_MODEL: "gpt-4o-mini"
  OPENAI_TIMEOUT: "300"

  # ── Telegram Bot ────────────────────────────────────────────────────────
  TELEGRAM__ENABLED: "true"
  TELEGRAM__BOT_TOKEN: "PLACEHOLDER"

  # ── Airflow ──────────────────────────────────────────────────────────────
  # Same value as POSTGRES_DATABASE_URL — Airflow uses the same Neon DB
  AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: "PLACEHOLDER"
  AIRFLOW__CORE__EXECUTOR: "LocalExecutor"
  # Random string — used to sign Airflow session cookies. Generate with:
  #   python -c "import secrets; print(secrets.token_hex(32))"
  AIRFLOW__WEBSERVER__SECRET_KEY: "PLACEHOLDER"
  AIRFLOW__API__AUTH_BACKENDS: "airflow.api.auth.backend.basic_auth"
  AIRFLOW__WEBSERVER__UPDATE_FAB_PERMS: "False"
  PYTHONWARNINGS: "ignore::FutureWarning:airflow,ignore::DeprecationWarning:airflow"

  # ── arXiv API overrides (production) ────────────────────────────────────
  ARXIV__MAX_RESULTS: "2"
  ARXIV__TIMEOUT_SECONDS: "120"
  ARXIV__RATE_LIMIT_DELAY: "5.0"
  ARXIV__BASE_URL: "https://export.arxiv.org/api/query"

  # ── PDF Parser overrides ────────────────────────────────────────────────
  PDF_PARSER__MAX_PAGES: "60"
  PDF_PARSER__DO_OCR: "false"

  # ── Text Chunking overrides ─────────────────────────────────────────────
  CHUNKING__CHUNK_SIZE: "600"
  CHUNKING__OVERLAP_SIZE: "100"
```

**Explanation:** The Secret template uses `stringData` (Kubernetes base64-encodes automatically) and defines every environment variable the application reads via pydantic-settings, using the exact `*__*` nested keys. The `OPENSEARCH__HOST` uses the K8s Service DNS name `opensearch`. External services (Neon Postgres, Upstash Redis, Bedrock, Langfuse, Logfire, Jina, OpenAI, Telegram) use `PLACEHOLDER` values that must be replaced when creating the real Secret via `kubectl create secret generic` or CI/CD.

---

### Step 18 — Grafana Cloud: `values.yaml`

Create the Grafana Cloud k8s-monitoring Helm values. It configures Prometheus metrics and Loki logs destinations, plus cluster, host, cost, event, and pod-log collection.

Create [`deployment/grafana/values.yaml`](Agentic-RAG-project-agentops/deployment/grafana/values.yaml):

```yaml
cluster:
  name: agentic-rag-cluster

destinations:
  grafana-cloud-metrics:
    type: prometheus
    url: https://prometheus-prod-43-prod-ap-south-1.grafana.net/api/prom/push
    auth:
      type: basic
      username: "3267520"
      password: "YOUR_GRAFANA_CLOUD_TOKEN"
  grafana-cloud-logs:
    type: loki
    url: https://logs-prod-028.grafana.net/loki/api/v1/push
    auth:
      type: basic
      username: "1629435"
      password: "YOUR_GRAFANA_CLOUD_TOKEN"

clusterMetrics:
  enabled: true
  collector: alloy-metrics

hostMetrics:
  enabled: true
  collector: alloy-metrics
  linuxHosts:
    enabled: true
  windowsHosts:
    enabled: false
  energyMetrics:
    enabled: true

costMetrics:
  enabled: true
  collector: alloy-metrics

clusterEvents:
  enabled: true
  collector: alloy-singleton

podLogsViaLoki:
  enabled: true
  collector: alloy-logs

collectors:
  alloy-metrics:
    presets:
      - clustered
      - statefulset
  alloy-singleton:
    presets:
      - singleton
  alloy-logs:
    presets:
      - filesystem-log-reader
      - daemonset

telemetryServices:
  kube-state-metrics:
    deploy: true
  node-exporter:
    deploy: true
  windows-exporter:
    deploy: false
  opencost:
    deploy: true
    metricsSource: grafana-cloud-metrics
    opencost:
      exporter:
        defaultClusterId: agentic-rag-cluster
      prometheus:
        existingSecretName: grafana-cloud-metrics-grafana-k8s-monitoring
        external:
          url: https://prometheus-prod-43-prod-ap-south-1.grafana.net/api/prom
  kepler:
    deploy: true
```

**Explanation:** The `values.yaml` configures the Grafana Cloud k8s-monitoring Helm chart. It defines two destinations: `grafana-cloud-metrics` (Prometheus remote-write) and `grafana-cloud-logs` (Loki push). It enables cluster metrics, host metrics (Linux + energy), cost metrics (OpenCost), cluster events, and pod logs via Loki. The collectors use presets (`clustered`, `statefulset`, `singleton`, `filesystem-log-reader`, `daemonset`), and telemetry services deploy `kube-state-metrics`, `node-exporter`, `opencost`, and `kepler`.

---

## 6. Configuration

The infrastructure phase relies on environment variables already defined in PHASE 2 (`Settings`) and the `.env` file. The containerized and Kubernetes deployments consume these via:

- **Docker Compose**: the `api` and `airflow` services read `.env` via `env_file`, with `OPENSEARCH__HOST` overridden to the container-network hostname `http://opensearch:9200`.
- **Kubernetes**: the `rag-app-secrets` Secret (Step 17) injects all variables via `envFrom.secretRef`, with `OPENSEARCH__HOST` overridden in the Deployment to `http://opensearch:9200`.

Key configuration values used in this phase:

| Setting | Value | Purpose |
|---------|-------|---------|
| `OPENSEARCH__HOST` | `http://opensearch:9200` | Container/K8s DNS name for OpenSearch |
| `AIRFLOW__API__AUTH_BACKENDS` | `airflow.api.auth.backend.basic_auth` | Enable Basic Auth for the Airflow REST API |
| `AIRFLOW__WEBSERVER__UPDATE_FAB_PERMS` | `False` | Skip slow perm sync in gunicorn workers (done once in entrypoint) |
| `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` | Neon Postgres URL | Airflow metadata DB (same as `POSTGRES_DATABASE_URL`) |
| `AIRFLOW__CORE__EXECUTOR` | `LocalExecutor` | Single-node executor |
| `DISABLE_SECURITY_PLUGIN` | `true` | OpenSearch security plugin off (learning setup) |
| `OPENSEARCH_JAVA_OPTS` | `-Xms1g -Xmx1g` | OpenSearch JVM heap (K8s) |
| `ARXIV__MAX_RESULTS` | `2` | Hardcoded ingestion volume |

---

## 7. Verification

### Docker Compose

```bash
# Build and start the full stack
docker compose up --build -d

# Check service health
docker compose ps

# API health endpoint
curl http://localhost:8000/api/v1/health

# OpenSearch cluster health
curl http://localhost:9200/_cluster/health

# OpenSearch Dashboards
open http://localhost:5601

# Airflow UI (login: admin / admin)
open http://localhost:8080
```

### Airflow DAG

```bash
# List DAGs
docker compose exec airflow airflow dags list

# Trigger the ingestion DAG manually
docker compose exec airflow airflow dags trigger arxiv_paper_ingestion

# Check DAG runs
docker compose exec airflow airflow dags list-runs -d arxiv_paper_ingestion
```

### Kubernetes

```bash
# Create the EKS cluster
eksctl create cluster -f deployment/eks/cluster.yaml

# Create the IRSA service account for Bedrock
eksctl create iamserviceaccount \
  --cluster agentic-rag-cluster --namespace production --name rag-api-sa \
  --attach-policy-arn arn:aws:iam::ACCOUNT_ID:policy/AgenticRAGBedrockPolicy \
  --approve

# Create the namespace and Secret
kubectl create namespace production
kubectl create secret generic rag-app-secrets --namespace production \
  --from-literal=POSTGRES_DATABASE_URL="..." \
  --dry-run=client -o yaml | kubectl apply -f -

# Apply the manifests
kubectl apply -f deployment/k8s/opensearch/service.yaml
kubectl apply -f deployment/k8s/opensearch/statefulset.yaml
kubectl apply -f deployment/k8s/api/deployment.yaml
kubectl apply -f deployment/k8s/api/service.yaml
kubectl apply -f deployment/k8s/api/hpa.yaml

# Verify rollout and status
kubectl rollout status deployment/rag-api -n production
kubectl get pods -n production
kubectl get hpa -n production
kubectl get svc rag-api -n production   # EXTERNAL-IP is the ELB hostname
```

---

## 8. Common Pitfalls

1. **Docling runtime libraries missing**: The API image must install `libxcb1`, `libx11-6`, `libpq-dev`, `poppler-utils`, `tesseract-ocr`, `libgl1`, and `libglib2.0-0` or PDF parsing fails at runtime with X11/poppler errors.
2. **CUDA torch pulled into the image**: `uv.lock` must resolve CPU-only torch wheels from the PyTorch CPU index; otherwise the image balloons by ~4 GB. The Airflow image installs CPU torch *before* `requirements-airflow.txt` so the resolver never fetches the CUDA variant.
3. **Airflow REST API returns 401**: Without `AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth`, the API uses session/cookie auth that only works in a browser.
4. **Airflow FAB permission errors on user create**: Run `airflow sync-perm` *before* `airflow users create` so the Admin/Viewer roles exist.
5. **Neon Postgres IPv6 routing failures**: The entrypoint resolves the Neon hostname to IPv4 and appends `hostaddr=` to the connection string.
6. **OpenSearch won't start in K8s**: `vm.max_map_count` must be ≥ 262144. The privileged `fix-kernel-settings` init container sets it before the main container starts.
7. **OpenSearch can't write to EBS volume**: The StatefulSet needs `fsGroup: 1000` and `runAsUser: 1000` so the volume is owned by the `opensearch` user.
8. **HPA not scaling**: Install the Metrics Server first; without it the HPA has no CPU/memory metrics.
9. **IRSA not working**: The cluster must be created with `iam.withOIDC: true`, and the `rag-api-sa` service account must exist before the Deployment is applied.
10. **Stale Airflow PID files**: Remove `airflow-webserver.pid` and `airflow-scheduler.pid` at container start or Airflow fails to bind its ports.

---

## 9. Definition of Done

- [ ] [`Dockerfile`](Agentic-RAG-project-agentops/Dockerfile) builds a multi-stage API image with CPU-only torch and Docling system libraries.
- [ ] [`compose.yml`](Agentic-RAG-project-agentops/compose.yml) starts all 4 services (`api`, `opensearch`, `opensearch-dashboards`, `airflow`) with healthchecks.
- [ ] [`entrypoint.sh`](Agentic-RAG-project-agentops/entrypoint.sh) migrates the Airflow DB, syncs permissions, creates the admin user, and starts webserver + scheduler.
- [ ] The Airflow DAG [`arxiv_paper_ingestion.py`](Agentic-RAG-project-agentops/airflow/dags/arxiv_paper_ingestion.py) and the modular `arxiv_ingestion` package (`common`, `setup`, `fetching`, `indexing`, `reporting`) are created.
- [ ] [`airflow/Dockerfile`](Agentic-RAG-project-agentops/airflow/Dockerfile) and [`airflow/entrypoint.sh`](Agentic-RAG-project-agentops/airflow/entrypoint.sh) build a working Airflow image.
- [ ] [`deployment/eks/cluster.yaml`](Agentic-RAG-project-agentops/deployment/eks/cluster.yaml) provisions the EKS cluster with OIDC and a managed node group.
- [ ] [`deployment/k8s/api/deployment.yaml`](Agentic-RAG-project-agentops/deployment/k8s/api/deployment.yaml), [`hpa.yaml`](Agentic-RAG-project-agentops/deployment/k8s/api/hpa.yaml), and [`service.yaml`](Agentic-RAG-project-agentops/deployment/k8s/api/service.yaml) deploy the API with autoscaling and a LoadBalancer.
- [ ] [`deployment/k8s/opensearch/statefulset.yaml`](Agentic-RAG-project-agentops/deployment/k8s/opensearch/statefulset.yaml) runs OpenSearch with persistent storage.
- [ ] [`deployment/k8s/secrets/secret-template.yaml`](Agentic-RAG-project-agentops/deployment/k8s/secrets/secret-template.yaml) defines all required environment variables.
- [ ] [`deployment/grafana/values.yaml`](Agentic-RAG-project-agentops/deployment/grafana/values.yaml) configures Grafana Cloud metrics and logs collection.
- [ ] `docker compose up --build -d` brings the full stack up with all services healthy.
- [ ] The Airflow DAG runs successfully and indexes papers into OpenSearch.
- [ ] The API is reachable via the LoadBalancer and `/api/v1/health` returns healthy.