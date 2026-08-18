# PHASE 6 — Embeddings

> **Build guide for the from-scratch reconstruction of the arXiv Paper Curator (Agentic RAG).**
>
> This phase implements the embeddings layer using the **Jina AI** embeddings API. It provides a [`JinaEmbeddingsClient`](../src/services/embeddings/jina_client.py) that generates 1024-dimensional embeddings for passages and queries using the `jina-embeddings-v3` model, plus the Pydantic request/response schemas and factory functions used throughout the application.

---

## 1. Phase Objective

By the end of this phase you will have:

- A [`JinaEmbeddingsClient`](../src/services/embeddings/jina_client.py) that embeds passages (in configurable batches) and single queries.
- Pydantic schemas [`JinaEmbeddingRequest`](../src/schemas/embeddings/jina.py) and [`JinaEmbeddingResponse`](../src/schemas/embeddings/jina.py) for typed API interaction.
- Factory functions [`make_embeddings_service`](../src/services/embeddings/factory.py) and [`make_embeddings_client`](../src/services/embeddings/factory.py) that create fresh client instances.
- The `JINA_API_KEY` configuration wired into the client.

These embeddings are consumed by the OpenSearch vector/hybrid search (PHASE 7) and the hybrid indexer (PHASE 8).

---

## 2. Prerequisites

- Completion of **PHASE 1 (Project Setup)** and **PHASE 2 (Configuration)** so that [`src/config.py`](../src/config.py) provides `Settings` with a `jina_api_key` field.
- Completion of **PHASE 4 (Schemas)** so that the `src/schemas/embeddings/` package structure exists.
- A valid **Jina AI API key** (`JINA_API_KEY`). You can obtain one from the Jina AI platform.
- Python 3.11+ and the project's virtual environment active.

---

## 3. Dependencies to Install

Add the following to `pyproject.toml` (or install via `uv add` / `pip install`):

```bash
httpx
pydantic
```

> `httpx` is used for the async HTTP client. `pydantic` is already a base dependency from PHASE 1.

---

## 4. Directory Structure to Create

```
src/
├── schemas/
│   └── embeddings/
│       ├── __init__.py            # (empty package marker)
│       └── jina.py                # JinaEmbeddingRequest, JinaEmbeddingResponse
└── services/
    └── embeddings/
        ├── __init__.py            # (empty package marker)
        ├── jina_client.py         # JinaEmbeddingsClient
        └── factory.py             # make_embeddings_service, make_embeddings_client
```

---

## 5. Step-by-Step Implementation

### Step 1 — Define the Jina embedding schemas

**Full file path:** `src/schemas/embeddings/jina.py`

```python
from typing import Dict, List

from pydantic import BaseModel


class JinaEmbeddingRequest(BaseModel):
    """Request payload for the Jina embeddings API."""

    model: str = "jina-embeddings-v3"
    task: str = "retrieval.passage"
    dimensions: int = 1024
    late_chunking: bool = False
    embedding_type: str = "float"
    input: List[str]


class JinaEmbeddingResponse(BaseModel):
    """Response payload from the Jina embeddings API."""

    model: str
    object: str = "list"
    usage: Dict[str, int]
    data: List[Dict]
```

**Explanation:** `JinaEmbeddingRequest` models the request body sent to `POST /embeddings`. It defaults to the `jina-embeddings-v3` model, the `retrieval.passage` task, and 1024 dimensions. `JinaEmbeddingResponse` models the response, which contains the model name, a `data` list of embedding objects, and token `usage` counts.

---

### Step 2 — Add the embeddings schema package `__init__.py`

**Full file path:** `src/schemas/embeddings/__init__.py`

```python
```

**Explanation:** This file is intentionally empty in the reference project. It marks the directory as a Python package so the schemas can be imported.

---

### Step 3 — Implement the Jina embeddings client

**Full file path:** `src/services/embeddings/jina_client.py`

```python
import logging
from typing import List

import httpx

from src.schemas.embeddings.jina import JinaEmbeddingRequest, JinaEmbeddingResponse

logger = logging.getLogger(__name__)


class JinaEmbeddingsClient:
    """Client for Jina AI embeddings API.

    Generates embeddings for passages and queries using the jina-embeddings-v3 model.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.jina.ai/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(timeout=30.0)

    async def embed_passages(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Embed a list of passages.

        :param texts: List of passage texts to embed
        :param batch_size: Number of texts to embed per API call
        :returns: List of embedding vectors
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            try:
                request = JinaEmbeddingRequest(
                    model="jina-embeddings-v3",
                    task="retrieval.passage",
                    dimensions=1024,
                    input=batch,
                )

                response = await self._client.post(
                    f"{self.base_url}/embeddings",
                    headers=self.headers,
                    json=request.model_dump(),
                )
                response.raise_for_status()

                parsed = JinaEmbeddingResponse(**response.json())

                for item in parsed.data:
                    embeddings.append(item["embedding"])

                logger.info(f"Embedded batch of {len(batch)} passages")

            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                raise

        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Embed a search query.

        :param query: Query text
        :returns: Query embedding vector
        """
        try:
            request = JinaEmbeddingRequest(
                model="jina-embeddings-v3",
                task="retrieval.query",
                dimensions=1024,
                input=[query],
            )

            response = await self._client.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=request.model_dump(),
            )
            response.raise_for_status()

            parsed = JinaEmbeddingResponse(**response.json())

            return parsed.data[0]["embedding"]

        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
```

**Explanation:** The client wraps the Jina `/embeddings` endpoint. `embed_passages` splits the input into batches (default 100) and sends each batch with the `retrieval.passage` task. `embed_query` sends a single query with the `retrieval.query` task — this task distinction improves retrieval quality. Both use `jina-embeddings-v3` at 1024 dimensions. The client is an async context manager (`__aenter__`/`__aexit__`) and exposes `close()` to release the underlying `httpx.AsyncClient`.

---

### Step 4 — Add the embeddings factory

**Full file path:** `src/services/embeddings/factory.py`

```python
from typing import Optional

from src.config import Settings, get_settings

from .jina_client import JinaEmbeddingsClient


def make_embeddings_service(settings: Optional[Settings] = None) -> JinaEmbeddingsClient:
    """Create an embeddings service.

    Creates a new client instance each time to avoid closed client issues.

    :param settings: Optional settings instance. If None, loads from config.
    :returns: JinaEmbeddingsClient instance
    """
    if settings is None:
        settings = get_settings()

    api_key = settings.jina_api_key
    return JinaEmbeddingsClient(api_key=api_key)


def make_embeddings_client(settings: Optional[Settings] = None) -> JinaEmbeddingsClient:
    """Create an embeddings client.

    Creates a new client instance each time to avoid closed client issues.

    :param settings: Optional settings instance. If None, loads from config.
    :returns: JinaEmbeddingsClient instance
    """
    if settings is None:
        settings = get_settings()

    api_key = settings.jina_api_key
    return JinaEmbeddingsClient(api_key=api_key)
```

**Explanation:** Both factory functions create a **new** `JinaEmbeddingsClient` on every call. This is deliberate — because the client owns an `httpx.AsyncClient` that may be closed after use, caching a singleton would risk reusing a closed client. `make_embeddings_service` and `make_embeddings_client` are aliases with identical behavior, kept for API clarity.

---

### Step 5 — Add the embeddings service package `__init__.py`

**Full file path:** `src/services/embeddings/__init__.py`

```python
```

**Explanation:** This file is intentionally empty in the reference project. It marks the directory as a Python package.

---

## 6. Configuration

Add the following environment variable to your `.env` file (see PHASE 2 for the full config setup):

```bash
# Jina AI embeddings API key
JINA_API_KEY=jina_...
```

**Expected results:** The `Settings` object loads `jina_api_key` from `JINA_API_KEY`. The factory functions pass this key into the `JinaEmbeddingsClient`.

---

## 7. Verification

Run the following checks to confirm the phase is complete.

### 7.1 Import and schema check

```bash
python -c "
from src.schemas.embeddings.jina import JinaEmbeddingRequest, JinaEmbeddingResponse
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.embeddings.factory import make_embeddings_client

req = JinaEmbeddingRequest(input=['hello world'])
print('model:', req.model)
print('dimensions:', req.dimensions)
print('task:', req.task)
"
```

**Expected results:** The request defaults print `jina-embeddings-v3`, `1024`, and `retrieval.passage`.

### 7.2 Factory smoke test

```bash
python -c "
from src.services.embeddings.factory import make_embeddings_client
client = make_embeddings_client()
print('Client created:', type(client).__name__)
"
```

**Expected results:** A `JinaEmbeddingsClient` instance is created successfully.

### 7.3 Live embedding test (requires a valid `JINA_API_KEY`)

```bash
python -c "
import asyncio
from src.services.embeddings.factory import make_embeddings_client

async def main():
    client = make_embeddings_client()
    try:
        vec = await client.embed_query('What is attention in transformers?')
        print('Query embedding dim:', len(vec))
        passages = await client.embed_passages(['Paper about attention.', 'Paper about RAG.'])
        print('Passage embeddings:', len(passages), 'each dim', len(passages[0]))
    finally:
        await client.close()

asyncio.run(main())
"
```

**Expected results:** The query embedding and each passage embedding have length `1024`. Two passage embeddings are returned.

---

## 8. Common Pitfalls

- **Do not cache the client as a singleton.** Unlike the LLM factories, the embeddings factories intentionally create a fresh client each call. The client owns an `httpx.AsyncClient` that can be closed; reusing a closed client raises an error. Always call `await client.close()` (or use `async with`) when done.
- **Passage vs. query task matters.** Use `retrieval.passage` for documents and `retrieval.query` for search queries. Mixing them up degrades retrieval quality because the model is fine-tuned per task.
- **Batch size limits.** The default `batch_size=100` is safe, but very long passages may exceed the API token limit. If you hit `400`/`413` errors, reduce the batch size or truncate inputs.
- **Dimension mismatch with OpenSearch.** The index mapping in PHASE 7 declares `dimension: 1024`. If you change the embedding model or dimensions here, you must update the OpenSearch `knn_vector` mapping to match, or vector search will fail.
- **Missing `JINA_API_KEY`.** Without the key, the client sends an empty `Bearer` token and the API returns `401`. Ensure `JINA_API_KEY` is set in `.env`.
- **Network/timeout.** The client uses a 30-second timeout. Slow batches may time out; consider reducing `batch_size` for large documents.

---

## 9. Definition of Done

- [ ] [`src/schemas/embeddings/jina.py`](../src/schemas/embeddings/jina.py) defines `JinaEmbeddingRequest` and `JinaEmbeddingResponse`.
- [ ] [`src/services/embeddings/jina_client.py`](../src/services/embeddings/jina_client.py) implements `JinaEmbeddingsClient` with `embed_passages` (batched) and `embed_query`.
- [ ] The client uses `jina-embeddings-v3` at 1024 dimensions with `retrieval.passage` / `retrieval.query` tasks.
- [ ] [`src/services/embeddings/factory.py`](../src/services/embeddings/factory.py) provides `make_embeddings_service` and `make_embeddings_client`, each creating a fresh client.
- [ ] `JINA_API_KEY` is wired through `Settings` into the client.
- [ ] Live embedding verification returns 1024-dimensional vectors for both queries and passages.
