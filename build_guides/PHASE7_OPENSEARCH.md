# PHASE 7 — OpenSearch

> **Build guide for the from-scratch reconstruction of the arXiv Paper Curator (Agentic RAG).**
>
> This phase implements the OpenSearch integration layer. It provides a unified [`OpenSearchClient`](../src/services/opensearch/client.py) supporting **BM25 keyword search**, **pure vector (KNN) search**, and **hybrid search** via native Reciprocal Rank Fusion (RRF). It also includes the [`QueryBuilder`](../src/services/opensearch/query_builder.py), the hybrid index mapping [`ARXIV_PAPERS_CHUNKS_MAPPING`](../src/services/opensearch/index_config_hybrid.py), and the RRF pipeline configuration.

---

## 1. Phase Objective

By the end of this phase you will have:

- A hybrid OpenSearch index (`arxiv-papers-chunks`) with a `knn_vector` field (1024-dim, HNSW, cosine similarity) alongside BM25 text fields.
- An RRF search pipeline (`hybrid-rrf-pipeline`) for native hybrid search.
- A [`QueryBuilder`](../src/services/opensearch/query_builder.py) that constructs BM25 queries with field weights, fuzziness, filters, highlighting, and sorting.
- An [`OpenSearchClient`](../src/services/opensearch/client.py) exposing `health_check`, `setup_indices`, `search_papers`, `search_chunks_vector`, `search_unified`, `search_chunks_hybrid`, `index_chunk`, and `bulk_index_chunks`.
- Factory functions [`make_opensearch_client`](../src/services/opensearch/factory.py) and [`make_opensearch_client_fresh`](../src/services/opensearch/factory.py).

---

## 2. Prerequisites

- Completion of **PHASE 1 (Project Setup)** and **PHASE 2 (Configuration)** so that [`src/config.py`](../src/config.py) provides `OpenSearchSettings` (host, index name, chunk suffix, vector dimension, RRF pipeline name).
- Completion of **PHASE 6 (Embeddings)** so that 1024-dimensional query embeddings are available for vector/hybrid search.
- A running **OpenSearch** cluster (local via Docker or remote). The default host is `http://localhost:9200`.
- The OpenSearch **k-NN plugin** enabled on the cluster (required for `knn_vector` and HNSW).
- Python 3.11+ and the project's virtual environment active.

---

## 3. Dependencies to Install

Add the following to `pyproject.toml` (or install via `uv add` / `pip install`):

```bash
opensearch-py
```

> `opensearch-py` is the official OpenSearch Python client. It provides the `OpenSearch` class and the `helpers.bulk` utility used for bulk indexing.

---

## 4. Directory Structure to Create

```
src/services/opensearch/
├── __init__.py                    # Re-exports OpenSearchClient, factories, QueryBuilder
├── client.py                      # OpenSearchClient
├── factory.py                     # make_opensearch_client, make_opensearch_client_fresh
├── index_config_hybrid.py         # ARXIV_PAPERS_CHUNKS_MAPPING, HYBRID_RRF_PIPELINE
└── query_builder.py               # QueryBuilder
```

---

## 5. Step-by-Step Implementation

### Step 1 — Define the hybrid index configuration

**Full file path:** `src/services/opensearch/index_config_hybrid.py`

```python
"""OpenSearch index configuration for hybrid search (BM25 + Vector).

This configuration supports both keyword search (BM25) and vector similarity search
using HNSW algorithm for approximate nearest neighbor search.
"""

ARXIV_PAPERS_CHUNKS_INDEX = "arxiv-papers-chunks"

# Index mapping for chunked papers with vector embeddings
ARXIV_PAPERS_CHUNKS_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index.knn": True,
        "index.knn.space_type": "cosinesimil",
        "analysis": {
            "analyzer": {
                "standard_analyzer": {"type": "standard", "stopwords": "_english_"},
                "text_analyzer": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "stop", "snowball"]},
            }
        },
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "chunk_id": {"type": "keyword"},
            "arxiv_id": {"type": "keyword"},
            "paper_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "chunk_text": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "chunk_word_count": {"type": "integer"},
            "start_char": {"type": "integer"},
            "end_char": {"type": "integer"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,  # Jina v3 embeddings dimension
                "method": {
                    "name": "hnsw",  # Hierarchical Navigable Small World
                    "space_type": "cosinesimil",  # Cosine similarity
                    "engine": "nmslib",
                    "parameters": {
                        "ef_construction": 512,  # Higher value = better recall, slower indexing
                        "m": 16,  # Number of bi-directional links
                    },
                },
            },
            "title": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "authors": {
                "type": "text",
                "analyzer": "standard_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "abstract": {"type": "text", "analyzer": "text_analyzer"},
            "categories": {"type": "keyword"},
            "published_date": {"type": "date"},
            "section_title": {"type": "keyword"},
            "embedding_model": {"type": "keyword"},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
        },
    },
}

HYBRID_RRF_PIPELINE = {
    "id": "hybrid-rrf-pipeline",
    "description": "Post processor for hybrid RRF search",
    "phase_results_processors": [
        {
            "score-ranker-processor": {
                "combination": {
                    "technique": "rrf",  # Reciprocal Rank Fusion
                    "rank_constant": 60,  # Default k=60 for RRF formula: 1/(k+rank)
                }
            }
        }
    ],
}

# Alternative: Weighted average pipeline (commented out - not used by default)
# This could be used if you need explicit control over BM25 vs vector weights
# However, RRF generally provides better results without manual weight tuning
"""
HYBRID_SEARCH_PIPELINE = {
    "id": "hybrid-ranking-pipeline",
    "description": "Hybrid search pipeline using weighted average for BM25 and vector similarity",
    "phase_results_processors": [
        {
            "normalization-processor": {
                "normalization": {
                    "technique": "l2"  # L2 normalization for better score distribution
                },
                "combination": {
                    "technique": "harmonic_mean",  # Harmonic mean often works better than arithmetic
                    "parameters": {
                        "weights": [0.3, 0.7]  # 30% BM25, 70% vector similarity
                    }
                }
            }
        }
    ]
}
"""
```

**Explanation:** The mapping enables the k-NN plugin (`index.knn: True`) with cosine similarity. The `embedding` field is a `knn_vector` of dimension 1024 using the HNSW algorithm (nmslib engine, `ef_construction=512`, `m=16`). Text fields use custom analyzers (`text_analyzer` with lowercase/stop/snowball filters). `dynamic: "strict"` means unknown fields are rejected. The `HYBRID_RRF_PIPELINE` defines a search pipeline that fuses BM25 and vector results using Reciprocal Rank Fusion with `rank_constant=60`. A commented-out weighted-average alternative is included for reference.

---

### Step 2 — Implement the query builder

**Full file path:** `src/services/opensearch/query_builder.py`

```python
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QueryBuilder:
    """Builds OpenSearch search queries for papers and chunks."""

    def __init__(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        fields: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        track_total_hits: bool = True,
        latest_papers: bool = False,
        search_chunks: bool = False,
    ):
        self.query = query
        self.size = size
        self.from_ = from_
        self.fields = fields
        self.categories = categories
        self.track_total_hits = track_total_hits
        self.latest_papers = latest_papers
        self.search_chunks = search_chunks

        # Set default fields based on search mode
        if self.fields is None:
            if self.search_chunks:
                self.fields = ["chunk_text^3", "title^2", "abstract^1"]
            else:
                self.fields = ["title^3", "abstract^2", "authors^1"]

    def build(self) -> Dict[str, Any]:
        """Build the complete search query body."""
        body = {
            "query": self._build_query(),
            "size": self.size,
            "from": self.from_,
            "track_total_hits": self.track_total_hits,
            "_source": self._build_source_fields(),
            "highlight": self._build_highlight(),
        }

        sort = self._build_sort()
        if sort:
            body["sort"] = sort

        return body

    def _build_query(self) -> Dict[str, Any]:
        """Build the query clause."""
        query = self._build_text_query()

        filters = self._build_filters()
        if filters:
            query = {"bool": {"must": [query], "filter": filters}}

        return query

    def _build_text_query(self) -> Dict[str, Any]:
        """Build the multi_match text query."""
        return {
            "multi_match": {
                "query": self.query,
                "fields": self.fields,
                "type": "best_fields",
                "operator": "or",
                "fuzziness": "AUTO",
                "prefix_length": 2,
            }
        }

    def _build_filters(self) -> List[Dict[str, Any]]:
        """Build filter clauses."""
        filters = []
        if self.categories:
            filters.append({"terms": {"categories": self.categories}})
        return filters

    def _build_source_fields(self) -> Any:
        """Build the _source field selection."""
        if self.search_chunks:
            return {"excludes": ["embedding"]}
        return [
            "title",
            "authors",
            "abstract",
            "categories",
            "published_date",
            "arxiv_id",
            "paper_id",
        ]

    def _build_highlight(self) -> Dict[str, Any]:
        """Build the highlight configuration."""
        if self.search_chunks:
            return {
                "fields": {
                    "chunk_text": {"fragment_size": 150, "number_of_fragments": 2},
                    "title": {"fragment_size": 100, "number_of_fragments": 1},
                }
            }
        return {
            "fields": {
                "title": {"fragment_size": 100, "number_of_fragments": 1},
                "abstract": {"fragment_size": 200, "number_of_fragments": 2},
            }
        }

    def _build_sort(self) -> Optional[List[Dict[str, Any]]]:
        """Build the sort clause."""
        if self.latest_papers:
            return [{"published_date": {"order": "desc"}}, {"_score": {"order": "desc"}}]
        return None
```

**Explanation:** The `QueryBuilder` constructs a BM25 `multi_match` query. Default field weights differ by mode: chunk search boosts `chunk_text^3` over `title^2`/`abstract^1`; paper search boosts `title^3` over `abstract^2`/`authors^1`. It uses `fuzziness=AUTO` with `prefix_length=2` for typo tolerance, optional category filters, `_source` exclusion of the heavy `embedding` field in chunk mode, per-field highlighting, and date-based sorting when `latest_papers` is set.

---

### Step 3 — Implement the OpenSearch client

**Full file path:** `src/services/opensearch/client.py`

```python
"""Unified OpenSearch client supporting both simple BM25 and hybrid search."""

import logging
from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch
from src.config import Settings

from .index_config_hybrid import ARXIV_PAPERS_CHUNKS_MAPPING, HYBRID_RRF_PIPELINE
from .query_builder import QueryBuilder

logger = logging.getLogger(__name__)


class OpenSearchClient:
    """OpenSearch client supporting BM25 and hybrid search with native RRF."""

    def __init__(self, host: str, settings: Settings):
        self.host = host
        self.settings = settings
        self.index_name = f"{settings.opensearch.index_name}-{settings.opensearch.chunk_index_suffix}"

        self.client = OpenSearch(
            hosts=[host],
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False,
        )

        logger.info(f"OpenSearch client initialized with host: {host}")

    def health_check(self) -> bool:
        """Check if OpenSearch cluster is healthy."""
        try:
            health = self.client.cluster.health()
            return health["status"] in ["green", "yellow"]
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics for the hybrid index."""
        try:
            if not self.client.indices.exists(index=self.index_name):
                return {"index_name": self.index_name, "exists": False, "document_count": 0}

            stats_response = self.client.indices.stats(index=self.index_name)
            index_stats = stats_response["indices"][self.index_name]["total"]

            return {
                "index_name": self.index_name,
                "exists": True,
                "document_count": index_stats["docs"]["count"],
                "deleted_count": index_stats["docs"]["deleted"],
                "size_in_bytes": index_stats["store"]["size_in_bytes"],
            }

        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"index_name": self.index_name, "exists": False, "document_count": 0, "error": str(e)}

    def setup_indices(self, force: bool = False) -> Dict[str, bool]:
        """Setup the hybrid search index and RRF pipeline."""
        results = {}
        results["hybrid_index"] = self._create_hybrid_index(force)
        results["rrf_pipeline"] = self._create_rrf_pipeline(force)
        return results

    def _create_hybrid_index(self, force: bool = False) -> bool:
        """Create hybrid index for all search types (BM25, vector, hybrid).

        :param force: If True, recreate index even if it exists
        :returns: True if created, False if already exists
        """
        try:
            if force and self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                logger.info(f"Deleted existing hybrid index: {self.index_name}")

            if not self.client.indices.exists(index=self.index_name):
                self.client.indices.create(index=self.index_name, body=ARXIV_PAPERS_CHUNKS_MAPPING)
                logger.info(f"Created hybrid index: {self.index_name}")
                return True

            logger.info(f"Hybrid index already exists: {self.index_name}")
            return False

        except Exception as e:
            # Handle race condition when multiple workers start simultaneously:
            # all check exists() -> False, all try to create, only one succeeds.
            if "resource_already_exists_exception" in str(e):
                logger.info(f"Hybrid index already exists (created by another worker): {self.index_name}")
                return False
            logger.error(f"Error creating hybrid index: {e}")
            raise

    def _create_rrf_pipeline(self, force: bool = False) -> bool:
        """Create RRF search pipeline for native hybrid search.

        :param force: If True, recreate pipeline even if it exists
        :returns: True if created, False if already exists
        """
        try:
            pipeline_id = HYBRID_RRF_PIPELINE["id"]

            if force:
                try:
                    self.client.ingest.get_pipeline(id=pipeline_id)
                    self.client.ingest.delete_pipeline(id=pipeline_id)
                    logger.info(f"Deleted existing RRF pipeline: {pipeline_id}")
                except Exception:
                    pass

            try:
                self.client.ingest.get_pipeline(id=pipeline_id)
                logger.info(f"RRF pipeline already exists: {pipeline_id}")
                return False
            except Exception:
                pass
            pipeline_body = {
                "description": HYBRID_RRF_PIPELINE["description"],
                "phase_results_processors": HYBRID_RRF_PIPELINE["phase_results_processors"],
            }

            self.client.transport.perform_request("PUT", f"/_search/pipeline/{pipeline_id}", body=pipeline_body)

            logger.info(f"Created RRF search pipeline: {pipeline_id}")
            return True

        except Exception as e:
            logger.error(f"Error creating RRF pipeline: {e}")
            raise

    def search_papers(
        self, query: str, size: int = 10, from_: int = 0, categories: Optional[List[str]] = None, latest: bool = True
    ) -> Dict[str, Any]:
        """BM25 search for papers."""
        return self._search_bm25_only(query=query, size=size, from_=from_, categories=categories, latest=latest)

    def search_chunks_vector(
        self, query_embedding: List[float], size: int = 10, categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Pure vector search on chunks.

        :param query_embedding: Query embedding vector
        :param size: Number of results
        :param categories: Optional category filter
        :returns: Search results
        """
        try:
            # Build filter
            filter_clause = []
            if categories:
                filter_clause.append({"terms": {"categories": categories}})

            search_body = {
                "size": size,
                "query": {"knn": {"embedding": {"vector": query_embedding, "k": size}}},
                "_source": {"excludes": ["embedding"]},
            }

            if filter_clause:
                search_body["query"] = {"bool": {"must": [search_body["query"]], "filter": filter_clause}}

            response = self.client.search(index=self.index_name, body=search_body)

            results = {"total": response["hits"]["total"]["value"], "hits": []}

            for hit in response["hits"]["hits"]:
                chunk = hit["_source"]
                chunk["score"] = hit["_score"]
                chunk["chunk_id"] = hit["_id"]
                results["hits"].append(chunk)

            return results

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return {"total": 0, "hits": []}

    def search_unified(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        size: int = 10,
        from_: int = 0,
        categories: Optional[List[str]] = None,
        latest: bool = False,
        use_hybrid: bool = True,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """Unified search method supporting BM25, vector, and hybrid modes.

        :param query: Text query for search
        :param query_embedding: Optional embedding for vector/hybrid search
        :param size: Number of results to return
        :param from_: Offset for pagination
        :param categories: Optional category filter
        :param latest: Sort by date instead of relevance
        :param use_hybrid: If True and embedding provided, use hybrid search
        :param min_score: Minimum score threshold
        :returns: Search results
        """
        try:
            # If no embedding provided or hybrid disabled, use BM25 only
            if not query_embedding or not use_hybrid:
                return self._search_bm25_only(query=query, size=size, from_=from_, categories=categories, latest=latest)

            # Use native OpenSearch hybrid search with RRF pipeline
            return self._search_hybrid_native(
                query=query, query_embedding=query_embedding, size=size, categories=categories, min_score=min_score
            )

        except Exception as e:
            logger.error(f"Unified search error: {e}")
            return {"total": 0, "hits": []}

    def _search_bm25_only(
        self, query: str, size: int, from_: int, categories: Optional[List[str]], latest: bool
    ) -> Dict[str, Any]:
        """Pure BM25 search implementation."""
        builder = QueryBuilder(
            query=query,
            size=size,
            from_=from_,
            categories=categories,
            latest_papers=latest,
            search_chunks=True,  # Enable chunk search mode
        )
        search_body = builder.build()

        response = self.client.search(index=self.index_name, body=search_body)

        results = {"total": response["hits"]["total"]["value"], "hits": []}

        for hit in response["hits"]["hits"]:
            chunk = hit["_source"]
            chunk["score"] = hit["_score"]
            chunk["chunk_id"] = hit["_id"]

            if "highlight" in hit:
                chunk["highlights"] = hit["highlight"]

            results["hits"].append(chunk)

        logger.info(f"BM25 search for '{query[:50]}...' returned {results['total']} results")
        return results

    def _search_hybrid_native(
        self, query: str, query_embedding: List[float], size: int, categories: Optional[List[str]], min_score: float
    ) -> Dict[str, Any]:
        """Native OpenSearch hybrid search with RRF pipeline."""
        builder = QueryBuilder(
            query=query, size=size * 2, from_=0, categories=categories, latest_papers=False, search_chunks=True
        )
        bm25_search_body = builder.build()

        bm25_query = bm25_search_body["query"]

        hybrid_query = {"hybrid": {"queries": [bm25_query, {"knn": {"embedding": {"vector": query_embedding, "k": size * 2}}}]}}

        search_body = {
            "size": size,
            "query": hybrid_query,
            "_source": bm25_search_body["_source"],
            "highlight": bm25_search_body["highlight"],
        }

        # Execute search with RRF pipeline
        response = self.client.search(
            index=self.index_name, body=search_body, params={"search_pipeline": HYBRID_RRF_PIPELINE["id"]}
        )

        results = {"total": response["hits"]["total"]["value"], "hits": []}

        for hit in response["hits"]["hits"]:
            if hit["_score"] < min_score:
                continue

            chunk = hit["_source"]
            chunk["score"] = hit["_score"]
            chunk["chunk_id"] = hit["_id"]

            if "highlight" in hit:
                chunk["highlights"] = hit["highlight"]

            results["hits"].append(chunk)

        results["total"] = len(results["hits"])
        logger.info(f"Native hybrid search for '{query[:50]}...' returned {results['total']} results")
        return results

    def search_chunks_hybrid(
        self,
        query: str,
        query_embedding: List[float],
        size: int = 10,
        categories: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """Hybrid search combining BM25 and vector similarity using native RRF."""
        return self._search_hybrid_native(
            query=query, query_embedding=query_embedding, size=size, categories=categories, min_score=min_score
        )

    def index_chunk(self, chunk_data: Dict[str, Any], embedding: List[float]) -> bool:
        """Index a single chunk with its embedding.

        :param chunk_data: Chunk data dictionary
        :param embedding: Embedding vector
        :returns: True if successful
        """
        try:
            chunk_data["embedding"] = embedding

            response = self.client.index(index=self.index_name, body=chunk_data, refresh=True)

            return response["result"] in ["created", "updated"]

        except Exception as e:
            logger.error(f"Error indexing chunk: {e}")
            return False

    def bulk_index_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Bulk index multiple chunks with embeddings.

        :param chunks: List of dicts with 'chunk_data' and 'embedding'
        :returns: Statistics
        """
        from opensearchpy import helpers

        try:
            actions = []
            for chunk in chunks:
                chunk_data = chunk["chunk_data"].copy()
                chunk_data["embedding"] = chunk["embedding"]

                action = {"_index": self.index_name, "_source": chunk_data}
                actions.append(action)

            success, failed = helpers.bulk(self.client, actions, refresh=True)

            logger.info(f"Bulk indexed {success} chunks, {len(failed)} failed")
            return {"success": success, "failed": len(failed)}

        except Exception as e:
            logger.error(f"Bulk chunk indexing error: {e}")
            raise

    def delete_paper_chunks(self, arxiv_id: str) -> bool:
        """Delete all chunks for a specific paper.

        :param arxiv_id: ArXiv ID of the paper
        :returns: True if deletion was successful
        """
        try:
            response = self.client.delete_by_query(
                index=self.index_name, body={"query": {"term": {"arxiv_id": arxiv_id}}}, refresh=True
            )

            deleted = response.get("deleted", 0)
            logger.info(f"Deleted {deleted} chunks for paper {arxiv_id}")
            return deleted > 0

        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return False

    def get_chunks_by_paper(self, arxiv_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific paper.

        :param arxiv_id: ArXiv ID of the paper
        :returns: List of chunks sorted by chunk_index
        """
        try:
            search_body = {
                "query": {"term": {"arxiv_id": arxiv_id}},
                "size": 1000,
                "sort": [{"chunk_index": "asc"}],
                "_source": {"excludes": ["embedding"]},
            }

            response = self.client.search(index=self.index_name, body=search_body)

            chunks = []
            for hit in response["hits"]["hits"]:
                chunk = hit["_source"]
                chunk["chunk_id"] = hit["_id"]
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Error getting chunks: {e}")
            return []
```

**Explanation:** The client wraps the `opensearchpy.OpenSearch` connection. The index name is derived from settings as `{index_name}-{chunk_index_suffix}` (e.g. `arxiv-papers-chunks`). Key methods:
- `setup_indices` creates the hybrid index and RRF pipeline (idempotent, with `force` to recreate).
- `search_papers` / `_search_bm25_only` run pure BM25 chunk search via `QueryBuilder`.
- `search_chunks_vector` runs pure KNN vector search.
- `search_unified` dispatches to BM25 or native hybrid based on whether an embedding is provided.
- `_search_hybrid_native` builds a `hybrid` query combining BM25 + KNN and executes it with the RRF search pipeline.
- `index_chunk` / `bulk_index_chunks` write chunks (with embeddings) into the index.
- `delete_paper_chunks` / `get_chunks_by_paper` support reindexing and retrieval.

---

### Step 4 — Add the OpenSearch factory

**Full file path:** `src/services/opensearch/factory.py`

```python
"""Unified factory for OpenSearch client."""

from functools import lru_cache
from typing import Optional

from src.config import Settings, get_settings

from .client import OpenSearchClient


@lru_cache(maxsize=1)
def make_opensearch_client(settings: Optional[Settings] = None) -> OpenSearchClient:
    """Create a cached OpenSearch client singleton.

    :param settings: Optional settings instance. If None, loads from config.
    :returns: OpenSearchClient instance
    """
    if settings is None:
        settings = get_settings()
    return OpenSearchClient(host=settings.opensearch.host, settings=settings)


def make_opensearch_client_fresh(settings: Optional[Settings] = None, host: Optional[str] = None) -> OpenSearchClient:
    """Create a fresh OpenSearch client (not cached).

    :param settings: Optional settings instance. If None, loads from config.
    :param host: Optional host override. If None, uses settings.opensearch.host.
    :returns: OpenSearchClient instance
    """
    if settings is None:
        settings = get_settings()
    return OpenSearchClient(host=host or settings.opensearch.host, settings=settings)
```

**Explanation:** `make_opensearch_client` is cached with `@lru_cache(maxsize=1)` for a shared singleton (used by the API for search). `make_opensearch_client_fresh` creates a new instance each call and accepts an optional host override — this is used by the indexing service (PHASE 8) so it can point at a different host without disturbing the cached search client.

---

### Step 5 — Add the OpenSearch package `__init__.py`

**Full file path:** `src/services/opensearch/__init__.py`

```python
from .client import OpenSearchClient
from .factory import make_opensearch_client, make_opensearch_client_fresh
from .query_builder import QueryBuilder

__all__ = ["OpenSearchClient", "make_opensearch_client", "make_opensearch_client_fresh", "QueryBuilder"]
```

**Explanation:** Re-exports the client, factories, and query builder for convenient imports.

---

## 6. Configuration

Add the following environment variables to your `.env` file (see PHASE 2 for the full config setup):

```bash
# OpenSearch connection
OPENSEARCH__HOST=http://localhost:9200
OPENSEARCH__INDEX_NAME=arxiv-papers
OPENSEARCH__CHUNK_INDEX_SUFFIX=chunks

# Vector search settings (must match the embedding dimension from PHASE 6)
OPENSEARCH__VECTOR_DIMENSION=1024
OPENSEARCH__VECTOR_SPACE_TYPE=cosinesimil

# Hybrid search settings
OPENSEARCH__RRF_PIPELINE_NAME=hybrid-rrf-pipeline
OPENSEARCH__HYBRID_SEARCH_SIZE_MULTIPLIER=2
```

**Expected results:** The `Settings` object loads these values. The client builds the index name `arxiv-papers-chunks` and connects to `http://localhost:9200`.

---

## 7. Verification

Run the following checks to confirm the phase is complete. These require a running OpenSearch cluster with the k-NN plugin.

### 7.1 Health check

```bash
python -c "
from src.services.opensearch.factory import make_opensearch_client
client = make_opensearch_client()
print('Healthy:', client.health_check())
"
```

**Expected results:** `Healthy: True` when the cluster is up (status `green` or `yellow`).

### 7.2 Setup indices and pipeline

```bash
python -c "
from src.services.opensearch.factory import make_opensearch_client
client = make_opensearch_client()
print(client.setup_indices())
"
```

**Expected results:** Prints `{'hybrid_index': True, 'rrf_pipeline': True}` on first run (created), and `False`/`False` on subsequent runs (already exist). Re-running with `setup_indices(force=True)` recreates both.

### 7.3 Index stats

```bash
python -c "
from src.services.opensearch.factory import make_opensearch_client
client = make_opensearch_client()
print(client.get_index_stats())
"
```

**Expected results:** Shows `exists: True` and a `document_count` (0 before any indexing).

### 7.4 BM25 search (after indexing in PHASE 8)

```bash
python -c "
from src.services.opensearch.factory import make_opensearch_client
client = make_opensearch_client()
results = client.search_papers('attention mechanism', size=5)
print('Total:', results['total'])
for hit in results['hits'][:3]:
    print(hit['arxiv_id'], hit['score'])
"
```

**Expected results:** Returns matching chunks with scores. (Requires data indexed in PHASE 8.)

### 7.5 Vector search (requires a query embedding from PHASE 6)

```bash
python -c "
import asyncio
from src.services.opensearch.factory import make_opensearch_client
from src.services.embeddings.factory import make_embeddings_client

async def main():
    emb = make_embeddings_client()
    vec = await emb.embed_query('transformer attention')
    await emb.close()

    client = make_opensearch_client()
    results = client.search_chunks_vector(vec, size=5)
    print('Total:', results['total'])

asyncio.run(main())
"
```

**Expected results:** Returns chunks ranked by cosine similarity to the query embedding.

---

## 8. Common Pitfalls

- **k-NN plugin required.** The `knn_vector` field and HNSW method require the OpenSearch k-NN plugin. Without it, index creation fails with a mapping error. Enable the plugin on the cluster (or use the official Docker image with the plugin bundled).
- **Dimension mismatch.** The `embedding` field is declared as `dimension: 1024`. If the embedding model in PHASE 6 changes dimensions, you must recreate the index with the matching dimension — you cannot change the mapping of an existing index.
- **`dynamic: "strict"` rejects unknown fields.** Any field not in the mapping causes a `strict_dynamic_mapping_exception`. Ensure the chunk documents produced in PHASE 8 only contain mapped fields.
- **Index name derivation.** The client uses `{index_name}-{chunk_index_suffix}`. If you change `OPENSEARCH__INDEX_NAME` or `OPENSEARCH__CHUNK_INDEX_SUFFIX`, the client points at a different index — recreate it with `setup_indices`.
- **Race condition on index creation.** Multiple workers may try to create the index simultaneously. The client handles the `resource_already_exists_exception` gracefully (returns `False`).
- **RRF pipeline is a search pipeline, not an ingest pipeline.** It is created via `PUT /_search/pipeline/{id}` (using `transport.perform_request`), not the standard ingest pipeline API. The `ingest.get_pipeline` / `ingest.delete_pipeline` calls are only used to check/delete existence.
- **Hybrid search needs an embedding.** `search_unified` and `search_chunks_hybrid` fall back to BM25-only when no `query_embedding` is provided. To exercise hybrid search, always pass a query embedding from PHASE 6.
- **`min_score` filtering.** In hybrid search, results below `min_score` are dropped and `total` is recomputed as the length of the surviving hits. A high `min_score` may return fewer results than requested.

---

## 9. Definition of Done

- [ ] [`src/services/opensearch/index_config_hybrid.py`](../src/services/opensearch/index_config_hybrid.py) defines `ARXIV_PAPERS_CHUNKS_MAPPING` (with `knn_vector` 1024-dim HNSW, cosine similarity) and `HYBRID_RRF_PIPELINE` (`rank_constant=60`).
- [ ] [`src/services/opensearch/query_builder.py`](../src/services/opensearch/query_builder.py) implements `QueryBuilder` with mode-aware field weights, `fuzziness=AUTO`, filters, highlighting, and sorting.
- [ ] [`src/services/opensearch/client.py`](../src/services/opensearch/client.py) implements `OpenSearchClient` with `health_check`, `setup_indices`, `search_papers`, `search_chunks_vector`, `search_unified`, `search_chunks_hybrid`, `index_chunk`, and `bulk_index_chunks`.
- [ ] [`src/services/opensearch/factory.py`](../src/services/opensearch/factory.py) provides cached `make_opensearch_client` and fresh `make_opensearch_client_fresh`.
- [ ] [`src/services/opensearch/__init__.py`](../src/services/opensearch/__init__.py) re-exports the client, factories, and query builder.
- [ ] `setup_indices()` creates the hybrid index and RRF pipeline successfully against a live cluster.
- [ ] BM25, vector, and hybrid searches return results against indexed data.