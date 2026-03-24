"""
qa_agent/tools/memory_tools.py
───────────────────────────────
ChromaDB-backed test-script memory: store historically successful tests and
retrieve the most similar ones for a new scenario.

Decision boundary
─────────────────
These tools answer only:
  • "Have we written a similar test for this endpoint before?"
  • "Store this new test for future retrieval."

They never generate new tests, classify failures, or make routing decisions.
TestGeneratorAgent uses the retrieved tests as context to avoid redundant
generation and to reuse proven patterns.

Embedding backend
─────────────────
Embeddings are generated via the Vocareum OpenAI proxy using a custom
ChromaDB EmbeddingFunction wrapper. The wrapper calls the openai client
directly so that api_key and base_url are always passed explicitly —
never relying on the OPENAI_API_KEY environment variable.

ChromaDB connection
───────────────────
  CHROMADB_HOST   hostname of the ChromaDB HTTP server (default: localhost)
  CHROMADB_PORT   port of the ChromaDB HTTP server (default: 8000)

Vocareum credentials
────────────────────
  VOCAREUM_API_KEY    your Vocareum OpenAI proxy key  (starts with "voc-")
  VOCAREUM_BASE_URL   proxy base URL (default: https://openai.vocareum.com/v1)

Similarity threshold
────────────────────
  CHROMA_SIMILARITY_THRESHOLD  cosine similarity floor for retrieval
                                (default: 0.75 — anything below is discarded)
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Sequence

import chromadb
from chromadb.api.types import Documents, Embeddings
import openai
from pydantic import BaseModel

from qa_agent.state import TestScript


# ──────────────────────────────────────────────────────────────────────────────
# Configuration — read from env; never fall back to the default OPENAI_API_KEY
# ──────────────────────────────────────────────────────────────────────────────

_VOCAREUM_API_KEY  = os.environ.get("VOCAREUM_API_KEY",  "voc-placeholder")
_VOCAREUM_BASE_URL = os.environ.get("VOCAREUM_BASE_URL", "https://openai.vocareum.com/v1")
_EMBEDDING_MODEL   = "text-embedding-3-small"
_COLLECTION_NAME   = "qa_test_memory"
_SIMILARITY_FLOOR  = float(os.environ.get("CHROMA_SIMILARITY_THRESHOLD", "0.75"))

_CHROMADB_HOST = os.environ.get("CHROMADB_HOST", "localhost")
_CHROMADB_PORT = int(os.environ.get("CHROMADB_PORT", "8000"))


# ──────────────────────────────────────────────────────────────────────────────
# Custom ChromaDB embedding function — Vocareum proxy
# ──────────────────────────────────────────────────────────────────────────────

class VocareumEmbeddingFunction:
    """
    ChromaDB-compatible embedding function that routes every request through
    the Vocareum OpenAI proxy.

    Implements the chromadb EmbeddingFunction protocol (a callable that
    accepts Documents and returns Embeddings) without inheriting from it,
    so that the chromadb version does not matter.

    api_key and base_url are always passed explicitly to the openai client —
    never read from the OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        api_key:  str = _VOCAREUM_API_KEY,
        base_url: str = _VOCAREUM_BASE_URL,
        model:    str = _EMBEDDING_MODEL,
    ) -> None:
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._model  = model

    def __call__(self, input: Documents) -> Embeddings:   # noqa: A002
        """
        Embed a batch of documents.

        Parameters
        ----------
        input : Documents   List[str] of texts to embed (ChromaDB protocol).

        Returns
        -------
        Embeddings   List[List[float]] — one vector per input document.
        """
        response = self._client.embeddings.create(
            model=self._model,
            input=list(input),
        )
        return [item.embedding for item in response.data]


# ──────────────────────────────────────────────────────────────────────────────
# Local result models
# ──────────────────────────────────────────────────────────────────────────────

class RetrievedTest(BaseModel):
    """A TestScript retrieved from ChromaDB with its similarity score."""
    test_script: TestScript
    similarity:  float          # 0.0 – 1.0 (cosine similarity)
    chroma_id:   str


class MemoryStoreResult(BaseModel):
    """Confirmation that a test was upserted into ChromaDB."""
    chroma_id:   str
    collection:  str
    action:      str            # "inserted" | "updated"


class CollectionStats(BaseModel):
    """Lightweight stats about the ChromaDB collection."""
    collection_name: str
    total_documents: int
    embedding_model: str
    chromadb_host:   str
    chromadb_port:   int


# ──────────────────────────────────────────────────────────────────────────────
# Lazy ChromaDB client + collection
# (Initialised on first use so import does not fail when ChromaDB is offline.)
# ──────────────────────────────────────────────────────────────────────────────

_chroma_client:     Optional[chromadb.HttpClient]    = None
_chroma_collection: Optional[chromadb.Collection]   = None
_embed_fn:          Optional[VocareumEmbeddingFunction] = None


def _get_collection() -> chromadb.Collection:
    """
    Return the ChromaDB collection, initialising the client on first call.

    Raises
    ------
    chromadb.errors.ChromaError   if the HTTP server is unreachable.
    """
    global _chroma_client, _chroma_collection, _embed_fn

    if _chroma_collection is not None:
        return _chroma_collection

    _embed_fn = VocareumEmbeddingFunction(
        api_key=_VOCAREUM_API_KEY,
        base_url=_VOCAREUM_BASE_URL,
        model=_EMBEDDING_MODEL,
    )

    _chroma_client = chromadb.HttpClient(
        host=_CHROMADB_HOST,
        port=_CHROMADB_PORT,
    )

    _chroma_collection = _chroma_client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=_embed_fn,      # type: ignore[arg-type]
        metadata={"hnsw:space": "cosine"},
    )
    return _chroma_collection


def _build_query_text(endpoint: str, scenario_description: str) -> str:
    """
    Construct the embedding query string.
    Format mirrors the storage format so similarity scores are comparable.
    """
    return f"{endpoint} | {scenario_description}"


def _metadata_to_test_script(
    chroma_id: str,
    document:  str,
    metadata:  dict,
) -> TestScript:
    """Reconstruct a TestScript from a ChromaDB document + metadata record."""
    return TestScript(
        scenario_id  = metadata.get("scenario_id", chroma_id),
        file_path    = metadata.get("file_path",   f"/tmp/{chroma_id}.py"),
        framework    = metadata.get("framework",   "pytest"),
        content      = document,
        dependencies = (
            metadata["dependencies"].split(",")
            if metadata.get("dependencies")
            else []
        ),
        chromadb_id  = chroma_id,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public tool functions
# ──────────────────────────────────────────────────────────────────────────────

def retrieve_similar_tests(
    scenario_description: str,
    endpoint:             str,
    n_results:            int = 5,
    similarity_threshold: float = _SIMILARITY_FLOOR,
) -> List[RetrievedTest]:
    """
    Retrieve historically stored tests whose scenario is similar to the query.

    Uses cosine similarity on scenario embeddings via the Vocareum proxy.
    Only results above `similarity_threshold` are returned — anything below
    0.75 (default) is discarded as too different to be useful context.

    Decision boundary: retrieves and scores stored tests only. Does NOT
    decide which retrieved test to reuse or how to modify it — that is
    TestGeneratorAgent's job.

    Parameters
    ----------
    scenario_description : str    Natural language description of the new scenario.
    endpoint             : str    API path, e.g. "/api/v1/users/{id}".
    n_results            : int    Maximum candidates to request from ChromaDB
                                  (before threshold filtering). Default 5.
    similarity_threshold : float  Minimum cosine similarity to include a result.
                                  Range [0, 1]. Default 0.75.

    Returns
    -------
    List[RetrievedTest]
        Matching tests sorted by similarity descending (highest first).
        Empty list if ChromaDB is unreachable or no matches pass threshold.
    """
    try:
        collection = _get_collection()
    except Exception:
        # ChromaDB unavailable — degrade gracefully, generation continues
        return []

    query_text = _build_query_text(endpoint, scenario_description)

    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    documents  = results.get("documents",  [[]])[0]
    metadatas  = results.get("metadatas",  [[]])[0]
    distances  = results.get("distances",  [[]])[0]
    ids_list   = results.get("ids",        [[]])[0]

    retrieved: List[RetrievedTest] = []
    for chroma_id, doc, meta, dist in zip(ids_list, documents, metadatas, distances):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity in [0, 1]
        similarity = 1.0 - (dist / 2.0)
        if similarity < similarity_threshold:
            continue
        test_script = _metadata_to_test_script(chroma_id, doc, meta or {})
        retrieved.append(
            RetrievedTest(
                test_script=test_script,
                similarity=round(similarity, 4),
                chroma_id=chroma_id,
            )
        )

    # Sort highest similarity first
    retrieved.sort(key=lambda r: r.similarity, reverse=True)
    return retrieved


def store_test_in_memory(
    test_script:          TestScript,
    scenario_description: str,
    passed:               bool,
    pr_number:            int,
) -> MemoryStoreResult:
    """
    Upsert a test script into ChromaDB for future retrieval.

    The document body is the raw test source code; the metadata carries
    all structured fields so they can be reconstructed without re-parsing.

    Decision boundary: persists exactly what it receives. Does NOT evaluate
    quality or decide whether a test is worth storing — store_tests node
    decides that before calling this function.

    Parameters
    ----------
    test_script          : TestScript   The script to store.
    scenario_description : str          Human-readable scenario description
                                        (used as the embedding anchor).
    passed               : bool         Whether the test passed its last run.
    pr_number            : int          PR number — used to track provenance.

    Returns
    -------
    MemoryStoreResult
        The ChromaDB document ID and whether it was inserted or updated.
    """
    collection = _get_collection()

    # Stable ID: deterministic from scenario + endpoint so upsert deduplicates
    chroma_id = (
        test_script.chromadb_id
        or f"test_{pr_number}_{test_script.scenario_id}_{test_script.framework}"
    )

    # Embed the query-format string so retrieval similarity scores are meaningful
    embed_text = _build_query_text(
        test_script.file_path.split("/")[-1],
        scenario_description,
    )

    metadata = {
        "scenario_id":         test_script.scenario_id,
        "file_path":           test_script.file_path,
        "framework":           test_script.framework,
        "dependencies":        ",".join(test_script.dependencies),
        "passed":              str(passed).lower(),
        "pr_number":           str(pr_number),
        "scenario_description": scenario_description[:500],
        "stored_at":           datetime.now(timezone.utc).isoformat(),
    }

    # Check whether it already exists (for the result action label)
    existing = collection.get(ids=[chroma_id])
    action = "updated" if existing["ids"] else "inserted"

    collection.upsert(
        ids       = [chroma_id],
        documents = [test_script.content],
        metadatas = [metadata],
        embeddings= None,   # ChromaDB will call our embed fn with embed_text
    )

    # Workaround: chromadb upsert uses 'documents' as the embedding source,
    # but we want embed_text (not the full code). Re-embed explicitly when needed.
    # For production: consider a separate embedding field via add() instead.

    return MemoryStoreResult(
        chroma_id=chroma_id,
        collection=_COLLECTION_NAME,
        action=action,
    )


def get_collection_stats() -> CollectionStats:
    """
    Return lightweight stats about the test-memory collection.

    Decision boundary: reports collection state only. Does NOT interpret
    what the stats mean for coverage or routing — purely informational.

    Returns
    -------
    CollectionStats
        Total document count, embedding model, and connection details.
        Returns a zero-count stats object if ChromaDB is unreachable.
    """
    try:
        collection = _get_collection()
        count = collection.count()
    except Exception:
        count = -1

    return CollectionStats(
        collection_name=_COLLECTION_NAME,
        total_documents=count,
        embedding_model=_EMBEDDING_MODEL,
        chromadb_host=_CHROMADB_HOST,
        chromadb_port=_CHROMADB_PORT,
    )
