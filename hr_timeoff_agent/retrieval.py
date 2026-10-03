"""Retrieval over the tenant's leave handbook and past human decisions.

Rules still decide whether a request is allowed (policy.py). Retrieval adds the
guidance a manager would look up next — what the handbook says to do when a
rule fails, and how similar requests were decided before — so the
recommendation can cite it.

Two properties matter more than recall:

1. Tenant and audience are enforced inside the vector query, before ranking.
   A passage the approver may not see is never a candidate, so it cannot leak
   through a score, a rank or a snippet.
2. The tenant comes from the loaded tenant, never from the model or the query.

Search is hybrid: a dense embedding (meaning) and BM25 (exact terms such as
"close", "unpaid", "coverage"), fused with reciprocal rank fusion. Dense alone
kept ranking the generic PTO passages first; policy language needs exact terms.

Vectors live in Qdrant. Here it runs in-process (`:memory:`), so a fresh clone
needs no server; the same client API talks to a Qdrant server in production.
Embeddings are content-addressed in fixtures/embeddings.json, so offline runs
never load either embedding model.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from functools import cached_property
from pathlib import Path

from qdrant_client import QdrantClient, models as qm

from .models import Finding, Passage

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Overridable (like the LLM fixtures) so a self-test can record into a scratch copy.
EMBED_CACHE = Path(
    os.environ.get("HR_AGENT_EMBEDDINGS")
    or Path(__file__).resolve().parent.parent / "fixtures" / "embeddings.json"
)

EMBED_MODEL = os.environ.get("HR_AGENT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384
SPARSE_MODEL = "Qdrant/bm25"

# Reciprocal rank fusion: score = sum of 1 / (rank + RRF_K) over both retrievers,
# rank starting at 0. RRF_K = 2 matches Qdrant's own default.
RRF_K = 2

HANDBOOK_K = 3
PRECEDENT_K = 2

# Which handbook audiences each reader may retrieve. The approver is a manager,
# so HR-only guidance never reaches the recommendation.
AUDIENCES = {
    "employee": ["all"],
    "manager": ["all", "manager"],
    "hr": ["all", "manager", "hr"],
}


class EmbeddingCacheMiss(RuntimeError):
    pass


class Embedder:
    """fastembed (dense + BM25) behind a content-addressed cache.

    Queries and passages are embedded differently (bge prefixes queries with a
    search instruction), so the kind is part of the key.
    """

    def __init__(self, model: str = EMBED_MODEL, cache_path: Path = EMBED_CACHE):
        self.model_name = model
        self.cache_path = cache_path
        self._cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self._dirty = False

    def _key(self, model: str, kind: str, text: str) -> str:
        blob = json.dumps({"model": model, "kind": kind, "text": text}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    @cached_property
    def _dense(self):
        from fastembed import TextEmbedding  # loaded only on a cache miss

        return TextEmbedding(self.model_name)

    @cached_property
    def _sparse(self):
        from fastembed import SparseTextEmbedding

        return SparseTextEmbedding(SPARSE_MODEL)

    def _cached(self, model: str, texts: list[str], kind: str, compute) -> list:
        keys = [self._key(model, kind, t) for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self._cache]
        if missing and self.cache_path.exists():
            # Another process (the MCP server Claude Code spawns while recording) may
            # have embedded these since this one loaded the file.
            self._cache = {**json.loads(self.cache_path.read_text()), **self._cache}
            missing = [(k, t) for k, t in missing if k not in self._cache]
        if missing:
            if os.environ.get("HR_AGENT_OFFLINE") == "1":
                raise EmbeddingCacheMiss(
                    f"{len(missing)} {kind} embedding(s) for {model} not cached and "
                    "HR_AGENT_OFFLINE=1. Unset it to compute them locally with fastembed."
                )
            for (k, t), value in zip(missing, compute([t for _, t in missing])):
                self._cache[k] = {"model": model, "kind": kind, "text": t[:80], **value}
            self._dirty = True
        return [self._cache[k] for k in keys]

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        def compute(batch):
            fn = self._dense.query_embed if kind == "query" else self._dense.passage_embed
            return ({"vector": [round(float(x), 6) for x in v]} for v in fn(batch))

        return [e["vector"] for e in self._cached(self.model_name, texts, kind, compute)]

    def embed_sparse(self, texts: list[str], kind: str) -> list[qm.SparseVector]:
        def compute(batch):
            fn = self._sparse.query_embed if kind == "query" else self._sparse.passage_embed
            return (
                {"indices": [int(i) for i in v.indices], "values": [round(float(x), 6) for x in v.values]}
                for v in fn(batch)
            )

        return [
            qm.SparseVector(indices=e["indices"], values=e["values"])
            for e in self._cached(SPARSE_MODEL, texts, kind, compute)
        ]

    def save(self) -> None:
        if self._dirty:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, sort_keys=True) + "\n")
            self._dirty = False


def _point_id(tenant_id: str, doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{tenant_id}/{doc_id}"))


class PolicyIndex:
    """Two Qdrant collections: handbook passages and precedents, for every tenant.

    All tenants share the collections (the pooled model); isolation comes from
    the tenant filter applied inside every query.
    """

    def __init__(self, data_dir: Path | None = None, embedder: Embedder | None = None, *,
                 docs: dict | None = None, qdrant_url: str | None = None):
        """In-process by default; with `qdrant_url`, a Qdrant server that keeps the index."""
        d = data_dir or DATA_DIR
        self.handbook = docs["handbook.json"] if docs else json.loads((d / "handbook.json").read_text())
        self.precedents = docs["precedents.json"] if docs else json.loads((d / "precedents.json").read_text())
        self.embedder = embedder or Embedder()
        self.client = QdrantClient(url=qdrant_url) if qdrant_url else QdrantClient(":memory:")
        self._load()

    def _load(self) -> None:
        for name, docs, id_field in (
            ("handbook", self.handbook, "passage_id"),
            ("precedents", self.precedents, "precedent_id"),
        ):
            # Idempotent: a server that already holds the collection keeps it, and
            # the upsert below rewrites the same deterministic point ids.
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    name,
                    vectors_config={"dense": qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE)},
                    sparse_vectors_config={"bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
                )
            texts = [f"{doc['title']}. {doc['text']}" if "title" in doc else doc["text"] for doc in docs]
            dense = self.embedder.embed(texts, kind="passage")
            sparse = self.embedder.embed_sparse(texts, kind="passage")
            self.client.upsert(
                name,
                points=[
                    qm.PointStruct(
                        id=_point_id(doc["tenant_id"], doc[id_field]),
                        vector={"dense": d, "bm25": sp},
                        payload=doc,
                    )
                    for doc, d, sp in zip(docs, dense, sparse)
                ],
            )
        self.embedder.save()

    def _query(self, collection: str, query: str, flt: qm.Filter, k: int, id_field: str) -> list[qm.ScoredPoint]:
        [dense] = self.embedder.embed([query], kind="query")
        [sparse] = self.embedder.embed_sparse([query], kind="query")
        self.embedder.save()
        # Each retriever runs in Qdrant with the filter, so neither ever ranks a
        # passage from another tenant or audience. Fusion happens here rather
        # than in Qdrant: its RRF orders tied scores inside each list differently
        # on macOS and Linux (BM25 ties are common), which changed which passages
        # made the cut and broke offline replay in CI. Ties break by id instead.
        fused: dict[str, float] = {}
        points: dict[str, qm.ScoredPoint] = {}
        for vector, using in ((dense, "dense"), (sparse, "bm25")):
            hits = self.client.query_points(
                collection, query=vector, using=using, query_filter=flt, limit=k * 4, with_payload=True
            ).points
            hits.sort(key=lambda p: (-round(p.score, 6), p.payload[id_field]))
            for rank, p in enumerate(hits):
                pid = p.payload[id_field]
                fused[pid] = fused.get(pid, 0.0) + 1.0 / (rank + RRF_K)
                points[pid] = p
        order = sorted(fused, key=lambda pid: (-round(fused[pid], 9), pid))[:k]
        return [points[pid].model_copy(update={"score": fused[pid]}) for pid in order]

    def search_handbook(self, query: str, *, tenant_id: str, reader: str = "manager", k: int = HANDBOOK_K) -> list[Passage]:
        flt = qm.Filter(
            must=[
                qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id)),
                qm.FieldCondition(key="audience", match=qm.MatchAny(any=AUDIENCES[reader])),
            ]
        )
        return [
            Passage(
                passage_id=p.payload["passage_id"],
                tenant_id=p.payload["tenant_id"],
                kind="handbook",
                title=p.payload["title"],
                text=p.payload["text"],
                score=round(p.score, 4),
            )
            for p in self._query("handbook", query, flt, k, "passage_id")
        ]

    def search_precedents(self, query: str, *, tenant_id: str, k: int = PRECEDENT_K) -> list[Passage]:
        flt = qm.Filter(must=[qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))])
        return [
            Passage(
                passage_id=p.payload["precedent_id"],
                tenant_id=p.payload["tenant_id"],
                kind="precedent",
                title=f"{p.payload['outcome']} by {p.payload['decided_by']} on {p.payload['decided_at']}",
                text=f"{p.payload['text']} Outcome: {p.payload['outcome']}. {p.payload['note']}",
                score=round(p.score, 4),
            )
            for p in self._query("precedents", query, flt, k, "precedent_id")
        ]


def build_query(request: dict, findings: list[Finding]) -> str:
    """Deterministic retrieval query built from what the manager has to resolve.

    Dates and hour counts are left out: they match every PTO passage equally and
    drown out the terms that distinguish one situation from another.
    """
    issues = [f"{f.rule_name}: {f.detail}" for f in findings if f.status != "pass"]
    if not issues:
        return f"Routine time off request where every policy rule passes. Worker note: {request['note']}"
    return "What can the manager do? " + " ".join(issues) + f" Worker note: {request['note']}"
