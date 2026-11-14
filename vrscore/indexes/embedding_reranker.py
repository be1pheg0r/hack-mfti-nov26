"""Embedding-based reranking for an existing, bounded vacancy shortlist."""

from collections.abc import Callable, Mapping, Sequence
from uuid import uuid4

from vrscore.vector_store import MilvusVectorStore

from .base import SearchHit, SearchResults


class EmbeddingShortlistReranker:
    """Rerank supplied candidates by cosine similarity without loading a model.

    The caller owns gate evaluation, lexical shortlist construction, query
    encoding, and the implementation behind ``encode``. Only shortlist vacancy
    text is passed to that injected encoder. Each rerank uses uniquely scoped
    records and removes them after the search.
    """

    def __init__(
        self,
        vector_store: MilvusVectorStore,
        *,
        encode: Callable[[str], Sequence[float]],
        namespace: str = "embedding-rerank:",
    ) -> None:
        """Create a reranker using a caller-provided text-to-vector function."""
        if not callable(encode):
            raise ValueError("encode must be callable")
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be a non-empty string")
        self.vector_store = vector_store
        self.encode = encode
        self.namespace = namespace

    def rerank(
        self,
        shortlist: SearchResults,
        query_vector: Sequence[float],
        vacancy_texts: Mapping[str, str],
    ) -> SearchResults:
        """Rerank exactly the supplied BM25 candidates, retaining their evidence."""
        hits = shortlist.hits
        ids = [hit.vacancy_id for hit in hits]
        if len(ids) != len(set(ids)):
            raise ValueError("Shortlist contains duplicate vacancy IDs")
        if not hits:
            return SearchResults("embedding_rerank", shortlist.query_id, ())

        record_ids = []
        vectors = []
        request_namespace = f"{self.namespace}{uuid4().hex}:"
        for hit in hits:
            text = vacancy_texts.get(hit.vacancy_id)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing vacancy text for {hit.vacancy_id}")
            record_id = request_namespace + hit.vacancy_id
            record_ids.append(record_id)
            vectors.append(self.encode(text))

        try:
            self.vector_store.upsert_many(dict(zip(record_ids, vectors, strict=True)))
            results = self.vector_store.search(
                query_vector, limit=len(record_ids), record_ids=record_ids
            )
        finally:
            self.vector_store.delete_many(record_ids)
        score_by_id = {result.record_id: result.score for result in results}
        if set(score_by_id) != set(record_ids):
            raise RuntimeError("Vector store did not return every shortlist candidate")
        reranked = sorted(
            hits,
            key=lambda hit: (
                -score_by_id[request_namespace + hit.vacancy_id],
                hit.vacancy_id,
            ),
        )
        return SearchResults(
            index_name="embedding_rerank",
            query_id=shortlist.query_id,
            hits=tuple(
                SearchHit(
                    hit.vacancy_id,
                    score_by_id[request_namespace + hit.vacancy_id],
                    hit.matches,
                    hit.contributions,
                )
                for hit in reranked
            ),
        )
