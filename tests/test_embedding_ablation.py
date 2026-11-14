"""Small contract test for matched-pool ablation."""

from vrscore.benchmarking.embedding_ablation import compare_rankings
from vrscore.indexes.base import SearchHit, SearchResults


def test_compare_rankings_keeps_shortlist_and_ideal_pool():
    baseline = SearchResults("bm25", "r", (SearchHit("a", 2), SearchHit("b", 1)))
    reranked = SearchResults(
        "embedding_rerank", "r", (SearchHit("b", 0.8), SearchHit("a", 0.2))
    )
    row = compare_rankings(baseline, reranked, {"a": 1, "b": 2, "c": 3}, [1, 2])
    assert row["bm25"]["ndcg@1"] == 1 / 7
    assert row["embedding"]["ndcg@1"] == 3 / 7
    assert row["candidate_recall_grade_1"] == 2 / 3
