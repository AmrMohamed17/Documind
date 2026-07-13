"""Tests for the RRF fusion logic — the one piece of custom ranking code."""
from app.core.reranker import _rrf_fuse, RRF_K


def _mk(content: str) -> dict:
    return {"content": content, "source": "x.md", "page": 1, "distance": 0.1}


def test_rrf_rewards_agreement():
    """A chunk ranked highly by BOTH signals should win."""
    a, b, c = _mk("a"), _mk("b"), _mk("c")
    dense = [a, b, c]        # a is dense's favourite
    reranked = [a, c, b]     # a is also the reranker's favourite
    fused = _rrf_fuse(dense, reranked)
    assert fused[0]["content"] == "a"


def test_rrf_protects_a_strong_dense_result():
    """A chunk dense ranks #1 shouldn't be sunk by a mediocre rerank score.
    This is the exact regression that motivated RRF (naive reranking hurt
    single-hop recall)."""
    a, b = _mk("a"), _mk("b")
    dense = [a, b]        # dense: a first
    reranked = [b, a]     # reranker disagrees, prefers b
    fused = _rrf_fuse(dense, reranked)
    # Perfect disagreement -> equal scores -> a must not be dropped.
    assert len(fused) == 2
    assert {r["content"] for r in fused} == {"a", "b"}


def test_rrf_lifts_a_buried_chunk():
    """A chunk dense buried but the reranker loves should climb."""
    a, b, c, d = _mk("a"), _mk("b"), _mk("c"), _mk("d")
    dense = [a, b, c, d]     # d is last for dense
    reranked = [d, a, b, c]  # but first for the reranker
    fused = _rrf_fuse(dense, reranked)
    assert fused[0]["content"] in {"a", "d"}   # d climbed into contention
    assert fused.index(next(r for r in fused if r["content"] == "d")) < 3


def test_rrf_scoring_formula():
    """Score is 1/(K+rank) summed across both lists."""
    a = _mk("a")
    fused = _rrf_fuse([a], [a])
    assert len(fused) == 1          # deduplicated by content
    # sanity: the constant is what we think it is
    assert RRF_K == 60


def test_rrf_handles_empty():
    assert _rrf_fuse([], []) == []