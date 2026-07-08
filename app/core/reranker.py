# app/core/reranker.py
"""
Two-stage retrieval, stage 2: re-score dense-retrieval candidates with a
lightweight local cross-encoder (FlashRank). The cross-encoder reads the query
and each candidate chunk TOGETHER, so it judges relevance far more precisely
than the separate-vector comparison dense search can afford.
"""
from flashrank import Ranker, RerankRequest

_ranker = None
RRF_K = 60   # standard smoothing constant for Reciprocal Rank Fusion
RERANK_CANDIDATES = 12   # over-retrieve this many, then rerank down



def _rrf_fuse(dense: list[dict], reranked: list[dict]) -> list[dict]:
    """Fuse two orderings of the SAME candidates via Reciprocal Rank Fusion.
    Each item scores 1/(RRF_K + rank) in each list; the two are summed."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranking in (dense, reranked):
        for rank, r in enumerate(ranking, start=1):
            key = r["content"]                 # same chunk => same content
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            items[key] = r
    return sorted(items.values(), key=lambda r: scores[r["content"]], reverse=True)


def rerank_with_rrf(query: str, dense_results: list[dict]) -> list[dict]:
    """Two-stage + fusion: cross-encode the dense candidates, then RRF-fuse the
    dense order with the rerank order so BOTH signals count."""
    if not dense_results:
        return dense_results
    reranked = rerank_results(query, dense_results)
    return _rrf_fuse(dense_results, reranked)


def _get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        # Downloads a ~34MB ONNX model on first use, then caches it.
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    return _ranker


def rerank_results(query: str, results: list[dict]) -> list[dict]:
    """Re-order dense results best-first by cross-encoder relevance.
    Returns the same result dicts, re-sorted, each with a 'rerank_score' added."""
    if not results:
        return results
    passages = [{"id": i, "text": r["content"], "meta": r}
                for i, r in enumerate(results)]
    ranked = _get_ranker().rerank(RerankRequest(query=query, passages=passages))
    out = []
    for item in ranked:                 # already sorted best-first
        r = item["meta"]                # the original result dict
        r["rerank_score"] = item["score"]
        out.append(r)
    return out