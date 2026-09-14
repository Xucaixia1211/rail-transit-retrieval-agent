from __future__ import annotations

import math
from typing import Iterable


def first_relevant_rank(ranked_ids: Iterable[str], relevant_ids: set[str], cutoff: int) -> int | None:
    for rank, chunk_id in enumerate(list(ranked_ids)[:cutoff], start=1):
        if chunk_id in relevant_ids:
            return rank
    return None


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def hit_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return float(bool(set(ranked_ids[:k]) & relevant_ids))


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    rank = first_relevant_rank(ranked_ids, relevant_ids, k)
    return 0.0 if rank is None else 1.0 / rank


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranked_ids[:k], start=1)
        if chunk_id in relevant_ids
    )
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / idcg if idcg else 0.0
