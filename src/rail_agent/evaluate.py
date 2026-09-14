from __future__ import annotations

import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_json, atomic_jsonl
from .metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from .retrieval import RetrievalPipeline


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def evaluate_pipeline(
    pipeline: RetrievalPipeline,
    queries: list[dict[str, Any]],
    modes: list[str],
    output_path: Path,
    details_path: Path,
) -> dict[str, Any]:
    evaluation_config = pipeline.config["evaluation"]
    recall_ks = [int(value) for value in evaluation_config.get("recall_k", [5, 10, 20])]
    mrr_k = int(evaluation_config.get("mrr_k", 10))
    ndcg_k = int(evaluation_config.get("ndcg_k", 10))
    max_k = max([*recall_ks, mrr_k, ndcg_k])
    summaries: dict[str, Any] = {}
    details: list[dict[str, Any]] = []
    setup_seconds: dict[str, float] = {}

    for mode in modes:
        setup_seconds[mode] = pipeline.prepare(mode)
        per_query: list[dict[str, Any]] = []
        for query in queries:
            response = pipeline.search(query["query"], mode=mode, top_k=max_k)
            ranked_ids = [item["chunk_id"] for item in response["results"]]
            relevant_ids = set(query["relevant_chunk_ids"])
            metrics = {
                **{f"recall@{k}": recall_at_k(ranked_ids, relevant_ids, k) for k in recall_ks},
                **{f"hit@{k}": hit_at_k(ranked_ids, relevant_ids, k) for k in recall_ks},
                f"mrr@{mrr_k}": reciprocal_rank(ranked_ids, relevant_ids, mrr_k),
                f"ndcg@{ndcg_k}": ndcg_at_k(ranked_ids, relevant_ids, ndcg_k),
            }
            record = {
                "query_id": query["query_id"],
                "mode": mode,
                "language": query["language"],
                "source_id": query["source_id"],
                "latency_ms": response["latency_ms"],
                "stage_latency_ms": response["stage_latency_ms"],
                "metrics": metrics,
                "relevant_chunk_ids": query["relevant_chunk_ids"],
                "retrieved_chunk_ids": ranked_ids,
                "first_result_source_id": response["results"][0]["source_id"] if response["results"] else None,
            }
            per_query.append(record)
            details.append(record)

        latencies = [record["latency_ms"] for record in per_query]
        metric_names = list(per_query[0]["metrics"]) if per_query else []
        summary = {
            "query_count": len(per_query),
            "metrics": {name: mean([record["metrics"][name] for record in per_query]) for name in metric_names},
            "latency_ms": {
                "mean": mean(latencies),
                "p50": float(np.percentile(latencies, 50)) if latencies else 0.0,
                "p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
                "max": max(latencies, default=0.0),
            },
            "by_language": {},
        }
        for language in sorted({query["language"] for query in queries}):
            subset = [record for record in per_query if record["language"] == language]
            summary["by_language"][language] = {
                name: mean([record["metrics"][name] for record in subset]) for name in metric_names
            }
        summaries[mode] = summary

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "chunk_count": len(pipeline.chunks),
            "source_count": len({chunk["source_id"] for chunk in pipeline.chunks}),
        },
        "evaluation_set": {
            "query_count": len(queries),
            "languages": {
                language: sum(query["language"] == language for query in queries)
                for language in sorted({query["language"] for query in queries})
            },
        },
        "models": pipeline.config["models"],
        "setup_seconds": setup_seconds,
        "results": summaries,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "notes": [
            "Latency excludes one-time model loading and corpus embedding construction.",
            "Recall uses all verified overlapping evidence chunks; Hit@K reports whether at least one verified chunk was retrieved.",
        ],
    }
    atomic_json(output_path, report)
    atomic_jsonl(details_path, details)
    return report
