from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .tokenize import tokenize


def top_indices(scores: np.ndarray, top_k: int) -> list[int]:
    if len(scores) == 0:
        return []
    count = min(top_k, len(scores))
    partition = np.argpartition(-scores, count - 1)[:count]
    return partition[np.argsort(-scores[partition], kind="stable")].tolist()


def result_record(chunk: dict[str, Any], score: float, rank: int, stage: str) -> dict[str, Any]:
    return {
        "rank": rank,
        "score": float(score),
        "stage": stage,
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "source_id": chunk["source_id"],
        "title": chunk["title"],
        "text": chunk["text"],
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "section_path": chunk.get("section_path") or [],
        "source_page": chunk.get("source_page"),
        "repository_policy": chunk.get("repository_policy"),
        "license": chunk.get("license"),
    }


def query_aware_excerpt(query: str, text: str, max_characters: int = 420) -> str:
    """Select a model-sized window instead of blindly truncating a long chunk.

    Chinese characters frequently map to one or more subword tokens, so a
    700--1,000 character chunk can lose its answer span at a 512-token model
    limit. Windows are scored only for selection; the Cross-Encoder still makes
    the final relevance decision.
    """
    if len(text) <= max_characters:
        return text
    terms = {
        token
        for token in tokenize(query)
        if len(token) >= 2 and not token.isspace()
    }
    stride = max(120, max_characters // 2)
    starts = list(range(0, max(1, len(text) - max_characters + 1), stride))
    last_start = max(0, len(text) - max_characters)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    def score(window: str) -> tuple[int, int]:
        weighted_matches = sum((len(term) ** 2) * window.lower().count(term) for term in terms)
        return weighted_matches, -len(window)

    best_start = max(starts, key=lambda start: score(text[start : start + max_characters]))
    return text[best_start : best_start + max_characters]


class BM25Retriever:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        corpus = [tokenize(f"{chunk['title']} {chunk['text']}") for chunk in chunks]
        self.index = BM25Okapi(corpus)

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        scores = np.asarray(self.index.get_scores(tokenize(query)), dtype=np.float32)
        return [
            result_record(self.chunks[index], float(scores[index]), rank, "bm25")
            for rank, index in enumerate(top_indices(scores, top_k), start=1)
        ]


class DenseRetriever:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        model_config: dict[str, Any],
        cache_dir: Path,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.config = model_config
        self.cache_dir = cache_dir
        self.model = SentenceTransformer(
            model_config["name"],
            revision=model_config.get("revision"),
            cache_folder=str(cache_dir / "huggingface"),
            trust_remote_code=False,
            device=model_config.get("device", "cpu"),
        )
        self.model.max_seq_length = int(model_config.get("max_seq_length", 512))
        self.embeddings = self._load_or_build_embeddings()

    def _cache_key(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.config["name"].encode("utf-8"))
        digest.update(str(self.config.get("revision", "main")).encode("utf-8"))
        for chunk in self.chunks:
            digest.update(chunk["chunk_id"].encode("utf-8"))
            digest.update(chunk.get("content_sha256", "").encode("utf-8"))
        return digest.hexdigest()[:20]

    def _load_or_build_embeddings(self) -> np.ndarray:
        key = self._cache_key()
        directory = self.cache_dir / "embeddings"
        directory.mkdir(parents=True, exist_ok=True)
        matrix_path = directory / f"{key}.npy"
        metadata_path = directory / f"{key}.json"
        if matrix_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("chunk_ids") == [chunk["chunk_id"] for chunk in self.chunks]:
                matrix = np.load(matrix_path)
                if matrix.shape[0] == len(self.chunks):
                    return matrix
        prefix = self.config.get("passage_prefix", "")
        passages = [f"{prefix}{chunk['title']}\n{chunk['text']}" for chunk in self.chunks]
        matrix = self.model.encode(
            passages,
            batch_size=int(self.config.get("batch_size", 16)),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        temporary = matrix_path.with_suffix(".npy.part")
        with temporary.open("wb") as handle:
            np.save(handle, matrix)
        temporary.replace(matrix_path)
        metadata_path.write_text(
            json.dumps(
                {
                    "model": self.config["name"],
                    "revision": self.config.get("revision"),
                    "chunk_ids": [chunk["chunk_id"] for chunk in self.chunks],
                    "shape": list(matrix.shape),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return matrix

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        vector = self.model.encode(
            [f"{self.config.get('query_prefix', '')}{query}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self.embeddings @ vector.astype(np.float32)
        return [
            result_record(self.chunks[index], float(scores[index]), rank, "dense")
            for rank, index in enumerate(top_indices(scores, top_k), start=1)
        ]


def reciprocal_rank_fusion(
    rankings: list[list[dict[str, Any]]],
    chunk_lookup: dict[str, dict[str, Any]],
    rrf_k: int = 60,
    top_k: int = 50,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    components: dict[str, dict[str, int]] = {}
    for ranking in rankings:
        stage = ranking[0]["stage"] if ranking else "unknown"
        for rank, item in enumerate(ranking, start=1):
            chunk_id = item["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            components.setdefault(chunk_id, {})[stage] = rank
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    results: list[dict[str, Any]] = []
    for rank, chunk_id in enumerate(ordered, start=1):
        record = result_record(chunk_lookup[chunk_id], scores[chunk_id], rank, "hybrid")
        record["component_ranks"] = components[chunk_id]
        results.append(record)
    return results


class Reranker:
    def __init__(self, model_config: dict[str, Any], cache_dir: Path) -> None:
        from sentence_transformers import CrossEncoder

        self.config = model_config
        self.model = CrossEncoder(
            model_config["name"],
            revision=model_config.get("revision"),
            cache_folder=str(cache_dir / "huggingface"),
            trust_remote_code=False,
            device=model_config.get("device", "cpu"),
            max_length=int(model_config.get("max_length", 512)),
        )

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not candidates:
            return []
        pairs = [
            (
                query,
                f"{item['title']}\n{query_aware_excerpt(query, item['text'])}",
            )
            for item in candidates
        ]
        scores = np.asarray(
            self.model.predict(
                pairs,
                batch_size=int(self.config.get("batch_size", 16)),
                show_progress_bar=False,
            )
        ).reshape(-1)
        order = top_indices(scores, min(top_k, len(candidates)))
        results: list[dict[str, Any]] = []
        for rank, index in enumerate(order, start=1):
            record = dict(candidates[index])
            record.update(
                {
                    "rank": rank,
                    "score": float(scores[index]),
                    "stage": "rerank",
                    "pre_rerank_rank": candidates[index]["rank"],
                }
            )
            results.append(record)
        return results


class RetrievalPipeline:
    MODES = ("bm25", "dense", "hybrid", "rerank")

    def __init__(self, chunks: list[dict[str, Any]], config: dict[str, Any], cache_dir: Path) -> None:
        if not chunks:
            raise ValueError("Chunk corpus is empty")
        self.chunks = chunks
        self.config = config
        self.cache_dir = cache_dir
        self.lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
        if len(self.lookup) != len(chunks):
            raise ValueError("Duplicate chunk IDs in corpus")
        self.bm25 = BM25Retriever(chunks)
        self._dense: DenseRetriever | None = None
        self._reranker: Reranker | None = None

    @property
    def retrieval_config(self) -> dict[str, Any]:
        return self.config["retrieval"]

    def prepare(self, mode: str) -> float:
        started = time.perf_counter()
        if mode in {"dense", "hybrid", "rerank"} and self._dense is None:
            self._dense = DenseRetriever(self.chunks, self.config["models"]["embedding"], self.cache_dir)
        if mode == "rerank" and self._reranker is None:
            self._reranker = Reranker(self.config["models"]["reranker"], self.cache_dir)
        return time.perf_counter() - started

    def search(self, query: str, mode: str = "rerank", top_k: int | None = None) -> dict[str, Any]:
        if mode not in self.MODES:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        self.prepare(mode)
        top_k = top_k or int(self.retrieval_config["final_top_k"])
        started = time.perf_counter()
        stage_latency: dict[str, float] = {}
        if mode == "bm25":
            stage = time.perf_counter()
            results = self.bm25.search(query, top_k)
            stage_latency["bm25_ms"] = (time.perf_counter() - stage) * 1000
        elif mode == "dense":
            stage = time.perf_counter()
            assert self._dense is not None
            results = self._dense.search(query, top_k)
            stage_latency["dense_ms"] = (time.perf_counter() - stage) * 1000
        else:
            candidate_k = max(
                int(self.retrieval_config["bm25_top_k"]),
                int(self.retrieval_config["dense_top_k"]),
                int(self.retrieval_config.get("rerank_candidates", 30)),
            )
            stage = time.perf_counter()
            bm25_results = self.bm25.search(query, candidate_k)
            stage_latency["bm25_ms"] = (time.perf_counter() - stage) * 1000
            stage = time.perf_counter()
            assert self._dense is not None
            dense_results = self._dense.search(query, candidate_k)
            stage_latency["dense_ms"] = (time.perf_counter() - stage) * 1000
            stage = time.perf_counter()
            hybrid_results = reciprocal_rank_fusion(
                [bm25_results, dense_results],
                self.lookup,
                rrf_k=int(self.retrieval_config.get("rrf_k", 60)),
                top_k=int(self.retrieval_config.get("hybrid_top_k", candidate_k)),
            )
            stage_latency["rrf_ms"] = (time.perf_counter() - stage) * 1000
            if mode == "hybrid":
                results = hybrid_results[:top_k]
                for rank, result in enumerate(results, start=1):
                    result["rank"] = rank
            else:
                stage = time.perf_counter()
                assert self._reranker is not None
                results = self._reranker.rerank(
                    query,
                    hybrid_results[: int(self.retrieval_config.get("rerank_candidates", 30))],
                    top_k,
                )
                stage_latency["rerank_ms"] = (time.perf_counter() - stage) * 1000
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {"query": query, "mode": mode, "latency_ms": elapsed_ms, "stage_latency_ms": stage_latency, "results": results}
