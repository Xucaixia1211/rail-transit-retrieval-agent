from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .answering import answer_question
from .evaluate import evaluate_pipeline
from .io import load_jsonl, load_yaml
from .retrieval import RetrievalPipeline


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def compact_result(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": item["rank"],
        "score": round(item["score"], 6),
        "source_id": item["source_id"],
        "title": item["title"],
        "page_start": item.get("page_start"),
        "page_end": item.get("page_end"),
        "section_path": item.get("section_path") or [],
        "chunk_id": item["chunk_id"],
        "preview": item["text"][:240].replace("\n", " "),
    }


def build_parser() -> argparse.ArgumentParser:
    root = default_project_root()
    parser = argparse.ArgumentParser(description="Rail-transit hybrid retrieval and grounded QA")
    parser.add_argument("--config", type=Path, default=root / "configs" / "retrieval.yaml")
    parser.add_argument("--chunks", type=Path, default=root / "data" / "processed" / "chunks.jsonl")
    parser.add_argument("--queries", type=Path, default=root / "evaluation" / "queries.jsonl")
    parser.add_argument("--cache-dir", type=Path, default=root / "data" / "cache")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="Build BM25 and dense indexes")
    index.add_argument("--with-reranker", action="store_true")

    search = subparsers.add_parser("search", help="Retrieve evidence chunks")
    search.add_argument("query")
    search.add_argument("--mode", choices=RetrievalPipeline.MODES, default="hybrid")
    search.add_argument("--top-k", type=int, default=5)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate retrieval stages")
    evaluate.add_argument("--modes", default="bm25,dense,hybrid,rerank")
    evaluate.add_argument("--output", type=Path, default=root / "evaluation" / "results.json")
    evaluate.add_argument("--details", type=Path, default=root / "evaluation" / "detailed_results.jsonl")

    ask = subparsers.add_parser("ask", help="Answer from retrieved evidence")
    ask.add_argument("query")
    ask.add_argument("--mode", choices=RetrievalPipeline.MODES, default="hybrid")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--llm", choices=("auto", "never", "required"), default="auto")
    ask.add_argument("--model")

    subparsers.add_parser("doctor", help="Check corpus, models and optional API configuration")
    return parser


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main() -> int:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    if args.command == "doctor":
        status = {
            "chunks": {"path": str(args.chunks), "exists": args.chunks.exists()},
            "queries": {"path": str(args.queries), "exists": args.queries.exists()},
            "embedding_model": config["models"]["embedding"]["name"],
            "reranker_model": config["models"]["reranker"]["name"],
            "openai_api_key": "configured" if os.environ.get("OPENAI_API_KEY") else "not configured; extractive fallback available",
        }
        print_json(status)
        return 0 if args.chunks.exists() else 1

    chunks = load_jsonl(args.chunks)
    pipeline = RetrievalPipeline(chunks, config, args.cache_dir)
    if args.command == "index":
        mode = "rerank" if args.with_reranker else "dense"
        seconds = pipeline.prepare(mode)
        if args.with_reranker:
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            (args.cache_dir / ".models_ready").write_text(
                "Both embedding and reranker models loaded successfully.\n",
                encoding="utf-8",
            )
        print_json({"status": "ready", "chunks": len(chunks), "mode": mode, "setup_seconds": seconds})
        return 0
    if args.command == "search":
        response = pipeline.search(args.query, mode=args.mode, top_k=args.top_k)
        response["results"] = [compact_result(item) for item in response["results"]]
        print_json(response)
        return 0
    if args.command == "evaluate":
        modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
        invalid = set(modes) - set(RetrievalPipeline.MODES)
        if invalid:
            raise ValueError(f"Unknown modes: {', '.join(sorted(invalid))}")
        queries = load_jsonl(args.queries)
        report = evaluate_pipeline(pipeline, queries, modes, args.output, args.details)
        print_json(report["results"])
        return 0
    if args.command == "ask":
        response = pipeline.search(args.query, mode=args.mode, top_k=args.top_k)
        answer = answer_question(
            args.query,
            response["results"],
            config["generation"],
            llm_mode=args.llm,
            model=args.model,
        )
        answer["retrieval"] = {
            "mode": args.mode,
            "latency_ms": response["latency_ms"],
            "stage_latency_ms": response["stage_latency_ms"],
        }
        print_json(answer)
        return 0
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
