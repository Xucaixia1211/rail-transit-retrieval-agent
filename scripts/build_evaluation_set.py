#!/usr/bin/env python3
"""Resolve reviewed evaluation specifications against the current chunk corpus."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def page_distance(chunk: dict[str, Any], hint: int | None) -> int:
    if hint is None or chunk.get("page_start") is None:
        return 0
    start = int(chunk["page_start"])
    end = int(chunk.get("page_end") or start)
    if start <= hint <= end:
        return 0
    return min(abs(start - hint), abs(end - hint))


def resolve_question(spec: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    terms = [normalized(str(term)) for term in spec["evidence_terms"]]
    candidates: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk["source_id"] != spec["source_id"]:
            continue
        text = normalized(chunk["text"])
        if all(term in text for term in terms):
            candidates.append(chunk)
    if not candidates:
        raise ValueError(f"{spec['id']}: no chunk contains all evidence terms: {spec['evidence_terms']}")
    candidates.sort(
        key=lambda chunk: (
            page_distance(chunk, spec.get("page_hint")),
            chunk["character_count"],
            chunk["chunk_index"],
        )
    )
    primary = candidates[0]
    relevant = candidates[:4]
    if primary.get("page_start") is not None:
        locator = {
            "type": "pdf_page",
            "page_start": primary["page_start"],
            "page_end": primary["page_end"],
        }
    else:
        locator = {"type": "web_section", "section_path": primary.get("section_path") or []}
    return {
        "query_id": spec["id"],
        "query": spec["query"],
        "language": spec["language"],
        "answer_reference": spec["answer"],
        "source_id": spec["source_id"],
        "relevant_document_ids": sorted({chunk["document_id"] for chunk in relevant}),
        "relevant_chunk_ids": [chunk["chunk_id"] for chunk in relevant],
        "primary_chunk_id": primary["chunk_id"],
        "source_locator": locator,
        "source_page": primary.get("source_page"),
        "evidence_terms": spec["evidence_terms"],
        "review_status": "verified_against_local_source",
    }


def atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".part", delete=False) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specs = yaml.safe_load(args.specs.read_text(encoding="utf-8"))
    chunks = load_jsonl(args.chunks)
    questions = [resolve_question(spec, chunks) for spec in specs["questions"]]
    ids = [question["query_id"] for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate query ids")
    atomic_jsonl(args.output, questions)
    summary = {
        "questions": len(questions),
        "languages": {language: sum(q["language"] == language for q in questions) for language in sorted({q["language"] for q in questions})},
        "sources": len({q["source_id"] for q in questions}),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
