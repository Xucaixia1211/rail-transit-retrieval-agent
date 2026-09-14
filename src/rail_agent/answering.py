from __future__ import annotations

import os
import re
from typing import Any

from .retrieval import query_aware_excerpt
from .tokenize import tokenize


def locator(item: dict[str, Any]) -> str:
    if item.get("page_start") is not None:
        if item.get("page_end") and item["page_end"] != item["page_start"]:
            return f"pp. {item['page_start']}-{item['page_end']}"
        return f"p. {item['page_start']}"
    section = " > ".join(item.get("section_path") or [])
    return section or "web page"


def evidence_items(results: list[dict[str, Any]], limit: int, max_characters: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    consumed = 0
    for item in results:
        if len(selected) >= limit or consumed >= max_characters:
            break
        text = item["text"][: max(0, max_characters - consumed)]
        consumed += len(text)
        selected.append({**item, "text": text, "citation": f"S{len(selected) + 1}"})
    return selected


def extractive_answer(query: str, evidence: list[dict[str, Any]]) -> str:
    query_tokens = set(tokenize(query))
    candidates: list[tuple[int, int, str, str]] = []
    for source_index, item in enumerate(evidence):
        sentences = re.split(r"(?<=[。！？!?;；.])\s+|\n+", item["text"])
        for order, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            overlap = len(query_tokens & set(tokenize(sentence)))
            candidates.append((overlap, -order, sentence, item["citation"]))
    candidates.sort(reverse=True)
    chosen: list[str] = []
    used: set[str] = set()
    for _, _, sentence, citation in candidates:
        normalized = re.sub(r"\s+", "", sentence)
        if normalized in used:
            continue
        used.add(normalized)
        chosen.append(f"{sentence} [{citation}]")
        if len(chosen) == 2:
            break
    if not chosen:
        return "未在当前检索证据中找到足以回答该问题的内容。"
    return " ".join(chosen)


def openai_answer(query: str, evidence: list[dict[str, Any]], model: str) -> str:
    from openai import OpenAI

    context = "\n\n".join(
        f"[{item['citation']}] {item['title']} | {locator(item)} | {item.get('source_page')}\n{item['text']}"
        for item in evidence
    )
    instructions = (
        "你是轨道交通运维检索助手。只能依据提供的证据回答，不得补充证据中没有的事实。"
        "在每个事实后使用[S1]这类编号引用。证据不足时明确说不知道。回答简洁，并区分事实与建议。"
    )
    prompt = f"问题：{query}\n\n证据：\n{context}"
    client = OpenAI()
    response = client.responses.create(model=model, instructions=instructions, input=prompt)
    return response.output_text.strip()


def answer_question(
    query: str,
    results: list[dict[str, Any]],
    generation_config: dict[str, Any],
    llm_mode: str = "auto",
    model: str | None = None,
) -> dict[str, Any]:
    evidence = evidence_items(
        results,
        int(generation_config.get("max_evidence_chunks", 5)),
        int(generation_config.get("max_evidence_characters", 6500)),
    )
    api_key_available = bool(os.environ.get("OPENAI_API_KEY"))
    if llm_mode == "required" and not api_key_available:
        raise RuntimeError("OPENAI_API_KEY is required for --llm required")
    use_llm = llm_mode != "never" and api_key_available
    if use_llm:
        selected_model = model or os.environ.get(generation_config.get("model_env", "OPENAI_MODEL")) or generation_config["default_model"]
        answer = openai_answer(query, evidence, selected_model)
        generator = f"openai:{selected_model}"
    else:
        answer = extractive_answer(query, evidence)
        generator = "extractive-fallback"
    sources = [
        {
            "citation": item["citation"],
            "title": item["title"],
            "source_id": item["source_id"],
            "chunk_id": item["chunk_id"],
            "locator": locator(item),
            "page_start": item.get("page_start"),
            "page_end": item.get("page_end"),
            "section_path": item.get("section_path") or [],
            "source_page": item.get("source_page"),
            "evidence": query_aware_excerpt(query, item["text"], max_characters=600),
        }
        for item in evidence
    ]
    return {"query": query, "answer": answer, "generator": generator, "sources": sources}
