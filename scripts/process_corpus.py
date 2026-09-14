#!/usr/bin/env python3
"""Parse, clean and structurally chunk the local rail-transit corpus."""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from bs4 import BeautifulSoup, Tag, UnicodeDammit
from pypdf import PdfReader


PROCESSOR_VERSION = "1.0.0"
SUPPORTED_SUFFIXES = {".pdf", ".html", ".htm", ".zip", ".md", ".txt"}
ZIP_MEMBER_SUFFIXES = {".pdf", ".html", ".htm", ".md", ".txt"}
PAGE_NUMBER_RE = re.compile(r"^(?:第\s*)?[-—–]?\s*\d{1,4}\s*(?:页|/\s*\d{1,4})?\s*[-—–]?$")
HEADING_PATTERNS = (
    re.compile(r"^(?:第[一二三四五六七八九十百零〇0-9]+(?:章|节|部分|篇))"),
    re.compile(r"^[一二三四五六七八九十]+[、.]\s*\S+"),
    re.compile(r"^\(?[0-9]{1,2}\)?[、.]\s*\S+"),
    re.compile(r"^(?:chapter|section|part|appendix)\s+[A-Z0-9IVX]+", re.I),
    re.compile(r"^[A-Z][A-Z0-9 /&(),.-]{5,80}$"),
)
NOISE_LINES = {
    "打印本页",
    "关闭窗口",
    "打印",
    "返回顶部",
    "扫一扫在手机打开当前页",
}

logging.getLogger("pypdf").setLevel(logging.ERROR)


class ProcessingError(RuntimeError):
    """Raised when a raw file cannot be converted into usable text."""


@dataclass
class TextBlock:
    text: str
    kind: str = "paragraph"
    page: int | None = None
    section_path: tuple[str, ...] = ()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha1(payload).hexdigest()[:20]}"


def json_safe(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def clean_inline(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "").replace("\ufeff", "")
    text = re.sub(r"[\u200b-\u200f\u2060]", "", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def clean_block(text: str) -> str:
    text = clean_inline(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<=[\u3400-\u9fff])[ \t]+(?=[\u3400-\u9fff])", "", text)
    return text.strip()


def is_heading(text: str) -> bool:
    value = clean_block(text)
    if not value or len(value) > 100 or value[-1:] in "。！？.!?；;，,":
        return False
    return any(pattern.search(value) for pattern in HEADING_PATTERNS)


def update_section_path(current: list[str], heading: str, level: int | None = None) -> list[str]:
    heading = clean_block(heading)
    if level is None:
        level = 1 if len(heading) <= 35 else 2
    level = max(1, min(level, 6))
    updated = current[: level - 1]
    updated.append(heading)
    return updated


def decode_text(body: bytes) -> str:
    if body.startswith(b"\x1f\x8b"):
        try:
            body = gzip.decompress(body)
        except OSError as exc:
            raise ProcessingError(f"gzip解压失败: {exc}") from exc
    result = UnicodeDammit(body, is_html=False)
    if result.unicode_markup is None:
        return body.decode("utf-8", errors="replace")
    return result.unicode_markup


def pdf_lines_to_blocks(
    pages: list[list[str]], repeated_lines: set[str]
) -> tuple[list[TextBlock], list[int], list[str]]:
    blocks: list[TextBlock] = []
    page_char_counts: list[int] = []
    warnings: list[str] = []
    section_path: list[str] = []

    for page_number, raw_lines in enumerate(pages, start=1):
        lines: list[str] = []
        for raw in raw_lines:
            line = clean_block(raw)
            if not line or line in repeated_lines or PAGE_NUMBER_RE.fullmatch(line):
                continue
            lines.append(line)

        page_char_counts.append(sum(len(line) for line in lines))
        paragraph: list[str] = []

        def flush_paragraph() -> None:
            if not paragraph:
                return
            text = clean_block(" ".join(paragraph))
            if text:
                blocks.append(TextBlock(text, "paragraph", page_number, tuple(section_path)))
            paragraph.clear()

        for line in lines:
            if is_heading(line):
                flush_paragraph()
                section_path = update_section_path(section_path, line)
                blocks.append(TextBlock(line, "heading", page_number, tuple(section_path)))
                continue
            paragraph.append(line)
            if len(" ".join(paragraph)) >= 240 and line[-1:] in "。！？.!?；;：:":
                flush_paragraph()
        flush_paragraph()

    if page_char_counts:
        empty_pages = sum(count < 30 for count in page_char_counts)
        average = sum(page_char_counts) / len(page_char_counts)
        if empty_pages / len(page_char_counts) >= 0.4 or average < 80:
            warnings.append(
                f"ocr_recommended: {empty_pages}/{len(page_char_counts)}页文本过少，"
                f"平均每页{average:.1f}字符"
            )
    return blocks, page_char_counts, warnings


def extract_pdf_with_macos_vision(
    pdf_path: Path, ocr_script: Path
) -> tuple[list[TextBlock], list[int], list[str]]:
    swift = shutil.which("swift")
    if sys.platform != "darwin" or swift is None or not ocr_script.exists():
        raise ProcessingError(
            "PDF没有文本层且OCR不可用；macOS可使用Vision后备，其他系统请先用OCRmyPDF生成文本层。"
        )
    cache_root = Path(tempfile.gettempdir()) / "rail_transit_swift_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["SWIFT_MODULECACHE_PATH"] = str(cache_root)
    environment["CLANG_MODULE_CACHE_PATH"] = str(cache_root)
    try:
        result = subprocess.run(
            [swift, str(ocr_script), str(pdf_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,
            env=environment,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        details = getattr(exc, "stderr", "") or str(exc)
        raise ProcessingError(f"macOS Vision OCR失败: {details.strip()}") from exc

    page_lines: list[list[str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProcessingError(f"OCR输出不是有效JSON: {line[:120]}") from exc
        page_lines.append(str(record.get("text", "")).splitlines())
    if not page_lines:
        raise ProcessingError("macOS Vision OCR未返回任何页面。")
    blocks, page_char_counts, warnings = pdf_lines_to_blocks(page_lines, set())
    warnings.append("ocr_applied: macOS Vision")
    return blocks, page_char_counts, warnings


def extract_pdf(
    body: bytes, pdf_path: Path | None = None, ocr_script: Path | None = None
) -> tuple[list[TextBlock], dict[str, Any], list[str]]:
    try:
        reader = PdfReader(io.BytesIO(body), strict=False)
    except Exception as exc:
        raise ProcessingError(f"PDF打开失败: {exc}") from exc

    page_lines: list[list[str]] = []
    edge_candidates: Counter[str] = Counter()
    extraction_warnings: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            try:
                text = page.extract_text(extraction_mode="layout") or ""
            except TypeError:
                text = page.extract_text() or ""
        except Exception as exc:
            text = ""
            extraction_warnings.append(f"page_{page_number}_extract_failed: {exc}")
        lines = [clean_block(line) for line in text.splitlines()]
        lines = [line for line in lines if line]
        page_lines.append(lines)
        for line in [*lines[:3], *lines[-3:]]:
            if 2 <= len(line) <= 100 and not PAGE_NUMBER_RE.fullmatch(line):
                edge_candidates[line] += 1

    repeat_threshold = max(3, int(len(page_lines) * 0.4 + 0.999))
    repeated = {line for line, count in edge_candidates.items() if count >= repeat_threshold}
    blocks, page_char_counts, warnings = pdf_lines_to_blocks(page_lines, repeated)
    warnings.extend(extraction_warnings)
    if repeated:
        warnings.append(f"removed_repeated_headers_or_footers: {len(repeated)}")
    if sum(page_char_counts) < 40 and pdf_path is not None and ocr_script is not None:
        blocks, page_char_counts, ocr_warnings = extract_pdf_with_macos_vision(pdf_path, ocr_script)
        warnings = [warning for warning in warnings if not warning.startswith("ocr_recommended")]
        warnings.extend(ocr_warnings)
    metadata = {
        "page_count": len(reader.pages),
        "page_character_counts": page_char_counts,
        "pdf_metadata": json_safe(dict(reader.metadata or {})),
    }
    return blocks, metadata, warnings


def candidate_score(node: Tag) -> float:
    text = clean_block(node.get_text(" ", strip=True))
    if not text:
        return -1.0
    link_text = " ".join(link.get_text(" ", strip=True) for link in node.find_all("a"))
    link_ratio = min(len(link_text) / max(len(text), 1), 1.0)
    paragraph_bonus = min(len(node.find_all("p")), 30) * 80
    return len(text) * (1.0 - 0.75 * link_ratio) + paragraph_bonus


def choose_main_node(soup: BeautifulSoup) -> Tag:
    selectors = [
        "article",
        "main",
        "#UCAP-CONTENT",
        ".TRS_Editor",
        ".article-content",
        ".article_content",
        ".article",
        ".detail-content",
        ".detail_content",
        ".pages_content",
        ".zhenwen_neir",
        ".content",
        "#zoom",
        "#content",
    ]
    candidates: list[Tag] = []
    seen: set[int] = set()
    for selector in selectors:
        for node in soup.select(selector):
            if id(node) not in seen:
                seen.add(id(node))
                candidates.append(node)
    substantial = [node for node in candidates if len(clean_block(node.get_text(" ", strip=True))) >= 200]
    if substantial:
        return max(substantial, key=candidate_score)
    if soup.body is not None:
        return soup.body
    if not candidates:
        raise ProcessingError("HTML中未找到可解析的正文节点。")
    return max(candidates, key=candidate_score)


def extract_html(body: bytes) -> tuple[list[TextBlock], dict[str, Any], list[str]]:
    compressed = body.startswith(b"\x1f\x8b")
    if compressed:
        try:
            body = gzip.decompress(body)
        except OSError as exc:
            raise ProcessingError(f"HTML gzip解压失败: {exc}") from exc
    charset_match = re.search(br"charset\s*=\s*[\"']?([^\"'\s;>]+)", body[:20000], re.I)
    declared_encoding = charset_match.group(1).decode("ascii", errors="ignore") if charset_match else None
    if declared_encoding and declared_encoding.lower().replace("-", "") in {"gb2312", "gbk", "gb18030"}:
        declared_encoding = "gb18030"
    try:
        decoded_html = body.decode(declared_encoding or "utf-8")
    except (LookupError, UnicodeDecodeError):
        decoded_html = decode_text(body)
    soup = BeautifulSoup(decoded_html, "html.parser")
    for tag in soup.find_all(
        ["script", "style", "noscript", "template", "svg", "nav", "header", "footer", "aside", "form", "iframe"]
    ):
        tag.decompose()

    main = choose_main_node(soup)
    blocks: list[TextBlock] = []
    section_path: list[str] = []
    block_names = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr", "pre"}

    for node in main.find_all(block_names):
        if any(parent is not main and getattr(parent, "name", None) in block_names for parent in node.parents):
            continue
        name = node.name.lower()
        if name == "tr":
            cells = [clean_block(cell.get_text(" ", strip=True)) for cell in node.find_all(["th", "td"])]
            text = " | ".join(cell for cell in cells if cell)
        else:
            text = clean_block(node.get_text(" ", strip=True))
        if not text or text in NOISE_LINES:
            continue
        if name.startswith("h") or is_heading(text):
            level = int(name[1]) if name.startswith("h") else None
            section_path = update_section_path(section_path, text, level)
            blocks.append(TextBlock(text, "heading", None, tuple(section_path)))
        else:
            blocks.append(TextBlock(text, "code" if name == "pre" else "paragraph", None, tuple(section_path)))

    if not blocks:
        fallback = clean_block(main.get_text(" ", strip=True))
        if fallback:
            blocks = [TextBlock(fallback)]
    if not blocks:
        raise ProcessingError("HTML正文为空。")

    metadata = {
        "html_title": clean_block(soup.title.get_text(" ", strip=True)) if soup.title else None,
        "detected_encoding": declared_encoding or soup.original_encoding or "utf-8",
        "was_gzip_encoded": compressed,
    }
    return blocks, metadata, []


def strip_markdown(text: str) -> str:
    text = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s*>\s?", "", text)
    text = re.sub(r"(?<!\\)[*_~]{1,3}", "", text)
    return clean_block(text)


def extract_markdown(body: bytes) -> tuple[list[TextBlock], dict[str, Any], list[str]]:
    text = decode_text(body)
    blocks: list[TextBlock] = []
    section_path: list[str] = []
    paragraph: list[str] = []
    in_fence = False

    def flush(kind: str = "paragraph") -> None:
        if not paragraph:
            return
        value = clean_block("\n".join(paragraph)) if kind == "code" else strip_markdown(" ".join(paragraph))
        if value:
            blocks.append(TextBlock(value, kind, None, tuple(section_path)))
        paragraph.clear()

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            flush("code" if in_fence else "paragraph")
            in_fence = not in_fence
            continue
        if in_fence:
            paragraph.append(line)
            continue
        heading = re.match(r"^\s*(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush()
            value = strip_markdown(heading.group(2))
            section_path = update_section_path(section_path, value, len(heading.group(1)))
            blocks.append(TextBlock(value, "heading", None, tuple(section_path)))
        elif not line.strip():
            flush()
        else:
            paragraph.append(line)
    flush("code" if in_fence else "paragraph")
    if not blocks:
        raise ProcessingError("Markdown或文本正文为空。")
    return blocks, {}, []


def parse_content(
    body: bytes,
    suffix: str,
    pdf_path: Path | None = None,
    ocr_script: Path | None = None,
) -> tuple[list[TextBlock], dict[str, Any], list[str], str]:
    suffix = suffix.lower()
    if suffix == ".pdf":
        blocks, metadata, warnings = extract_pdf(body, pdf_path=pdf_path, ocr_script=ocr_script)
        parser = "pypdf+macos-vision" if any(w.startswith("ocr_applied") for w in warnings) else "pypdf"
        return blocks, metadata, warnings, parser
    if suffix in {".html", ".htm"}:
        blocks, metadata, warnings = extract_html(body)
        return blocks, metadata, warnings, "beautifulsoup4"
    if suffix in {".md", ".txt"}:
        blocks, metadata, warnings = extract_markdown(body)
        return blocks, metadata, warnings, "markdown"
    raise ProcessingError(f"不支持的文件类型: {suffix}")


def assemble_text(blocks: list[TextBlock]) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    spans: list[dict[str, Any]] = []
    cursor = 0
    for block in blocks:
        text = clean_block(block.text)
        if not text:
            continue
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append(
            {
                "start": start,
                "end": cursor,
                "kind": block.kind,
                "page": block.page,
                "section_path": list(block.section_path),
            }
        )
    return "".join(parts), spans


def source_metadata(source: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "title",
        "language",
        "jurisdiction",
        "category",
        "version",
        "publication_date",
        "source_page",
        "repository_policy",
        "license",
    )
    return json_safe({key: source.get(key) for key in keys})


def make_document(
    source: dict[str, Any],
    project_root: Path,
    raw_path: Path,
    body: bytes,
    logical_path: str,
    suffix: str,
    member_path: str | None = None,
) -> dict[str, Any]:
    ocr_script = project_root / "scripts" / "ocr_pdf_macos.swift"
    blocks, parser_metadata, warnings, parser_name = parse_content(
        body,
        suffix,
        pdf_path=raw_path if suffix.lower() == ".pdf" and member_path is None else None,
        ocr_script=ocr_script,
    )
    text, spans = assemble_text(blocks)
    if len(text) < 40:
        raise ProcessingError(f"有效正文过短，仅{len(text)}字符。")
    identity = f"{source['id']}::{logical_path}"
    document_id = stable_id("doc", identity)
    title = source.get("title")
    if member_path:
        title = f"{title} - {Path(member_path).name}"
    return {
        "document_id": document_id,
        "source": source_metadata(source),
        "title": title,
        "raw_path": str(raw_path.relative_to(project_root)),
        "member_path": member_path,
        "file_type": suffix.lstrip("."),
        "content_sha256": sha256_bytes(body),
        "parser": parser_name,
        "processor_version": PROCESSOR_VERSION,
        "character_count": len(text),
        "block_count": len(spans),
        "text": text,
        "spans": spans,
        "parser_metadata": json_safe(parser_metadata),
        "warnings": warnings,
    }


def zip_member_allowed(name: str, patterns: list[str]) -> bool:
    path = Path(name)
    if name.startswith("/") or ".." in path.parts or path.suffix.lower() not in ZIP_MEMBER_SUFFIXES:
        return False
    return not patterns or any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def parse_raw_file(
    source: dict[str, Any], project_root: Path, raw_path: Path
) -> list[dict[str, Any]]:
    body = raw_path.read_bytes()
    suffix = raw_path.suffix.lower()
    relative = str(raw_path.relative_to(project_root))
    if suffix != ".zip":
        return [make_document(source, project_root, raw_path, body, relative, suffix)]

    patterns = list(source.get("process", {}).get("include_members", []))
    documents: list[dict[str, Any]] = []
    try:
        archive = zipfile.ZipFile(io.BytesIO(body))
    except zipfile.BadZipFile as exc:
        raise ProcessingError(f"ZIP打开失败: {exc}") from exc

    with archive:
        candidates = [info for info in archive.infolist() if not info.is_dir() and zip_member_allowed(info.filename, patterns)]
        if len(candidates) > 100:
            raise ProcessingError(f"ZIP中待解析成员过多: {len(candidates)}")
        for info in candidates:
            if info.file_size > 30 * 1024 * 1024:
                raise ProcessingError(f"ZIP成员超过30 MB限制: {info.filename}")
            member_body = archive.read(info)
            logical_path = f"{relative}!/{info.filename}"
            documents.append(
                make_document(
                    source,
                    project_root,
                    raw_path,
                    member_body,
                    logical_path,
                    Path(info.filename).suffix.lower(),
                    member_path=info.filename,
                )
            )
    if not documents:
        raise ProcessingError("ZIP中没有符合process.include_members的可解析文件。")
    return documents


def choose_chunk_end(text: str, start: int, max_chars: int, min_chars: int) -> int:
    hard_end = min(len(text), start + max_chars)
    if hard_end == len(text):
        return hard_end
    search_start = min(hard_end, start + max(min_chars, int(max_chars * 0.6)))
    candidates: list[int] = []
    for delimiter in ("\n\n", "。", "！", "？", ". ", "! ", "? ", "；", "; "):
        position = text.rfind(delimiter, search_start, hard_end)
        if position >= 0:
            candidates.append(position + len(delimiter))
    return max(candidates) if candidates else hard_end


def choose_next_start(text: str, start: int, end: int, overlap: int, min_chars: int) -> int:
    target = max(start + 1, end - overlap)
    floor = max(start + 1, target - overlap // 2)
    candidates: list[int] = []
    for delimiter in ("\n\n", "。", "！", "？", ". ", "! ", "? ", "；", "; "):
        position = text.rfind(delimiter, floor, end)
        if position >= 0:
            candidates.append(position + len(delimiter))
    next_start = max(candidates) if candidates else target
    if len(text) - next_start < min_chars:
        next_start = max(start + 1, len(text) - min_chars)
    return next_start


def overlapping_spans(spans: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    return [span for span in spans if span["start"] < end and span["end"] > start]


def chunk_document(
    document: dict[str, Any], max_chars: int, overlap: int, min_chars: int
) -> list[dict[str, Any]]:
    text = document["text"]
    chunks: list[dict[str, Any]] = []
    start = 0
    while start < len(text):
        end = choose_chunk_end(text, start, max_chars, min_chars)
        value = text[start:end].strip()
        if not value:
            start = end
            continue
        leading = len(text[start:end]) - len(text[start:end].lstrip())
        trailing = len(text[start:end]) - len(text[start:end].rstrip())
        actual_start = start + leading
        actual_end = end - trailing
        spans = overlapping_spans(document["spans"], actual_start, actual_end)
        pages = sorted({span["page"] for span in spans if span.get("page") is not None})
        section_paths: list[list[str]] = []
        seen_sections: set[tuple[str, ...]] = set()
        for span in spans:
            path = tuple(span.get("section_path") or [])
            if path and path not in seen_sections:
                seen_sections.add(path)
                section_paths.append(list(path))
        chunk_index = len(chunks)
        chunks.append(
            {
                "chunk_id": stable_id("chk", document["document_id"], actual_start, actual_end, value),
                "document_id": document["document_id"],
                "source_id": document["source"]["id"],
                "chunk_index": chunk_index,
                "title": document["title"],
                "text": value,
                "character_count": len(value),
                "char_start": actual_start,
                "char_end": actual_end,
                "page_start": pages[0] if pages else None,
                "page_end": pages[-1] if pages else None,
                "section_path": section_paths[0] if section_paths else [],
                "section_paths": section_paths,
                "source_page": document["source"].get("source_page"),
                "raw_path": document["raw_path"],
                "member_path": document.get("member_path"),
                "content_sha256": document["content_sha256"],
                "repository_policy": document["source"].get("repository_policy"),
                "license": document["source"].get("license"),
            }
        )
        if end >= len(text):
            break
        next_start = choose_next_start(text, start, end, overlap, min_chars)
        start = next_start if next_start > start else end
    return chunks


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".part", delete=False) as handle:
        temp_path = Path(handle.name)
        for record in records:
            handle.write(json.dumps(json_safe(record), ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
    temp_path.replace(path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".part", delete=False) as handle:
        temp_path = Path(handle.name)
        json.dump(json_safe(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
    temp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[], metavar="SOURCE_ID")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--min-chunk-size", type=int, default=180)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.chunk_size < 200:
        raise ProcessingError("--chunk-size不能小于200。")
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        raise ProcessingError("--chunk-overlap必须大于等于0且小于chunk-size。")
    if args.min_chunk_size < 1 or args.min_chunk_size > args.chunk_size:
        raise ProcessingError("--min-chunk-size必须在1和chunk-size之间。")


def main() -> int:
    args = parse_args()
    validate_args(args)
    manifest_path = args.manifest.expanduser().resolve()
    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    project_root = manifest_path.parent
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    source_map = {source["id"]: source for source in manifest["sources"]}
    selected = set(args.only)
    unknown = selected - set(source_map)
    if unknown:
        raise ProcessingError(f"未知source id: {', '.join(sorted(unknown))}")

    documents: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    raw_files: list[tuple[dict[str, Any], Path]] = []
    for group in ("open", "restricted"):
        group_dir = raw_dir / group
        if not group_dir.exists():
            continue
        for source_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            source = source_map.get(source_dir.name)
            if source is None or (selected and source_dir.name not in selected):
                continue
            for path in sorted(source_dir.iterdir()):
                if (
                    path.is_file()
                    and not path.name.startswith((".", "_"))
                    and path.suffix.lower() in SUPPORTED_SUFFIXES
                ):
                    raw_files.append((source, path))

    for source, path in raw_files:
        print(f"[{source['id']}] {path.name}")
        try:
            parsed = parse_raw_file(source, project_root, path)
            documents.extend(parsed)
            for document in parsed:
                print(
                    f"  parsed: {document['document_id']} "
                    f"chars={document['character_count']} warnings={len(document['warnings'])}"
                )
        except (OSError, ProcessingError, zipfile.BadZipFile) as exc:
            errors.append({"source_id": source["id"], "raw_path": str(path), "error": str(exc)})
            print(f"  failed: {exc}", file=sys.stderr)

    documents.sort(key=lambda item: (item["source"]["id"], item["raw_path"], item.get("member_path") or ""))
    chunks: list[dict[str, Any]] = []
    for document in documents:
        chunks.extend(chunk_document(document, args.chunk_size, args.chunk_overlap, args.min_chunk_size))

    documents_path = output_dir / "documents.jsonl"
    chunks_path = output_dir / "chunks.jsonl"
    report_path = output_dir / "processing_report.json"
    atomic_write_jsonl(documents_path, documents)
    atomic_write_jsonl(chunks_path, chunks)

    source_counts: dict[str, int] = defaultdict(int)
    format_counts: dict[str, int] = defaultdict(int)
    warning_count = 0
    ocr_recommended: list[str] = []
    for document in documents:
        source_counts[document["source"]["id"]] += 1
        format_counts[document["file_type"]] += 1
        warning_count += len(document["warnings"])
        if any(str(warning).startswith("ocr_recommended") for warning in document["warnings"]):
            ocr_recommended.append(document["document_id"])

    report = {
        "processor_version": PROCESSOR_VERSION,
        "generated_at": utc_now(),
        "manifest_path": str(manifest_path),
        "raw_dir": str(raw_dir),
        "configuration": {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "min_chunk_size": args.min_chunk_size,
        },
        "raw_file_count": len(raw_files),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "source_counts": dict(sorted(source_counts.items())),
        "format_counts": dict(sorted(format_counts.items())),
        "warning_count": warning_count,
        "ocr_recommended_document_ids": ocr_recommended,
        "errors": errors,
    }
    atomic_write_json(report_path, report)
    print(f"完成：{len(documents)}个文档，{len(chunks)}个切片，{len(errors)}个错误。")
    print(f"文档：{documents_path}")
    print(f"切片：{chunks_path}")
    print(f"报告：{report_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ProcessingError, OSError, KeyError, yaml.YAMLError) as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        raise SystemExit(2)
