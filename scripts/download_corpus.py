#!/usr/bin/env python3
"""Download a license-aware, reproducible raw corpus from source_manifest.yaml."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

import yaml


SOURCE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
OPEN_REPOSITORY_POLICIES = {
    "raw_allowed_with_attribution",
    "raw_allowed_noncommercial_sharealike",
    "raw_allowed_with_third_party_exclusions",
}


class ManifestError(ValueError):
    """Raised when the source manifest is invalid."""


class DownloadError(RuntimeError):
    """Raised when a source cannot be downloaded safely."""


class LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self._href = href
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            text = " ".join("".join(self._text).split())
            self.links.append((self._href, text))
            self._href = None
            self._text = []


@dataclass(frozen=True)
class DownloadSettings:
    timeout_seconds: float
    retries: int
    request_delay_seconds: float
    max_file_size_bytes: int
    require_https: bool
    user_agent: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(value: str) -> str:
    decoded = unquote(value)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", decoded).strip("-._")
    return cleaned[:180] or "download"


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestError(f"无法读取manifest: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError("manifest顶层必须是映射。")
    return data


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("manifest_version") != 1:
        raise ManifestError("仅支持manifest_version: 1。")
    defaults = manifest.get("defaults")
    if not isinstance(defaults, dict):
        raise ManifestError("缺少defaults映射。")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ManifestError("sources必须是非空列表。")

    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ManifestError("每个source必须是映射。")
        source_id = source.get("id")
        if not isinstance(source_id, str) or not SOURCE_ID_PATTERN.fullmatch(source_id):
            raise ManifestError(f"非法source id: {source_id!r}")
        if source_id in seen:
            raise ManifestError(f"source id重复: {source_id}")
        seen.add(source_id)

        if source.get("repository_policy") not in OPEN_REPOSITORY_POLICIES | {"manifest_only"}:
            raise ManifestError(f"{source_id}: repository_policy非法。")
        fetch = source.get("fetch")
        if not isinstance(fetch, dict):
            raise ManifestError(f"{source_id}: 缺少fetch映射。")
        kind = fetch.get("kind")
        if kind not in {"file", "webpage", "webpage_bundle", "govuk_raib_collection"}:
            raise ManifestError(f"{source_id}: 不支持fetch.kind={kind!r}")
        urls = fetch.get("urls") if kind == "webpage_bundle" else [fetch.get("url")]
        if not isinstance(urls, list) or not urls or not all(isinstance(url, str) for url in urls):
            raise ManifestError(f"{source_id}: 下载URL缺失。")
        allowed_hosts = fetch.get("allowed_hosts")
        if not isinstance(allowed_hosts, list) or not allowed_hosts:
            raise ManifestError(f"{source_id}: allowed_hosts不能为空。")

    return sources


def settings_from(manifest: dict[str, Any], fetch: dict[str, Any]) -> DownloadSettings:
    defaults = manifest["defaults"]
    max_mb = float(fetch.get("max_file_size_mb", defaults.get("max_file_size_mb", 100)))
    return DownloadSettings(
        timeout_seconds=float(defaults.get("timeout_seconds", 60)),
        retries=int(defaults.get("retries", 3)),
        request_delay_seconds=float(defaults.get("request_delay_seconds", 0.5)),
        max_file_size_bytes=int(max_mb * 1024 * 1024),
        require_https=bool(defaults.get("require_https", True)),
        user_agent=str(defaults.get("user_agent", "RailTransitRetrievalAgent/0.1")),
    )


def validate_url(url: str, allowed_hosts: Iterable[str], require_https: bool) -> None:
    parsed = urlparse(url)
    allowed = {host.lower() for host in allowed_hosts}
    if require_https and parsed.scheme != "https":
        raise DownloadError(f"拒绝非HTTPS URL: {url}")
    if parsed.scheme not in {"http", "https"}:
        raise DownloadError(f"不支持的URL协议: {url}")
    if (parsed.hostname or "").lower() not in allowed:
        raise DownloadError(f"URL主机不在allowlist中: {parsed.hostname}")


def request_bytes(
    url: str,
    allowed_hosts: list[str],
    settings: DownloadSettings,
) -> tuple[bytes, dict[str, str]]:
    validate_url(url, allowed_hosts, settings.require_https)
    last_error: Exception | None = None

    for attempt in range(settings.retries):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": settings.user_agent,
                    "Accept": "text/html,application/pdf,application/zip,application/octet-stream;q=0.9,*/*;q=0.8",
                },
            )
            with urlopen(request, timeout=settings.timeout_seconds) as response:
                final_url = response.geturl()
                validate_url(final_url, allowed_hosts, settings.require_https)
                length = response.headers.get("Content-Length")
                if length and int(length) > settings.max_file_size_bytes:
                    raise DownloadError(
                        f"文件超过大小限制: {int(length)} > {settings.max_file_size_bytes} bytes"
                    )
                body = response.read(settings.max_file_size_bytes + 1)
                if len(body) > settings.max_file_size_bytes:
                    raise DownloadError(f"文件超过大小限制: {settings.max_file_size_bytes} bytes")
                metadata = {
                    "requested_url": url,
                    "final_url": final_url,
                    "content_type": response.headers.get("Content-Type", ""),
                    "etag": response.headers.get("ETag", ""),
                    "last_modified": response.headers.get("Last-Modified", ""),
                }
                return body, metadata
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, DownloadError) as exc:
            last_error = exc
            if attempt + 1 < settings.retries:
                time.sleep(min(2**attempt, 8))

    raise DownloadError(f"下载失败 {url}: {last_error}") from last_error


def validate_payload(body: bytes, filename: str, expected_content_type: str | None) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf" and not body.startswith(b"%PDF-"):
        raise DownloadError(f"{filename}: 响应不是有效PDF。")
    if suffix == ".zip" and not body.startswith(b"PK"):
        raise DownloadError(f"{filename}: 响应不是有效ZIP。")
    if expected_content_type and expected_content_type not in {"application/octet-stream", "binary/octet-stream"}:
        # 真实内容由magic bytes再次校验；这里仅保留期望值用于下载凭证。
        return


def write_atomic(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def extract_links(html: bytes, base_url: str) -> list[tuple[str, str]]:
    parser = LinkExtractor()
    parser.feed(html.decode("utf-8", errors="replace"))
    return [(urljoin(base_url, href), text) for href, text in parser.links]


def make_receipt(
    source: dict[str, Any],
    path: Path,
    body: bytes,
    response_metadata: dict[str, str],
) -> dict[str, Any]:
    return {
        "source_id": source["id"],
        "title": source.get("title"),
        "language": source.get("language"),
        "category": source.get("category"),
        "repository_policy": source.get("repository_policy"),
        "license": source.get("license"),
        "source_page": source.get("source_page"),
        "local_path": str(path),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "downloaded_at": utc_now(),
        **response_metadata,
    }


def download_one(
    source: dict[str, Any],
    url: str,
    filename: str,
    output_dir: Path,
    manifest: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    fetch = source["fetch"]
    settings = settings_from(manifest, fetch)
    path = output_dir / safe_filename(filename)
    if path.exists() and not force:
        body = path.read_bytes()
        return {
            "source_id": source["id"],
            "title": source.get("title"),
            "local_path": str(path),
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "status": "cached",
            "checked_at": utc_now(),
            "repository_policy": source.get("repository_policy"),
            "license": source.get("license"),
            "source_page": source.get("source_page"),
            "requested_url": url,
        }

    body, response_metadata = request_bytes(url, fetch["allowed_hosts"], settings)
    validate_payload(body, filename, fetch.get("expected_content_type"))
    write_atomic(path, body)
    receipt = make_receipt(source, path, body, response_metadata)
    receipt["status"] = "downloaded"
    time.sleep(settings.request_delay_seconds)
    return receipt


def download_raib_collection(
    source: dict[str, Any],
    output_dir: Path,
    manifest: dict[str, Any],
    force: bool,
    max_items_override: int | None,
) -> list[dict[str, Any]]:
    fetch = source["fetch"]
    settings = settings_from(manifest, fetch)
    listing_url = fetch["url"]
    listing, listing_meta = request_bytes(listing_url, fetch["allowed_hosts"], settings)
    write_atomic(output_dir / "_listing.html", listing)

    report_pages: list[str] = []
    seen: set[str] = set()
    for url, text in extract_links(listing, listing_url):
        parsed = urlparse(url)
        is_report_title = text.lower().startswith(("report ", "bulletin ", "safety digest "))
        is_govuk_report = parsed.hostname == "www.gov.uk" and (
            parsed.path.startswith("/raib-reports/")
            or "/government/news/report-" in parsed.path
            or "/government/publications/" in parsed.path
        )
        if is_report_title and is_govuk_report and url not in seen:
            seen.add(url)
            report_pages.append(url)

    limit = max_items_override or int(fetch.get("max_items", 10))
    receipts: list[dict[str, Any]] = []
    for index, page_url in enumerate(report_pages[:limit], start=1):
        page, page_meta = request_bytes(page_url, fetch["allowed_hosts"], settings)
        pdf_links = [
            url
            for url, _ in extract_links(page, page_url)
            if urlparse(url).path.lower().endswith(".pdf")
            and (urlparse(url).hostname or "").lower() in set(fetch["allowed_hosts"])
        ]
        slug = safe_filename(Path(urlparse(page_url).path).name)
        if not pdf_links:
            page_path = output_dir / f"{index:02d}_{slug}.html"
            write_atomic(page_path, page)
            receipt = make_receipt(source, page_path, page, page_meta)
            receipt.update({"status": "page_only", "report_page": page_url})
            receipts.append(receipt)
            continue

        pdf_url = pdf_links[0]
        filename = f"{index:02d}_{slug}.pdf"
        receipt = download_one(source, pdf_url, filename, output_dir, manifest, force)
        receipt["report_page"] = page_url
        receipts.append(receipt)

    if not report_pages:
        raise DownloadError("RAIB列表页未发现报告链接；网页结构可能已变化。")
    if len(receipts) < limit:
        receipts.append(
            {
                "source_id": source["id"],
                "status": "warning",
                "message": f"仅发现{len(receipts)}份报告，目标为{limit}份。",
                "listing_url": listing_meta.get("final_url", listing_url),
                "checked_at": utc_now(),
            }
        )
    return receipts


def output_group(source: dict[str, Any]) -> str:
    return "open" if source["repository_policy"] in OPEN_REPOSITORY_POLICIES else "restricted"


def selected_sources(
    sources: list[dict[str, Any]],
    only: set[str],
    open_only: bool,
    include_disabled: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for source in sources:
        if only and source["id"] not in only:
            continue
        if open_only and output_group(source) != "open":
            continue
        if not source["fetch"].get("enabled", True) and not include_disabled:
            continue
        selected.append(source)
    unknown = only - {source["id"] for source in sources}
    if unknown:
        raise ManifestError(f"未知source id: {', '.join(sorted(unknown))}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", action="append", default=[], metavar="SOURCE_ID")
    parser.add_argument("--open-only", action="store_true", help="只下载明确开放许可的来源。")
    parser.add_argument("--include-disabled", action="store_true", help="包括默认关闭的大文件。")
    parser.add_argument("--force", action="store_true", help="覆盖已有文件。")
    parser.add_argument("--dry-run", action="store_true", help="打印下载计划但不联网。")
    parser.add_argument("--validate-only", action="store_true", help="只校验manifest。")
    parser.add_argument("--list", action="store_true", help="列出来源。")
    parser.add_argument("--max-collection-items", type=int, help="覆盖动态集合的最大下载数。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    sources = validate_manifest(manifest)

    if args.validate_only:
        print(f"manifest有效：{len(sources)}个来源。")
        return 0

    selected = selected_sources(
        sources,
        set(args.only),
        args.open_only,
        args.include_disabled,
    )
    if args.list:
        for source in sources:
            enabled = source["fetch"].get("enabled", True)
            print(
                f"{source['id']:<44} group={output_group(source):<10} "
                f"enabled={str(enabled):<5} {source['title']}"
            )
        return 0

    project_root = manifest_path.parent
    default_output = project_root / str(manifest["defaults"].get("output_root", "data/raw"))
    output_root = (args.output_dir or default_output).expanduser().resolve()

    if args.dry_run:
        print(f"output_root={output_root}")
        for source in selected:
            print(
                f"PLAN {source['id']} kind={source['fetch']['kind']} "
                f"group={output_group(source)} policy={source['repository_policy']}"
            )
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = project_root / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for source in selected:
        source_dir = output_root / output_group(source) / source["id"]
        source_dir.mkdir(parents=True, exist_ok=True)
        fetch = source["fetch"]
        print(f"[{source['id']}] {source['title']}")
        try:
            if fetch["kind"] == "govuk_raib_collection":
                source_receipts = download_raib_collection(
                    source,
                    source_dir,
                    manifest,
                    args.force,
                    args.max_collection_items,
                )
            elif fetch["kind"] == "webpage_bundle":
                urls = fetch["urls"]
                filenames = fetch.get("filenames", [])
                if len(urls) != len(filenames):
                    raise ManifestError(f"{source['id']}: urls和filenames长度不一致。")
                source_receipts = [
                    download_one(source, url, filename, source_dir, manifest, args.force)
                    for url, filename in zip(urls, filenames)
                ]
            else:
                source_receipts = [
                    download_one(
                        source,
                        fetch["url"],
                        fetch["filename"],
                        source_dir,
                        manifest,
                        args.force,
                    )
                ]
            receipts.extend(source_receipts)
            for receipt in source_receipts:
                print(f"  {receipt['status']}: {receipt.get('local_path', receipt.get('message', ''))}")
        except (DownloadError, ManifestError, OSError) as exc:
            failure = {
                "source_id": source["id"],
                "status": "failed",
                "error": str(exc),
                "failed_at": utc_now(),
            }
            failures.append(failure)
            print(f"  failed: {exc}", file=sys.stderr)

    corpus_index = {
        "manifest_version": manifest["manifest_version"],
        "manifest_path": str(manifest_path),
        "generated_at": utc_now(),
        "successful_records": receipts,
        "failures": failures,
    }
    index_path = output_root / "_corpus_index.json"
    write_atomic(index_path, json.dumps(corpus_index, ensure_ascii=False, indent=2).encode("utf-8"))

    log_path = log_dir / "download_report.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        for record in [*receipts, *failures]:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"完成：{len(receipts)}条成功记录，{len(failures)}个来源失败。")
    print(f"语料索引：{index_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
