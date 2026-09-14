#!/usr/bin/env python3
"""Build a size-bounded, allowlisted release ZIP without restricted corpus content."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT_FILES = {
    ".gitignore",
    "README.md",
    "requirements.txt",
    "source_manifest.yaml",
}
ROOT_DIRECTORIES = {"configs", "docs", "evaluation", "scripts", "src", "tests"}
EXCLUDED_NAMES = {".DS_Store", "__pycache__"}
EXCLUDED_SUFFIXES = {".pyc", ".part"}


def include_file(project_root: Path, path: Path) -> bool:
    relative = path.relative_to(project_root)
    if any(part in EXCLUDED_NAMES for part in relative.parts) or path.suffix in EXCLUDED_SUFFIXES:
        return False
    if len(relative.parts) == 1:
        return relative.name in ROOT_FILES
    if relative.parts[0] not in ROOT_DIRECTORIES:
        return False
    if relative.as_posix() == "evaluation/question_specs.yaml":
        return True
    return True


def create_demo_chunks(project_root: Path, destination: Path, count: int = 12) -> int:
    chunks_path = project_root / "data" / "processed" / "chunks.jsonl"
    if not chunks_path.exists():
        return 0
    selected = []
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        chunk = json.loads(line)
        if chunk["source_id"] != "ch_openrail_rcm_dx":
            continue
        selected.append(chunk)
        if len(selected) == count:
            break
    if not selected:
        return 0
    demo_dir = destination / "demo_data"
    demo_dir.mkdir(parents=True, exist_ok=True)
    with (demo_dir / "chunks.sample.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in selected:
            handle.write(json.dumps(chunk, ensure_ascii=False, separators=(",", ":")) + "\n")
    (demo_dir / "README.md").write_text(
        "# Demo data\n\n"
        "These sample chunks are derived from the OpenRail Association RCM-DX specification "
        "under the Eclipse Public License 2.0. Attribution: OpenRail Association RCM-DX "
        "contributors. Source: https://github.com/OpenRailAssociation/rcm-dx\n",
        encoding="utf-8",
    )
    return len(selected)


def ensure_safe_archive(zip_path: Path, max_size_mb: int) -> None:
    forbidden = ("data/raw/", "data/processed/", "data/cache/", ".venv/", "artifacts/")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        bad = [name for name in names if any(marker in name for marker in forbidden)]
        if bad:
            raise RuntimeError(f"forbidden paths found in release: {bad[:5]}")
    if zip_path.stat().st_size > max_size_mb * 1024 * 1024:
        raise RuntimeError(f"release exceeds {max_size_mb} MB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-size-mb", type=int, default=100)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    archive_root_name = "rail-transit-retrieval-agent"

    with tempfile.TemporaryDirectory(prefix="rail_agent_release_") as temporary_name:
        staging_root = Path(temporary_name) / archive_root_name
        staging_root.mkdir(parents=True)
        for path in sorted(project_root.rglob("*")):
            if not path.is_file() or not include_file(project_root, path):
                continue
            relative = path.relative_to(project_root)
            destination = staging_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        demo_count = create_demo_chunks(project_root, staging_root)
        temporary_zip = output.with_suffix(output.suffix + ".part")
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(staging_root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging_root.parent))
        temporary_zip.replace(output)

    ensure_safe_archive(output, args.max_size_mb)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    with zipfile.ZipFile(output) as archive:
        file_count = len([info for info in archive.infolist() if not info.is_dir()])
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": output.stat().st_size,
                "sha256": digest,
                "files": file_count,
                "demo_chunks": demo_count,
                "max_size_mb": args.max_size_mb,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
