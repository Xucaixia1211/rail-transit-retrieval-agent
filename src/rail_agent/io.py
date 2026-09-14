from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Iterable

import yaml


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{number}: {exc}") from exc
    return records


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".part", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".part", delete=False) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)
