#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT="${1:-${PROJECT_ROOT}/artifacts/rail-transit-retrieval-agent.zip}"

exec "${PROJECT_ROOT}/.venv/bin/python" \
  "${PROJECT_ROOT}/scripts/build_release_zip.py" \
  --project-root "${PROJECT_ROOT}" \
  --output "${OUTPUT}" \
  --max-size-mb 100
