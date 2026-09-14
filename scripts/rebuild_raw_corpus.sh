#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

if ! "${VENV_DIR}/bin/python" -c "import yaml" >/dev/null 2>&1; then
  "${VENV_DIR}/bin/python" -m pip install \
    --disable-pip-version-check \
    -r "${PROJECT_ROOT}/requirements.txt"
fi

exec "${VENV_DIR}/bin/python" \
  "${PROJECT_ROOT}/scripts/download_corpus.py" \
  --manifest "${PROJECT_ROOT}/source_manifest.yaml" \
  "$@"
