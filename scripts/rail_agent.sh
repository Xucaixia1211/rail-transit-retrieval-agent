#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

if ! "${VENV_DIR}/bin/python" -c "import importlib.util, sys; sys.exit(any(importlib.util.find_spec(name) is None for name in ('yaml', 'numpy', 'rank_bm25', 'jieba', 'sentence_transformers', 'openai')))" >/dev/null 2>&1; then
  "${VENV_DIR}/bin/python" -m pip install \
    --disable-pip-version-check \
    -r "${PROJECT_ROOT}/requirements.txt"
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${PROJECT_ROOT}/data/cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
export HF_HUB_DISABLE_TELEMETRY="1"

# Once both configured models have been loaded successfully, stay offline on
# subsequent runs. This avoids slow version checks and makes demos reproducible.
if [[ -f "${PROJECT_ROOT}/data/cache/.models_ready" ]]; then
  export HF_HUB_OFFLINE="1"
  export TRANSFORMERS_OFFLINE="1"
fi

exec "${VENV_DIR}/bin/python" -m rail_agent.cli "$@"
