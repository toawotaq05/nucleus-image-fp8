#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$ROOT_DIR/.venv/bin/activate"

EXTRA_ARGS=("$@")

has_flag() {
  local flag="$1"
  shift
  local arg
  for arg in "$@"; do
    if [ "$arg" = "$flag" ]; then
      return 0
    fi
  done
  return 1
}

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d '[:space:]')"
else
  GPU_COUNT=0
fi

DEFAULT_ARGS=(--local-files-only)

if [ "${GPU_COUNT:-0}" -ge 2 ]; then
  if ! has_flag --max-gpu0 "${EXTRA_ARGS[@]}"; then
    DEFAULT_ARGS+=(--max-gpu0 10GiB)
  fi
  if ! has_flag --max-gpu1 "${EXTRA_ARGS[@]}"; then
    DEFAULT_ARGS+=(--max-gpu1 15GiB)
  fi
fi

python "$ROOT_DIR/interactive_generate.py" \
  "${DEFAULT_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
