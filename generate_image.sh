#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -lt 1 ]; then
  echo "Usage: ./generate_image.sh \"your prompt\" [extra generate.py args]" >&2
  echo "Example: ./generate_image.sh \"A cinematic portrait of an astronaut in snowfall\"" >&2
  exit 1
fi

PROMPT="$1"
shift

source "$ROOT_DIR/.venv/bin/activate"

python "$ROOT_DIR/generate.py" \
  --local-files-only \
  --prompt "$PROMPT" \
  "$@"
