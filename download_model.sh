#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p models/Nucleus-Image-FP8

if ! command -v hf >/dev/null 2>&1; then
  echo "'hf' CLI not found. Activate .venv first." >&2
  exit 1
fi

hf download \
  D-Squarius-Green-Jr/Nucleus-Image-FP8 \
  --local-dir ./models/Nucleus-Image-FP8 \
  --resume-download
