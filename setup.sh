#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -d .venv ] && [ ! -f .venv/.standalone_nucleus_env ]; then
  backup=".venv.backup.$(date +%Y%m%d-%H%M%S)"
  mv .venv "$backup"
  printf 'Existing .venv moved to %s\n' "$backup"
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
python -m pip install -r requirements.txt
touch .venv/.standalone_nucleus_env

printf '\nSetup complete. Activate with:\n'
printf 'source %s/.venv/bin/activate\n' "$(pwd)"
