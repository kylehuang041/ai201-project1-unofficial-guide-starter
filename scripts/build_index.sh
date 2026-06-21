#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv. Create it with: python -m venv .venv" >&2
  exit 1
fi

source ".venv/bin/activate"
python -m src.build_index
