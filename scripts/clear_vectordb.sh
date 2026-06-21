#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d "chroma_db" ]]; then
  rm -rf "chroma_db"
  echo "Cleared chroma_db/."
else
  echo "No chroma_db/ directory found."
fi
