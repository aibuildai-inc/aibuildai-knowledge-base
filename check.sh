#!/usr/bin/env bash
# The machine gate for the Kb service and its offline producers.
set -euo pipefail
cd "$(dirname "$0")"

PYBIN="$(dirname "$(command -v python3 || command -v python)")"
export PATH="${PYBIN}:${PATH}"
for tool in pyright ruff; do
    command -v "${tool}" >/dev/null || {
        echo "${tool} not found in ${PYBIN} or PATH." >&2
        exit 1
    }
done

echo "pyright"
pyright

echo "ruff"
ruff check --no-cache --select F,B904,S110,PGH003 aibuildai_mcp sourcing

echo "ruff (parameter annotations)"
ruff check --no-cache --select ANN001,ANN002,ANN003 aibuildai_mcp sourcing

echo "clean"
