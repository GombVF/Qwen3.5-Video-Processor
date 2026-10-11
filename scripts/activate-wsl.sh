#!/usr/bin/env bash
# activate-wsl.sh — Convenience script to activate the WSL environment.
#
# Usage (must be sourced, not executed):
#   source scripts/activate-wsl.sh
#
# Override the virtual environment location:
#   VENV_DIR=/path/to/venv source scripts/activate-wsl.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

_PROJECT_KEY="$(printf '%s' "$(basename "$PROJECT_ROOT")" | tr -c 'A-Za-z0-9._-' '_')"
_DEFAULT_VENV="${VENV_DIR:-$HOME/.venvs/$_PROJECT_KEY/.venv-wsl}"

if [[ ! -x "$_DEFAULT_VENV/bin/python" ]]; then
    echo "[ERROR] Virtual environment not found at $_DEFAULT_VENV" >&2
    echo "[ERROR] Run: bash scripts/setup-wsl.sh" >&2
else
    cd "$PROJECT_ROOT" || return 1
    # shellcheck disable=SC1091
    source "$_DEFAULT_VENV/bin/activate"
    if [[ -f "$PROJECT_ROOT/.env-wsl" ]]; then
        set -a
        # shellcheck disable=SC1091
        source "$PROJECT_ROOT/.env-wsl"
        set +a
    fi
    echo "[OK] Activated $_DEFAULT_VENV"
    echo "[OK] Working directory: $PROJECT_ROOT"
fi

unset _PROJECT_KEY _DEFAULT_VENV
