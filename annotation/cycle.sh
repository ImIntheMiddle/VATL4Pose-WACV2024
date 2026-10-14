#!/usr/bin/env bash
# Use the tracking environment; Python owns argument validation and subprocess execution.
set -euo pipefail
TRACKING_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$TRACKING_ROOT/.venv/bin/python" "$TRACKING_ROOT/cycle.py" "$@"
