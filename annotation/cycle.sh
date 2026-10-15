#!/usr/bin/env bash
# Run cycle.py in annotation/.venv.
set -euo pipefail
TRACKING_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$TRACKING_ROOT/.venv/bin/python" "$TRACKING_ROOT/cycle.py" "$@"
