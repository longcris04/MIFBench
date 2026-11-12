#!/usr/bin/env bash
# Start the review server:   review_server/run.sh [--port 8777] [--host 127.0.0.1]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
exec "$MIF_PYTHON" review_server/server.py "$@"
