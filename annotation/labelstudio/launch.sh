#!/usr/bin/env bash
# Start Label Studio on localhost, create the project of the pending round and import its tasks.
# cycle.sh runs this as its "annotate" step. The settings come from common.sh.
#   launch.sh            the newest round that has queries but no answers yet
#   launch.sh 1          round 1
# Label Studio keeps running in the background; stop it with stop.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# the pending round
if [[ $# -ge 1 ]]; then
  ROUND=$(printf '%02d' "$((10#$1))")
else
  QUERIES_LATEST=$(ls "$ORACLE_DIR"/round_*_queries.json 2>/dev/null | sort | tail -1 || true)
  [[ -n "$QUERIES_LATEST" ]] || { echo "no round_XX_queries.json in $ORACLE_DIR; run the atl step of cycle.sh first" >&2; exit 1; }
  ROUND=$(basename "$QUERIES_LATEST" | sed -E 's/round_([0-9]+)_queries.json/\1/')
fi
QUERIES="$ORACLE_DIR/round_${ROUND}_queries.json"
[[ -f "$QUERIES" ]] || { echo "$QUERIES does not exist" >&2; exit 1; }
[[ ! -f "$ORACLE_DIR/round_${ROUND}_answers.json" ]] || echo "note: round $ROUND already has an answers file"

# Label Studio in its own environment (does not touch the lock of this repository)
if [[ ! -x "$LS_VENV/bin/label-studio" ]]; then
  echo "installing Label Studio into $LS_VENV (once, a few minutes)"
  uv venv --python 3.12 "$LS_VENV"
  uv pip install --python "$LS_VENV/bin/python" label-studio
fi
mkdir -p "$LS_DATA_DIR"

port_in_use() { ss -ltn "sport = :$LS_PORT" | grep -q LISTEN; }
own_pid() { [[ -f "$LS_PID_FILE" ]] && grep -q label-studio "/proc/$(cat "$LS_PID_FILE")/cmdline" 2>/dev/null; }

if own_pid; then
  echo "Label Studio is already running (pid $(cat "$LS_PID_FILE"))"
else
  if port_in_use; then
    echo "port $LS_PORT is in use by another process; choose another with LS_PORT=<port>" >&2
    exit 1
  fi
  DOC_ROOT="$(realpath "$DATA_ROOT")"
  echo "starting Label Studio on $LS_HOST:$LS_PORT (the first start builds the database and takes a few minutes)"
  LABEL_STUDIO_BASE_DATA_DIR="$LS_DATA_DIR" \
  LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
  LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$DOC_ROOT" \
  LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN=true \
  COLLECT_ANALYTICS=false LATEST_VERSION_CHECK=false \
  setsid nohup "$LS_VENV/bin/label-studio" start --no-browser --internal-host "$LS_HOST" -p "$LS_PORT" \
    --data-dir "$LS_DATA_DIR" --username "$LS_USER" --password "$LS_PASSWORD" --user-token "$LS_TOKEN" \
    > "$LS_LOG" 2>&1 < /dev/null &
  echo $! > "$LS_PID_FILE"
  for _ in $(seq 1 900); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "$LS_URL/health" || true)
    [[ "$code" == "200" ]] && break
    own_pid || { echo "Label Studio exited; see $LS_LOG" >&2; tail -20 "$LS_LOG" >&2; exit 1; }
    sleep 2
  done
  [[ "$code" == "200" ]] || { echo "Label Studio did not answer in 30 minutes; see $LS_LOG" >&2; exit 1; }
fi

# tasks and labeling config from the queries of the round, then the project through the API
TASKS="$LS_DATA_DIR/tasks_round_${ROUND}.json"
CONFIG="$LS_DATA_DIR/label_config.xml"
"$LS_VENV/bin/python" "$LS_DIR/queries_to_labelstudio.py" --queries "$QUERIES" --round "$((10#$ROUND))" \
  --skip-joints left_ear,right_ear --out-tasks "$TASKS" --out-config "$CONFIG"
"$LS_VENV/bin/python" "$LS_DIR/ls_setup.py" --url "$LS_URL" --token "$LS_TOKEN" \
  --title "vatl4pose demo round $ROUND" --config "$CONFIG" --tasks "$TASKS" --images-dir "$(realpath "$DATA_ROOT")/images"

echo
echo "open $LS_URL in the browser; log in with $LS_USER / $LS_PASSWORD"
echo "stop with: bash $ANNOTATION_DIR/cycle.sh <run folder> stop"
