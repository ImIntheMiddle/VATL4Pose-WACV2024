#!/usr/bin/env bash
# Stop Label Studio by PID: the process of launch.sh and the processes of its session.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ ! -f "$LS_PID_FILE" ]]; then
  echo "no pid file $LS_PID_FILE; Label Studio was not started by launch.sh"
  exit 0
fi
PID="$(cat "$LS_PID_FILE")"
if ! grep -q label-studio "/proc/$PID/cmdline" 2>/dev/null; then
  echo "pid $PID is not a Label Studio process; removing the stale pid file"
  rm "$LS_PID_FILE"
  exit 0
fi
# launch.sh starts it with setsid, so the session id is the pid; list the members by pid
PIDS="$(ps -o pid= -s "$PID" | tr -s ' \n' ' ' || true)"
echo "stopping pids: $PIDS"
kill $PIDS
for _ in $(seq 1 30); do
  kill -0 "$PID" 2>/dev/null || break
  sleep 1
done
REST="$(ps -o pid= -s "$PID" | tr -s ' \n' ' ' || true)"
if [[ -n "${REST// /}" ]]; then
  echo "still running: $REST; sending SIGKILL"
  kill -9 $REST
fi
rm "$LS_PID_FILE"
echo "stopped"
