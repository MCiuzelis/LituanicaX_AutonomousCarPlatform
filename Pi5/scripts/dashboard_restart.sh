#!/usr/bin/env bash
# (Re)start dashboard.py detached, waiting for the old instance to release the camera/GPIO.
# Log: logs/dashboard.log
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
mkdir -p logs

pkill -f "python3 dashboard.py"
for _ in $(seq 1 40); do pgrep -f "python3 dashboard.py" >/dev/null || break; sleep 0.25; done
pkill -9 -f "python3 dashboard.py"; sleep 1

[ -s logs/dashboard.log ] && mv -f logs/dashboard.log logs/dashboard.log.1   # keep the previous run's log
nohup python3 dashboard.py > logs/dashboard.log 2>&1 &
PID=$!
for _ in $(seq 1 60); do
  curl -sf -o /dev/null localhost:8080/api/telemetry && { echo "dashboard up (pid $PID) -> http://$(hostname):8080"; exit 0; }
  sleep 0.5
done
echo "dashboard failed to start:"; grep -a -E "Error|Traceback" logs/dashboard.log | tail -3; exit 1
