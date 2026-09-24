#!/usr/bin/env bash
# Pi 5 + Hailo-8 soak: DUR seconds of 4x CPU stress-ng + back-to-back Hailo benchmarks,
# with the PMIC sampler running. Outputs: results/soak/{power_long.csv,cpu_long.log,ai_long.log}
#   tests/soak_test.sh [seconds]     (default 300)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RES="$ROOT/results/soak"
HEF=/usr/share/hailo-models/yolov8s_h8.hef
DUR="${1:-300}"
CSV="$RES/power_long.csv"
mkdir -p "$RES"
[ -f "$HEF" ] || { echo "no model: $HEF"; exit 1; }
# shellcheck source=pmic_sampler.sh
source "$ROOT/tests/pmic_sampler.sh"

start_sampler "$CSV"
echo "[$(date +%T)] phase: concurrent Hailo + 4xCPU soak (${DUR}s)"
(
  timeout --kill-after=6 $((DUR+10)) stress-ng -c 4 --cpu-method matrixprod --timeout "${DUR}s" --metrics-brief > "$RES/cpu_long.log" 2>&1
  echo "cpu done rc=$?"
) &
SP=$!
(
  end=$((SECONDS+DUR))
  while [ $SECONDS -lt $end ]; do
    timeout --kill-after=6 40 hailortcli benchmark "$HEF" >> "$RES/ai_long.log" 2>&1
  done
  echo "ai done"
) &
AP=$!
wait "$SP" 2>/dev/null; wait "$AP" 2>/dev/null
stop_sampler
sync

echo "[$(date +%T)] soak finished"
echo "throttle end:  $(vcgencmd get_throttled)"
echo "temp end:      $(vcgencmd measure_temp)"
echo
echo "==================== SOAK SUMMARY ==================="
print_stats "$CSV"
echo "== CPU (stress-ng, ${DUR}s) =="; grep -P "^\s*cpu " "$RES/cpu_long.log"
echo "== Hailo FPS (last benchmark run's HW-only) =="; grep -oP 'FPS: \K[0-9.]+' "$RES/ai_long.log" | tail -3
echo "== benchmark runs completed =="
echo "  $(grep -c "Starting Measurements" "$RES/ai_long.log") benchmark iterations in ${DUR}s"
echo "==================== END ==================="
