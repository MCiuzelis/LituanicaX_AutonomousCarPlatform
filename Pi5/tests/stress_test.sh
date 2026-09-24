#!/usr/bin/env bash
# Pi 5 + Hailo-8 stress test (~90 s) with PMIC power monitoring:
#   idle -> 4x CPU stress-ng -> Hailo yolov8s benchmark -> both together -> Hailo measure-power
# Outputs: results/power_{idle,cpu,ai,both,sensor}.csv + the stress-ng / hailortcli logs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RES="$ROOT/results"
HEF=/usr/share/hailo-models/yolov8s_h8.hef
mkdir -p "$RES"
[ -f "$HEF" ] || { echo "no model: $HEF"; exit 1; }
# shellcheck source=pmic_sampler.sh
source "$ROOT/tests/pmic_sampler.sh"

echo "[$(date +%T)] phase: idle 8s"
start_sampler "$RES/power_idle.csv"; sleep 8; stop_sampler

echo "[$(date +%T)] phase: CPU-only stress-ng x4 (25s)"
start_sampler "$RES/power_cpu.csv"
timeout --kill-after=6 25 stress-ng -c 4 --cpu-method matrixprod --metrics-brief > "$RES/cpu_metrics.log" 2>&1
stop_sampler

echo "[$(date +%T)] phase: Hailo-8-only yolov8s inference (22s)"
start_sampler "$RES/power_ai.csv"
timeout --kill-after=6 22 hailortcli benchmark "$HEF" > "$RES/ai_benchmark.log" 2>&1
echo "benchmark rc=$?"
stop_sampler

echo "[$(date +%T)] phase: CPU + Hailo together (20s)"
start_sampler "$RES/power_both.csv"
timeout --kill-after=6 20 stress-ng -c 4 --cpu-method matrixprod --metrics-brief > "$RES/cpu_metrics_both.log" 2>&1 &
SP=$!
timeout --kill-after=6 20 hailortcli benchmark "$HEF" > "$RES/ai_benchmark_both.log" 2>&1 &
AP=$!
wait "$SP" 2>/dev/null; wait "$AP" 2>/dev/null
stop_sampler

echo "[$(date +%T)] phase: Hailo power sensor (idle, 12s; unsupported on the AI HAT -> rc != 0 expected)"
start_sampler "$RES/power_sensor.csv"
timeout --kill-after=6 15 hailortcli measure-power --duration 12 > "$RES/ai_measure_power.log" 2>&1
echo "measure-power rc=$?"
stop_sampler

echo
echo "==================== SUMMARY ===================="
echo "throttle now:  $(vcgencmd get_throttled)"
for phase in idle cpu ai both; do
  printf "[%s]\n" "$phase"; print_stats "$RES/power_$phase.csv"
done
echo "== CPU metrics (solo) =="; grep -E "successful|bogo ops" "$RES/cpu_metrics.log" | head -6
echo "== CPU metrics (with AI) =="; grep -E "successful|bogo ops" "$RES/cpu_metrics_both.log" | head -6
echo "== Hailo benchmark (solo) =="; grep -iE "fps|throughput" "$RES/ai_benchmark.log" | head -6
echo "== Hailo benchmark (with CPU) =="; grep -iE "fps|throughput" "$RES/ai_benchmark_both.log" | head -6
echo "== Hailo measure-power =="; tail -8 "$RES/ai_measure_power.log"
echo "==================== END ===================="
