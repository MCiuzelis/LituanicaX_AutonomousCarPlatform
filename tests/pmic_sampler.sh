#!/usr/bin/env bash
# Shared by stress_test.sh and soak_test.sh: background PMIC/thermal sampler at 1 Hz -> CSV.
# (The Python equivalent is hardware.PowerLogger.)
#
#   source "$(dirname "$0")/pmic_sampler.sh"
#   start_sampler results/power_x.csv
#   ...workload...
#   stop_sampler
#   print_stats results/power_x.csv
#
# CSV columns: elapsed,core_i_A,core_v_V,core_w_W,sys3v3_i_A,sys1v1_i_A,temp_C,arm_freq_MHz,throttled

SAMPLER_PID=""

start_sampler() {
  local out="$1"
  echo "elapsed,core_i_A,core_v_V,core_w_W,sys3v3_i_A,sys1v1_i_A,temp_C,arm_freq_MHz,throttled" > "$out"
  local t0
  t0=$(date +%s)
  {
    while :; do
      local pmic ci cv s3 v1 tmp frq thr w
      pmic=$(vcgencmd pmic_read_adc)
      ci=$(echo "$pmic" | grep -oP 'VDD_CORE_A current\(\d+\)=\K[0-9.]+' | head -1)
      cv=$(echo "$pmic" | grep -oP 'VDD_CORE_V volt\(\d+\)=\K[0-9.]+' | head -1)
      s3=$(echo "$pmic" | grep -oP '3V3_SYS_A current\(\d+\)=\K[0-9.]+' | head -1)
      v1=$(echo "$pmic" | grep -oP '1V1_SYS_A current\(\d+\)=\K[0-9.]+' | head -1)
      tmp=$(vcgencmd measure_temp | sed -E 's/temp=([0-9.]+).*/\1/')
      frq=$(vcgencmd measure_clock arm | sed -E 's/.*=([0-9]+)/\1/')
      thr=$(vcgencmd get_throttled | sed -E 's/throttled=//')
      w=$(awk "BEGIN{print $ci*$cv}")
      echo "$(($(date +%s)-t0)),$ci,$cv,$w,$s3,$v1,$tmp,$((frq/1000000)),$thr" >> "$out"
      sleep 1
    done
  } >/dev/null 2>&1 &
  SAMPLER_PID=$!
  disown "$SAMPLER_PID" 2>/dev/null
}

stop_sampler() {
  [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null
  SAMPLER_PID=""
}

# avg/min/max of core power, temperature and ARM clock, plus throttled sample count
print_stats() {
  local f="$1"
  [ -s "$f" ] || { echo "  (no samples)"; return; }
  awk -F, 'NR>1 {
    w=$4+0; t=$7+0; m=$8+0; n++;
    sw+=w; if(n==1||w<minw)minw=w; if(w>maxw)maxw=w;
    if(n==1||t<minT)minT=t; if(t>maxT)maxT=t;
    if(n==1||m<minM)minM=m; if(m>maxM)maxM=m;
    if($9!="0x0")thr++;
  } END{
    if(n==0){print "  (no samples)"; exit}
    printf "  samples=%d core_w(avg=%.2f min=%.2f max=%.2f W) temp(min=%.1f max=%.1f C) armMHz(min=%d max=%d) throttled_samples=%d\n", n, sw/n, minw, maxw, minT, maxT, minM, maxM, thr+0
  }' "$f"
}
