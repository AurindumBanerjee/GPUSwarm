#!/bin/bash
# Wait until no other process uses the GPU (and stays idle), then start the GPUAll run (resumable).
#   nohup setsid bash GPUAll/run_when_gpu_free.sh [args for run_gpuall.sh] > GPUAll/waiter.out 2>&1 < /dev/null &
# Env: IDLE_POLLS=6 (consecutive free polls, 30 s apart), GPU_MEM_FREE_MB=500 (max used memory counted as free).
G="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N=${IDLE_POLLS:-6}; MAXMB=${GPU_MEM_FREE_MB:-500}; ok=0
echo "[$(date +%T)] waiting for a free GPU"
while [ $ok -lt $N ]; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$apps" = 0 ] && [ "$used" -le "$MAXMB" ]; then ok=$((ok+1)); else ok=0; fi
  sleep 30
done
echo "[$(date +%T)] GPU free; starting"
exec bash "$G/run_gpuall.sh" "$@"
