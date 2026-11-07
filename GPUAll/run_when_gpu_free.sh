#!/bin/bash
# Watch the GPU and start the GPUAll run (resumable) once another job has used it and then left it idle.
#   nohup setsid bash GPUAll/run_when_gpu_free.sh [args for run_gpuall.sh] > GPUAll/waiter.out 2>&1 < /dev/null &
# Phase 1 (WAIT_FOR_BUSY=1): wait until the GPU is busy (the experiment starts). Without it, an idle GPU would launch us at once.
#                            Gives up (no launch) after MAX_WAIT_BUSY_H hours without seeing the GPU busy.
# Phase 2: wait until the GPU has been idle for IDLE_SECS in a row, then exec run_gpuall.sh.
#   (desktop sessions of other users hold ~0.9 GiB of graphics memory on this GPU, so the memory limit must sit above that)
#   idle = no compute process, utilisation <= UTIL_MAX %, memory used <= MEM_FREE_MB.   A single busy sample restarts the window.
# Env: WAIT_FOR_BUSY=1  MAX_WAIT_BUSY_H=72  IDLE_SECS=1800  POLL_S=10  UTIL_MAX=5  GPU_MEM_FREE_MB=2000
G="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAIT_BUSY=${WAIT_FOR_BUSY:-1}; MAXH=${MAX_WAIT_BUSY_H:-72}; IDLE=${IDLE_SECS:-1800}; POLL=${POLL_S:-10}
UMAX=${UTIL_MAX:-5}; MMAX=${GPU_MEM_FREE_MB:-2000}
ts() { date '+%F %T'; }
sample() {                     # sets apps util used ; busy=0/1
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  read -r util used < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ',')
  util=${util:-100}; used=${used:-99999}
  if [ "$apps" = 0 ] && [ "$util" -le "$UMAX" ] && [ "$used" -le "$MMAX" ]; then busy=0; else busy=1; fi
}
echo "[$(ts)] watcher up: wait_for_busy=$WAIT_BUSY idle_window=${IDLE}s poll=${POLL}s idle<=util ${UMAX}% mem ${MMAX}MiB; args: $*"
if [ "$WAIT_BUSY" = 1 ]; then
  t0=$(date +%s); seen=0
  while [ $seen -lt 3 ]; do                      # 3 busy samples in a row = experiment started
    sample; if [ "$busy" = 1 ]; then seen=$((seen+1)); else seen=0; fi
    if [ $(( $(date +%s) - t0 )) -gt $((MAXH * 3600)) ]; then echo "[$(ts)] GPU never became busy in ${MAXH} h; exiting without launching"; exit 0; fi
    sleep "$POLL"
  done
  echo "[$(ts)] GPU busy (apps=$apps util=${util}% mem=${used}MiB): experiment started; waiting for it to end"
fi
idle_since=""
while true; do
  sample
  now=$(date +%s)
  if [ "$busy" = 0 ]; then
    [ -z "$idle_since" ] && { idle_since=$now; echo "[$(ts)] GPU idle; confirming for ${IDLE}s"; }
    [ $((now - idle_since)) -ge "$IDLE" ] && break
  else
    [ -n "$idle_since" ] && echo "[$(ts)] busy again (apps=$apps util=${util}% mem=${used}MiB); window restarted"
    idle_since=""
  fi
  sleep "$POLL"
done
echo "[$(ts)] GPU idle for ${IDLE}s; starting run_gpuall.sh $*"
[ "${DRY:-0}" = 1 ] && { echo "[$(ts)] DRY=1: not launching"; exit 0; }
exec bash "$G/run_gpuall.sh" "$@"
