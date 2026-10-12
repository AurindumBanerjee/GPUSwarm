#!/bin/bash
# Replays the whole sweep from the code copy in code/. The scripts write next to themselves, so they run in a scratch directory and
# the results are collected into a NEW folder (output/rerun_<timestamp>/{json,plots}); the original output/ and plots/ are never touched.
# Needs: conda env "work" (numpy, scipy, matplotlib) or any python with them; data at <repo>/Data (override: PDN_DATA).
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${GPUSWARM_ROOT:-$RUN/../../..}"
OUT="${RERUN_OUT:-$RUN/output/rerun_$(date +%Y%m%d_%H%M%S)}"
export PDN_DATA="${PDN_DATA:-$ROOT/Data}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
W="$(mktemp -d)"; cp "$RUN"/code/*.py "$W"/; mkdir -p "$OUT/order24" "$OUT/order20" "$OUT/order32"; cd "$W" || exit 1
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "START: $(stamp)  host: $(hostname)  out: $OUT" >> "$RUN/README.txt"
python decap_classify.py > "$OUT/decap_classify.log" 2>&1 && mv decap_classes.json "$OUT/"
fit() {  # board prob order outdir
  python pdn_poles.py --board $1 --prob $2 --order $3 --hi 1e8 --plot >> "$OUT/pdn_poles.log" 2>&1
  mv poles_$1_$2.json poles_$1_$2.png "$OUT/$4/"; }
for p in ddr21 ddr_full; do fit VDDQ_DDR3 $p 24 order24; done
for p in mer1 mer2 pll1v0 pll1v8 mphyvdd mphy_full; do fit MPHY $p 24 order24; done
fit VDDQ_DDR3 ddr21 20 order20
for spec in "VDDQ_DDR3 ddr21" "MPHY mer1" "MPHY mer2" "MPHY pll1v0" "MPHY mphyvdd"; do set -- $spec; fit $1 $2 32 order32; done
echo "END:   $(stamp)  exit code $?" >> "$RUN/README.txt"; rm -rf "$W"
