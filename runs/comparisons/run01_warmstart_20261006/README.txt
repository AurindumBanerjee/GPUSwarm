RUN comparisons/run01_warmstart_20261006  [finished]
Warm-start comparison on the 21-port benchmark PDN: naive (40% verbatim) vs current (FIX 11 elite + jitter 0.02) at SB.py thresholds (10 runs), and none/naive/current at full budget (5 runs).

Start: 2026-10-06 20:33    End: 2026-10-06 21:08 (last result file written)    Host: eeiitj123.iitj.ac.in
Launch: cd Comparisons && THREADS=16 NUM_RUNS=10 python -u warmstart_thresholds.py > thr.out ; THREADS=16 NUM_RUNS=5 python -u warmstart_arms.py > arms.out  (run in parallel, 16 threads each)
Environment: conda env work; BLAS pinned to 1 thread; 16 particle threads per script
Code (exact versions that ran, from git): 130fd29
  code/README.md  md5 5dd833fa7a605bef00b98598970bc9f3
  code/warmstart_arms.py  md5 a73fd4545c23590f18369bc383e85d36
  code/warmstart_thresholds.py  md5 2e7c7c4bab5d44e15efa0cdc44082bfc

Layout: code/ = copies of the code that ran; output/ = raw results (git-ignored); logs/ = console/study logs (git-ignored); summary/ and plots/ = derived tables and figures (tracked).
This run predates the runs/ convention: it was retrofitted from its original location after it finished; the originals were copied, not moved.

Notes:
  - Code is the committed version 130fd29, before the frozen-stage tolerance fix (commit 6b369c7): the 'frozen stages 0/N' counts in the tables are an artefact of the == 0.0 test; the printed spread (4.1e-17 vs 1.9e-2) is the measurement.
  - Result: naive beat current in 5/5 paired full-budget runs (median -14%); at threshold 0.05 naive 9.1 s vs current 19.1 s.
