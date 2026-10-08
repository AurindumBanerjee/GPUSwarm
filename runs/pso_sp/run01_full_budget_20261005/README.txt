RUN pso_sp/run01_full_budget_20261005  [finished]
PSO_SP full-budget runs (no early stop) on MPHY (6 problems) and VDDQ DDR3 (2 problems), numpy only, 3 runs each; lowest impedance each network reaches.

Start: 2026-10-05 01:12    End: 2026-10-05 16:45    Host: eeiitj123.iitj.ac.in
Launch: cd PSO_SP && TIME_LIMIT_S=86400 NUM_RUNS=3 STUDY_THREADS=16 nohup python -u pso_mphy.py > mphy.out & TIME_LIMIT_S=86400 NUM_RUNS=3 STUDY_THREADS=16 nohup python -u pso_vddq_ddr3.py > ddr3.out &
Environment: conda env work; BLAS pinned to 1 thread; 16 particle threads per job, two jobs in parallel; numpy 2.2.5, MKL 2023.1; timer perf_counter
Code (exact versions that ran, from git): a043279
  code/README.md  md5 2e77eadc3d46a7964f28e2473070a75f
  code/pso_mphy.py  md5 fa1903e37d1284f672fcc6a4b702ea02
  code/pso_sp_core.py  md5 d04dc385244623aafa4af6b712612720
  code/pso_vddq_ddr3.py  md5 fce7000bd1ecbcaf061bfe03f7323df6

Layout: code/ = copies of the code that ran; output/ = raw results (git-ignored); logs/ = console/study logs (git-ignored); summary/ and plots/ = derived tables and figures (tracked).
This run predates the runs/ convention: it was retrofitted from its original location after it finished; the originals were copied, not moved.

Notes:
  - Code is the committed version a043279; the working copy later gained a --thresholds mode and jitter 0 (commit 6b369c7) -- those are NOT in this run.
  - This run used WARM_START_JITTER = 0.02 (the then-default).
  - No run hit the 86400 s cap (timed_out false for all 24).
