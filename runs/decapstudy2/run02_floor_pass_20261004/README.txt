RUN decapstudy2/run02_floor_pass_20261004  [finished]
Floor pass: full-budget numpy PSO runs for the (key, band) floors Exp 1 cannot supply (B1: mer2, pll1v0, mphyvdd, ddr21 x10 runs; ddr_full x6), no early stop.

Start: 2026-10-04 17:08    End: 2026-10-04 19:36    Host: eeiitj123.iitj.ac.in
Launch: cd DecapStudy2 && STUDY_THREADS=32 nohup python -u floor_pass.py > logs/floor_pass.out
Environment: conda env work; BLAS pinned to 1 thread; STUDY_THREADS=32
Code (exact versions that ran, from git): ed9b2a1
  code/README.md  md5 26deee44c27cab7c849a29527db4a0b7
  code/analyze.py  md5 4326e1977cc316346a9fb1a31c159d66
  code/check_cost_agreement.py  md5 413b71402f5af49f3ff309b364a1d270
  code/derive_targets.py  md5 b7cf4f0913318aca71f023ffd8b66716
  code/experiments.py  md5 2065cfa5e66d953274ea06cec9bb588a
  code/floor_pass.py  md5 5eab7e28009dc86a6bebac755dd33a49
  code/run_all.py  md5 d1abd2ac1d537d66db16076e45fc7550
  code/run_exp2.sh  md5 30adbfb09dae0f0cfe6e52bd4480d57e
  code/study_config.py  md5 131602b5fe0a37879b11c0e4f1a0200b
  code/study_core.py  md5 d909de7623f7686a8b1ae72ea539b8a2
  code/validate.py  md5 0d9e2d15c19266f83edccb0cc66b7945

Layout: code/ = copies of the code that ran; output/ = raw results (git-ignored); logs/ = console/study logs (git-ignored); summary/ and plots/ = derived tables and figures (tracked).
This run predates the runs/ convention: it was retrofitted from its original location after it finished; the originals were copied, not moved.

Notes:
  - The same file is tracked at DecapStudy2/floor_pass/floor_runs.jsonl (input to derive_targets.py).
  - Needed because Exp 1's B1 runs stopped after 1-7 capacitors on its old target, so they are not floors; B2/B3 floors come from Exp 1 itself.
