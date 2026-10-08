RUN decapstudy/run01_exp1_20260930  [finished]
Exp 1 decap study: 5 scopes x 4 bands PSO grid (staged), Grid 2 inversion-method timing, consistency checks; assumed-voltage targets.

Start: 2026-09-30 01:06 (earliest result file)    End: 2026-10-04 14:43    Host: eeiitj123.iitj.ac.in
Launch: cd DecapStudy && STUDY_THREADS=32 nohup python -u run_all.py > run_all.out 2>&1  (resumable; see run_all.out)
Environment: conda env work; BLAS pinned to 1 thread; STUDY_THREADS=32 particle threads; numpy 2.2.5, scipy 1.15.3, MKL 2023.1
Code (exact versions that ran, from git): e403d48
  code/README.md  md5 adcf5f5f52c05110a1f1e1dc00ed0332
  code/analyze.py  md5 ae8a7d00dee5c0240de6dd02d40dbd7a
  code/experiments.py  md5 3844e58eb734d2ed1ab2261ae5a26575
  code/run_all.py  md5 7df40cc0c2acdf019b6fb9aaa4e97c51
  code/study_config.py  md5 2e4c0ccbfeff9125d35a2e1c67e2112b
  code/study_core.py  md5 26219dff6f00a048efc772f3e257daa6

Layout: code/ = copies of the code that ran; output/ = raw results (git-ignored); logs/ = console/study logs (git-ignored); summary/ and plots/ = derived tables and figures (tracked).
This run predates the runs/ convention: it was retrofitted from its original location after it finished; the originals were copied, not moved.

Notes:
  - Original results also remain tracked at DecapStudy/results (commit e403d48); this folder duplicates them under the run layout.
  - Winning band B2; best scopes A3 (MPHY) and A5 (DDR3); Grid 2 at N=73/79 hit the 1800 s cap.
