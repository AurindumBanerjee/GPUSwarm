RUN decapstudy2/run01_exp2_stopped_20261004  [stopped (killed at Stage 1 run 15/45; targets redesigned)]
Exp 2 first launch: targets from Exp 1 B2 floors x [1.02..1.5], one B2-derived target set applied to all bands, Grid 1 stopped on the tightest level. Stopped because B1 cells met every level with 0 capacitors (invalid cells).

Start: 2026-10-04 ~16:48 (first log line)    End: 2026-10-04 ~17:00 (killed)    Host: eeiitj123.iitj.ac.in
Launch: cd DecapStudy2 && STUDY_THREADS=32 ./run_exp2.sh
Environment: conda env work; BLAS pinned to 1 thread; STUDY_THREADS=32; PID 86342 (killed by user request)
Code (exact versions that ran, from git): 655bc52
  code/README.md  md5 5cede2579547761195f0e121faf8ff54
  code/analyze.py  md5 e0ce3de2066eaa83b0a49728530a4bf4
  code/check_cost_agreement.py  md5 413b71402f5af49f3ff309b364a1d270
  code/derive_targets.py  md5 9c33aec873767754b7946c99aaa0555a
  code/experiments.py  md5 718e680ed5a60f25d3902d00f1287a85
  code/run_all.py  md5 d1abd2ac1d537d66db16076e45fc7550
  code/run_exp2.sh  md5 30adbfb09dae0f0cfe6e52bd4480d57e
  code/study_config.py  md5 d5b64169e144fc596a560ae6e0db6c03
  code/study_core.py  md5 a1c1d9615135cf5a3f96e375eaeb8c70
  code/targets.json  md5 d36c9c88279aaa029bc14a1789f7f599
  code/validate.py  md5 75eaa36b2328121b0eb0b892786c0d6c

Layout: code/ = copies of the code that ran; output/ = raw results (git-ignored); logs/ = console/study logs (git-ignored); summary/ and plots/ = derived tables and figures (tracked).
This run predates the runs/ convention: it was retrofitted from its original location after it finished; the originals were copied, not moved.

Notes:
  - 51 records written before the stop; superseded by the per-band-target redesign (commit ed9b2a1: per-band targets, Grid 1 full budget, floor pass).
  - Not resumed: the per-band relaunch has not been started.
