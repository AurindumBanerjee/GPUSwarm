# 02 Mixed precision + FP64 iterative refinement

RTX A5000 (GA102): FP64 at 1/32 of FP32, no FP64 tensor cores. Woodbury
systems are LU-factorised in complex64 and refined with complex128
residuals (`--refine`, default 2); the pruning screen runs fully in
complex64 with a safety margin (`--screen-margin`, default 1e-2).

`accuracy.json` reports error vs refinement steps and the observed
complex64 screen error (a warning is printed if it exceeds the margin).

Methods: numpy, fast, wb_mp, fast_mp.
