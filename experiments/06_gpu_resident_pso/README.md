# 06 GPU-resident PSO

Whole PSO state on the GPU: torch RNG per (run, n_caps), tensor decoding,
vectorised FIX-12 port resolution (checked against the reference decoder
at start-up), batched Woodbury, cumulative-min reproduction of the
immediate-stop scan, one host sync per iteration.

Different RNG stream from the NumPy baseline -> compare time, success,
caps and evaluations, not trajectories.

Methods: numpy, fast, gpu_pso, gpu_pso_prune.
