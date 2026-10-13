# Speed experiments (exact: trajectories must equal the baseline)

| id | script | what we try | what to look at |
|---|---|---|---|
| s1 | `s1_port_scaling.py` | Woodbury cost depends on the capacitor count k, full inversion on the port count N. DDR3 is reduced (open-circuit) to N = 6..79 ports; one swarm evaluation is timed for numpy and woodbury_batch, then full PSO numpy vs fast on ddr21 and ddr79. | `summary/scaling_summary.txt`, `plots/scaling.png`: speed-up should grow with N |
| s2 | `s2_cpu_vs_gpu.py` | Split the gain: same three fitness codes on CPU (torch, `CPU_THREADS`) and GPU. algorithm gain = cpu numpy / cpu woodbury, hardware gain = cpu woodbury / gpu woodbury. | `summary/cpu_vs_gpu_summary.txt`, `plots/cpu_vs_gpu_bench.png` |
| s3 | `s3_profile_gpu_decode.py` | Profile fast per iteration (decode / evaluate / other), then decode on the GPU (vectorised FIX-12 port resolution) with the same NumPy RNG. | `summary/profile_summary.txt`, `plots/time_breakdown.png`, exact match must be 1.000 |
| s4 | `s4_batched_seeds.py` | Run all seeds of a threshold in lock-step, one batched GPU call per capacitor-count group, each run with its own RandomState (same stream as run_pso). | `summary/lockstep_summary.txt`: throughput gain, identical trajectories 1.000 |
