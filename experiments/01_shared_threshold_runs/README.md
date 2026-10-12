# 01 Shared-threshold runs

Seeding depends only on (run, n_caps), so a run aimed at the lowest target
visits exactly the states the higher-target runs visit before they stop.
One run per (method, run) records the first crossing of every threshold.

Runs both protocols (separate = GPUBench2, shared = new) and writes
`shared_vs_separate.json` with the per-threshold agreement rate (expected
1.0) and the total benchmark wall-time saving.

    python bench_shared_thresholds.py --methods numpy,fast
