# Search-quality experiments (change the search; every score is still exact)

| id | script | what we try | what to look at |
|---|---|---|---|
| q1 | `q1_polish_variants.py` | Gradient polish raised hard-target success but cost ~6x time. Variants: shorter descent, polish only near the target, no local sweep, every stage. | `summary/polish_variants_summary.txt`: time per success, Fisher p vs fast |
| q2 | `q2_library_reduction.py` | Smaller libraries made of real parts: dominance pruning on |Z(f)| (heuristic) and k-means representatives (`CLUSTER_K`). | `summary/library_summary.txt`: success / evals / caps vs full library |
| q3 | `q3_warm_start_sweep.py` | Warm-start fraction (0..1) x jitter grid with fast. 40 % beat 100 % at 0.04 in round 1. | `summary/warm_start_grid.csv`, `plots/warm_start_success.png` |
| q4 | `q4_optimizer_comparison.py` | PSO vs differential evolution, genetic algorithm and simulated annealing, same exact fitness, budget, staging, warm start and stop rule. | `summary/optimiser_summary.txt`, `plots/optimisers.png` |
