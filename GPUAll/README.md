# GPUAll -- the AllCPU study on one GPU, as a comparative study of separate arms

Uses **AllCPU unchanged** (`../AllCPU/common`: the four PDNs over their own ranges, bands, floor-based levels, 5 swarm configurations,
20 % warm start, patience, records, plots). GPUAll only adds GPU fitness and new search arms, so every result is directly comparable
with AllCPU's CPU runs. Data configuration as AllCPU: `ALLCPU_GRID=5g`, `ALLCPU_LIB=murata` (default). PDN 3/4 need `AllCPU/data_built`.

## Arms

| group | arm | what it is | compare on |
|---|---|---|---|
| exact (identical trajectory to CPU numpy) | `gpu_numpy` | full N x N inverse per placement on the GPU (only on `--numpy-swarms`, default 50x15,100x40) | speed reference |
| | `woodbury_batch` | multi-port Woodbury (k x k solve, all observation ports, all bands), whole swarm per GPU call | speed |
| | `fast` | woodbury_batch + per-run placement cache + exact lower-bound pruning against pbest | speed |
| search (exact scores) | `pso_polish` | AllCPU's PSO + gradient polish of the stage best (R-L-C relaxation, Adam, snap to real parts, exact sweep) | success, caps, best |
| | `ga` | genetic algorithm with the PSO's staging, seeding, warm start, budget, stop rule and patience | success, caps, best |
| | `ga_polish` | ga + polish | success, caps, best |

Separate studies: `greedy_oracle/` (exact best single capacitor over every part x pad, forward + 1-swap refinement, deterministic) and
`cost_model/` (measured + fitted cost per evaluation vs N, k, F, n_obs, swarm; leave-one-N-out validation; decision rule per PDN).

## Files

| file | role |
|---|---|
| `gpuall_eval.py` | GpuEval (multi-port Woodbury, gpu_numpy), GpuPool (batched / fast), dispatcher into allcpu_core, verify |
| `gpuall_search.py` | pso_polish / ga / ga_polish (AllCPU record format), R-L-C fit, Polisher |
| `run_gpuall.py` | driver: `--floor` (PDN 3/4 floors, AllCPU's floor rule) and the study (resumable) |
| `compare_gpuall.py` | summary, trajectory identity, paired speed-ups (vs gpu_numpy and AllCPU cpu_numpy / cpu_solve), Fisher tests, plots |
| `greedy_oracle/greedy_oracle.py`, `cost_model/cost_model.py` | the two separate studies |
| `run_gpuall.sh` | launcher: floor -> study -> greedy -> cost -> compare |
| `targets_gpuall.json` | created on first use from AllCPU's targets; floors filled by `--floor` (AllCPU's file is never written) |

## Run (server, GPU idle)

    cd /DATA/Aurindum/GPUSwarm
    bash GPUAll/run_gpuall.sh --quick                                                    # smoke test
    nohup setsid bash GPUAll/run_gpuall.sh > GPUAll/run_gpuall.out 2>&1 < /dev/null &   # everything
    bash GPUAll/run_gpuall.sh --steps study --pdns pdn1,pdn2 --swarms 50x15,100x40 --arms fast,ga,pso_polish --runs 10

Outputs: `out_<cfg>/<pdn>/<swarm>/<arm>/` (AllCPU layout incl. per-run impedance plots), `compare_<cfg>/summary.md`,
`greedy_oracle/out_<cfg>/summary.md`, `cost_model/out/decision.md`.

## Checked so far (cloud workspace, CPU torch; not yet on the GPU or on PDN 2-4)
* GPU arms vs AllCPU numpy: max relative difference ~5e-16 on every band.
* PDN 1, 50x15, T3: woodbury_batch, fast, gpu_numpy and AllCPU's CPU numpy (job_worker.py) identical in best, caps and evaluations;
  3-observation-port ratio objective with pruning identical to numpy.
