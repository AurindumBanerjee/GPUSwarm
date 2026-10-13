# GPUExperiments / round2

Second round of experiments, built on `GPUExperiments/gpuexp_common.py` (unchanged) and the experiment-5 polish
(`exp05_gradient_polish.py`). Shared helpers are in `r2_common.py`. Every experiment writes one isolated run folder under
`runs/gpu_round2/` through `reference/new_run.py`, like round 1.

| folder | what it answers |
|---|---|
| `speed/` | where the speed-up comes from and how to get more of it, without changing the search (all exact) |
| `search_quality/` | how to reach hard targets more often or with fewer capacitors (changes the search; scores stay exact) |
| `validation/` | whether the claims hold statistically and on the other PDNs |
| `plots/` | impedance curves with and without decaps, from any finished run |

Every experiment also saves `plots/before_after_<method>_T<target>.png`: |Z11| of the bare PDN and of the best placement on the
same axes, with the target line.

## Data and settings
* Datasets (`DATASET=` / `DATASETS=`): `ref21` (reference y2, legacy library, full band), `ddr21`, `ddr79`, `mer1`, `mer2`,
  `pll1v0`, `pll1v8`, `mphyvdd` (SP extractions with `Data/Decaps/combined_decaps.mat`, band-limited to `SP_FMAX`, default
  50 MHz -- the band-edge problem). `LIB_SOURCE=avx_shunt|legacy|all` filters the combined shunt library (default all).
* Targets: `ref21` uses absolute `TARGETS` (default 0.05,0.045,0.04); SP sets use `TARGET_FRACS` x bare peak (default
  0.6,0.5,0.4). Override with `TARGET_MODE=abs|relative`.
* PSO settings are the round-1 ones (`gpuexp_common.SETTINGS`); `run_round2.sh` defaults to 40 % warm start, jitter 0
  (`--warm`, `--jitter`). `QUICK=1` / `--quick` = tiny smoke test.

## Run
    bash GPUExperiments/round2/run_round2.sh --quick --only s1,q3            # smoke test, two experiments
    bash GPUExperiments/round2/run_round2.sh                                 # everything (s1..s4, q1..q4, v1, v2)
    bash GPUExperiments/round2/run_round2.sh --only v1,q1                    # a subset
Detached on the server:

    ssh -f EEDept 'cd /DATA/Aurindum/GPUSwarm && nohup setsid bash GPUExperiments/round2/run_round2.sh > GPUExperiments/round2/run_round2.log 2>&1 < /dev/null &'

Any extra environment variable listed in a script's docstring can be exported before the call (e.g. `NUM_RUNS_HARD=200`).
Results: `runs/gpu_round2/<run id>/summary/*.txt` (read these first), `plots/`, `output/results_long.jsonl`.

Not implemented: CUDA Graph capture of the fitness kernel (batched solve capture is fragile; S3/S4 address launch overhead
first).
