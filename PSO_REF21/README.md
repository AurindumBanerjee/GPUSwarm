# PSO_REF21

Does the new Murata library improve the 21-port reference PDN (`Data/y2.mat`, observation port 0, 20 pads, 100.99 Hz - 99.83 MHz)?
Same PSO core as PSO_SP (`pso_sp_core.py`, unchanged), same seeds for both libraries, one objective (peak |Z11| over the whole grid).

* `Data/decaps.mat` - legacy library (3348 models, admittance) - arm `ref`
* `Data/murata_decaps_ref21.mat` + `murata_index_ref21.csv` - 16171 Murata parts on the freq2 grid (v2 series-RLC extrapolation, admittance; row i = row i of the index) - arm `murata`
  made with `Dataset/Scripts/resample_murata_library_rlc.py` (`MURATA_GRID_MAT=Reference/freq2.mat`)
* Targets: 50, 45, 40 and 30 mOhm (the 4-threshold variant of SB.py) x NUM_RUNS (10) + FLOOR_RUNS (3) full-budget runs per library.

    python pso_ref21.py --validate
    bash run_study.sh           # validate, study, summary -> logs/summary.txt, results/<lib>/...
Bare peak 0.3612 ohm. PSO settings: 50 particles, 15 iterations (SB.py's values; override with N_PARTICLES / N_ITERATIONS).
