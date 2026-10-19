# PSO_PCB28

PSO decap placement on the 28-port PCB PDN (`Dataset/PCB_27ports`), 100.9 Hz - 2.9986 GHz, with the 16,171-part Murata library
(series-RLC extrapolation version). Reuses `pso_sp_core.py` (copy of PSO_SP's) unchanged.

Self-contained: `Data/` holds `y_pcb_netA.mat` (port 1 + its 11 pads, exact for Z11), `y_pcb28.mat` (full network, validation only),
`freq_pcb.mat`, `murata_decaps_pcb_rlc.mat` (decaps stored as ADMITTANCE, 16171x2828) and `murata_index_rlc.csv` (row i = part i).

Bands: B20 (<=20 MHz), B100, B1G, B3G (whole grid). Default objective `RANK_BAND=B3G`. Bare peaks: 0.522 / 2.68 / 234.9 / 234.9 ohm.

    python pso_pcb28.py --validate                       # data, passivity, netA-vs-full check, method agreement
    NUM_RUNS=3 STUDY_THREADS=4 python pso_pcb28.py       # PSO, results in results_murata/
    METHODS=numpy,solve,sm python pso_pcb28.py --validate
Env: RANK_BAND, NUM_RUNS (10), STUDY_THREADS (1), TIME_LIMIT_S (1800), OUT, PCB_DATA.
netA matrix index k (0..11) = original port [1,3,5,6,10,11,12,18,19,22,23,25][k]+... i.e. k=0 is port 1 (observation), k=1..11 are ports 3,5,6,10,11,12,18,19,22,23,25.
