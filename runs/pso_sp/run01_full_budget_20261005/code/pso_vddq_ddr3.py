#!/usr/bin/env python3
"""
PSO decap placement - VDDQ DDR3 board (SP_Sep2026 extraction, to 1 GHz).

One rail, VDD_DDR_1V35, in two sizes:

  y_sp_ddr21     21 ports   1 obs + 20 pads   same size as the original y2.mat
                                              benchmark, so speedups stay
                                              comparable to earlier results
  y_sp_ddr_full  79 ports   1 obs + 78 pads   every pad

Both report the same bare |Z11|, which is the correctness check on the
open-circuit reduction: dropping ports left OPEN cannot change the impedance at
the observation port. If those two ever differ, the reduction is wrong.

Cost note: Gauss-Jordan is O(n^3), so one pure_python evaluation costs about
6.0 s at 79 ports against 0.11 s at 21. Run pure_python on ddr21 only.

  python pso_vddq_ddr3.py --validate
  python pso_vddq_ddr3.py
  NUM_RUNS=3 STUDY_THREADS=16 python pso_vddq_ddr3.py
"""
import os
import pso_sp_core as core

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(core.DATA, "VDDQ_DDR3")

PROBLEMS = [
    dict(name="ddr21",    path=os.path.join(D, "y_sp_ddr21.mat"),    n_obs=1,
         obs_labels=["VDD_DDR_1V35"]),
    dict(name="ddr_full", path=os.path.join(D, "y_sp_ddr_full.mat"), n_obs=1,
         obs_labels=["VDD_DDR_1V35"]),
]

if __name__ == "__main__":
    core.main("VDDQ_DDR3", PROBLEMS, os.path.join(HERE, "results_vddq_ddr3"))
