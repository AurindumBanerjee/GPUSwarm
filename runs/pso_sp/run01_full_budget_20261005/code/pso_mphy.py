#!/usr/bin/env python3
"""
PSO decap placement - MPHY package (SP_Sep2026 extraction, to 1 GHz).

Five electrically independent rails plus the whole 73-port package. Each rail is
its own problem: cross-rail transfer impedance is >=25 dB below self impedance,
so a capacitor on one rail cannot help another.

  y_sp_mer1      16 ports   1 obs + 15 pads   VDDE1V0_MER1
  y_sp_mer2      16 ports   1 obs + 15 pads   VDDE1V0_MER2
  y_sp_pll1v0    15 ports   1 obs + 14 pads   VDD1V0_PLL
  y_sp_pll1v8    16 ports   1 obs + 15 pads   VDDA1V8_PLL
  y_sp_mphyvdd   10 ports   1 obs +  9 pads   MPHY_VDD
  y_sp_mphy_full 73 ports   5 obs + 68 pads   all five at once; the objective is
                                              the worst of the five observation
                                              ports, so rails share one budget

  python pso_mphy.py --validate      data, port rules and method agreement only
  python pso_mphy.py                 full run
  NUM_RUNS=3 STUDY_THREADS=16 python pso_mphy.py
  METHODS=numpy,solve,sm RANK_BAND=BFULL python pso_mphy.py

Note on mer1 and pll1v8: on this extraction their bare networks sit within 2-5%
of the best achievable, because the observation port is at the die and package
inductance swamps the pads. Expect near-flat curves there; it is a property of
the board, not a bug.
"""
import os
import pso_sp_core as core

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(core.DATA, "MPHY")

PROBLEMS = [
    dict(name="mer1",      path=os.path.join(D, "y_sp_mer1.mat"),    n_obs=1,
         obs_labels=["VDDE1V0_MER1"]),
    dict(name="mer2",      path=os.path.join(D, "y_sp_mer2.mat"),    n_obs=1,
         obs_labels=["VDDE1V0_MER2"]),
    dict(name="pll1v0",    path=os.path.join(D, "y_sp_pll1v0.mat"),  n_obs=1,
         obs_labels=["VDD1V0_PLL"]),
    dict(name="pll1v8",    path=os.path.join(D, "y_sp_pll1v8.mat"),  n_obs=1,
         obs_labels=["VDDA1V8_PLL"]),
    dict(name="mphyvdd",   path=os.path.join(D, "y_sp_mphyvdd.mat"), n_obs=1,
         obs_labels=["MPHY_VDD"]),
    dict(name="mphy_full", path=os.path.join(D, "y_sp_mphy_full.mat"), n_obs=5,
         obs_labels=["VDDE1V0_MER1", "VDDE1V0_MER2", "VDD1V0_PLL",
                     "VDDA1V8_PLL", "MPHY_VDD"]),
]

if __name__ == "__main__":
    core.main("MPHY", PROBLEMS, os.path.join(HERE, "results_mphy"))
