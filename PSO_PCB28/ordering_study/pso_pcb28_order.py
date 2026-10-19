#!/usr/bin/env python3
"""Same PSO, same PDN, same seeds as ../pso_pcb28.py -- only the ROW ORDER of the decap library differs.
  ORDER=srf|cap|part|random   (default srf)      results -> results_<ORDER>/
  python pso_pcb28_order.py --validate           # also checks that the permuted library holds the same parts
  ORDER=part NUM_RUNS=10 STUDY_THREADS=4 python pso_pcb28_order.py
Placements record part names, so results from different orders stay comparable. Everything else (bands, RANK_BAND,
NUM_RUNS, STUDY_THREADS, TIME_LIMIT_S, PCB_DATA, MURATA_MAT) works as in pso_pcb28.py."""
import os, sys
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(H))
import numpy as np
import pso_pcb28 as base
core = base.core
ORDER = os.environ.get("ORDER", "srf")
if __name__ == "__main__":
    core.library()                                              # loads Y and names (SRF-sorted file order)
    perm = np.load(os.path.join(H, "orders.npz"))[f"perm_{ORDER}"]
    d0, n0 = core._lib["d"], core._lib["names"]
    assert len(perm) == d0.shape[0] and len(set(perm.tolist())) == len(perm)
    core._lib["d"] = d0[perm]; core._lib["names"] = [n0[i] for i in perm]; del d0
    print(f"ORDER={ORDER}: library rows permuted ({len(perm)} parts); first rows -> {core._lib['names'][:3]}")
    if "--validate" in sys.argv:
        assert sorted(core._lib["names"]) == sorted(n0), "permutation lost parts"
    core.main(f"PCB28[{ORDER}]", base.PROBLEMS, os.environ.get("OUT", os.path.join(H, f"results_{ORDER}")))
