"""AllCPU registry of the four PDNs and everything needed to run one (PDN, band) cell with allcpu_core.

  pdn1  21-port reference board  (PSO_REF21/Data: y2.mat, decaps.mat legacy library)       1 obs, 20 pads, 1391 pts, 101 Hz - 99.8 MHz
  pdn2  28-port full PCB         (PSO_PCB28/Data: y_pcb28.mat, Murata RLC library)         1 obs, 27 pads, 2828 pts, 100.9 Hz - 3 GHz  (optimised on the whole grid, kept)
  pdn3  MPHY package, all rails  5 obs (indices 0-4), 68 pads
  pdn4  DDR3 board               1 obs (index 0), 78 pads
Capacitor allowance = number of ports - number of observation ports for EVERY PDN (only observation ports are excluded), read from the matrix.

PDN 3 / PDN 4 data configuration CFG = <grid>_<library>, from env (read at import):
  ALLCPU_GRID = 5g (default)  boards extracted to 5 GHz (build_data.py) | 1g  the existing 1 GHz sets of GPUSwarm/Data (110.9 Hz - 997 MHz, 2588 pts)
  ALLCPU_LIB  = murata (default)  Murata RLC library resampled on the board grid (build_data.py; measured to 3-6 GHz, series-RLC below, log-log above)
              | combined  legacy+AVX library of GPUSwarm/Data/Decaps (1g grid only; extrapolated above 99.83 MHz)
  ALLCPU_DATA = folder written by build_data.py (default <AllCPU>/data_built)
Bands (optimised separately, every record re-scored on all): B20 (<=20 MHz) B50 (<=50 MHz) B200 (<=200 MHz) B1G (<=1 GHz; = whole grid on 1g) and, on 5g only, B5G (whole grid, canonical 100 Hz - 5 GHz).
Objectives: pdn1/2/4 peak |Z11| in ohm; pdn3 ratio max_k peak_k/floor_k over its five observation ports (floor_k = rail-wise floor of the band).
Targets/levels in targets_allcpu.json (pdn3/pdn4 floors per CFG and band, filled by floor_pass.py)."""
import os, sys, csv, json

COMMON = os.path.dirname(os.path.abspath(__file__)); ALLCPU = os.path.dirname(COMMON)
ROOT = os.environ.get("GPUSWARM_ROOT", os.path.dirname(ALLCPU))              # the GPUSwarm folder (server: /DATA/Aurindum/GPUSwarm)
TARGETS_FILE = os.environ.get("ALLCPU_TARGETS", os.path.join(COMMON, "targets_allcpu.json"))
DATA_BUILT = os.environ.get("ALLCPU_DATA", os.path.join(ALLCPU, "data_built"))
GRID = os.environ.get("ALLCPU_GRID", "5g"); LIB34 = os.environ.get("ALLCPU_LIB", "murata"); CFG = f"{GRID}_{LIB34}"
assert GRID in ("1g", "5g") and LIB34 in ("murata", "combined") and CFG != "5g_combined", f"unsupported ALLCPU_GRID/ALLCPU_LIB {CFG} (the combined library only exists on the 1 GHz grid)"
sys.path.insert(0, COMMON)

BANDS_1G = {"B20": (None, 20e6), "B50": (None, 50e6), "B200": (None, 200e6), "B1G": None}
BANDS_5G = {"B20": (None, 20e6), "B50": (None, 50e6), "B200": (None, 200e6), "B1G": (None, 1e9), "B5G": None}
BANDS_34 = BANDS_5G if GRID == "5g" else BANDS_1G
BAND_LABEL = {"B20": "<=20 MHz", "B50": "<=50 MHz", "B200": "<=200 MHz", "B1G": "<=1 GHz", "B5G": "whole grid, to 5 GHz (canonical)",
              "BFULL": "101 Hz-99.8 MHz (whole PDN 1 grid)", "B3G": "100.9 Hz-3 GHz (whole PDN 2 grid)"}
MPHY_RAILS = ["mer1", "mer2", "pll1v0", "pll1v8", "mphyvdd"]                   # observation index 0..4 in the full MPHY matrix (DecapStudy2/study_config.py)


def _board(name):
    """paths of the PDN 3 / PDN 4 matrices for the active grid"""
    if GRID == "1g":
        return {"mphy_full": os.path.join(ROOT, "Data", "MPHY", "y_sp_mphy_full.mat"), "ddr_full": os.path.join(ROOT, "Data", "VDDQ_DDR3", "y_sp_ddr_full.mat"),
                **{r: os.path.join(ROOT, "Data", "MPHY", f"y_sp_{r}.mat") for r in MPHY_RAILS}}[name]
    return os.path.join(DATA_BUILT, "boards_5g", f"y_5g_{name}.mat")


RAIL_FILES = {r: _board(r) for r in MPHY_RAILS}
LIB_FILE = os.path.join(DATA_BUILT, f"murata_{GRID}_rlc.mat"); LIB_INDEX = os.path.join(DATA_BUILT, f"murata_{GRID}_rlc_index.csv")
LIB34_KIND = "murata_file" if LIB34 == "murata" else "combined"
EXTRAP34 = 1e9 if LIB34 == "murata" else 99.83e6

SPEC = {
    "pdn1": dict(label="PDN 1 (21-port)", name="pdn1_ref21", n_obs=1, bands={"BFULL": None}, opt_bands=["BFULL"], extrap_hz=1e9,
                 path=os.path.join(ROOT, "PSO_REF21", "Data", "y2.mat"), lib="ref21", ratio=False, est_cost=1),
    "pdn2": dict(label="PDN 2 (28-port)", name="pdn2_pcb28", n_obs=1, bands={"B3G": None, "B1G": (None, 1e9)}, opt_bands=["B3G"], extrap_hz=1e9,
                 path=os.path.join(ROOT, "PSO_PCB28", "Data", "y_pcb28.mat"), lib="pcb28", ratio=False, est_cost=3),
    "pdn3": dict(label="PDN 3 (MPHY, 73-port, 5 obs)", name=f"pdn3_mphy_{CFG}", n_obs=5, bands=dict(BANDS_34), opt_bands=list(BANDS_34), extrap_hz=EXTRAP34,
                 path=_board("mphy_full"), lib=LIB34_KIND, ratio=True, est_cost=40),
    "pdn4": dict(label="PDN 4 (DDR3, 79-port)", name=f"pdn4_ddr_{CFG}", n_obs=1, bands=dict(BANDS_34), opt_bands=list(BANDS_34), extrap_hz=EXTRAP34,
                 path=_board("ddr_full"), lib=LIB34_KIND, ratio=False, est_cost=45),
}
PDN_KEYS = list(SPEC)
METHODS = ["numpy", "solve", "sm", "iterative"]                                # pure_python is a separate baseline (not part of AllCPU)
SWARMS = ["50x15", "100x15", "50x40", "100x40", "200x40"]                      # particles x iterations; warm start is always 20 % of the particles


def targets():
    return json.load(open(TARGETS_FILE))


def load_library(core, kind):
    """Install the decap library of a PDN into core._lib and make core.library() return it."""
    import numpy as np, scipy.io as sio
    core._lib.clear()
    if kind == "combined":                                                       # GPUSwarm/Data/Decaps/combined_decaps.mat on the 2588-point grid (core default)
        core.DATA = os.path.join(ROOT, "Data"); core.library = lambda: core._lib if core._lib else _combined(core)
        core.library(); return
    if kind == "murata_file":                                                    # Murata RLC library resampled onto the PDN 3/4 grid by build_data.py
        m = sio.loadmat(LIB_FILE); d = m["decaps"]; f = m["freq"].ravel()
        names = [r["part_number"] for r in csv.DictReader(open(LIB_INDEX, encoding="utf-8"))] if os.path.exists(LIB_INDEX) else [f"murata_{i}" for i in range(d.shape[0])]
        assert len(names) == d.shape[0]
    elif kind == "ref21":
        D = os.path.join(ROOT, "PSO_REF21", "Data"); f = sio.loadmat(os.path.join(D, "freq2.mat"))["freq"].ravel()
        d = sio.loadmat(os.path.join(D, "decaps.mat"))["decaps"]; names = [f"ref_{i}" for i in range(d.shape[0])]
        if os.environ.get("PDN1_LIB", "ref") == "murata":                       # optional: the Murata library of PDN 1 (not the default of the earlier studies)
            m = sio.loadmat(os.path.join(D, "murata_decaps_ref21.mat")); d = m["decaps"]; assert np.allclose(m["freq"].ravel(), f, rtol=1e-9)
            names = [r["part_number"] for r in csv.DictReader(open(os.path.join(D, "murata_index_ref21.csv")))]
    elif kind == "pcb28":
        D = os.path.join(ROOT, "PSO_PCB28", "Data"); f = sio.loadmat(os.path.join(D, "freq_pcb.mat"))["freq"].ravel()
        m = sio.loadmat(os.path.join(D, "murata_decaps_pcb_rlc.mat")); d = m["decaps"]; assert np.allclose(m["freq"].ravel(), f, rtol=1e-9)
        names = [r["part_number"] for r in csv.DictReader(open(os.path.join(D, "murata_index_rlc.csv")))]
    else: raise ValueError(kind)
    assert d.shape[1] == len(f) and len(names) == d.shape[0] and np.isfinite(d).all()
    core._lib.update(d=d, names=names, slope=np.ones(d.shape[0]), freq=f, lists=None)
    core.library = lambda: core._lib


def _combined(core):
    import scipy.io as sio
    m = sio.loadmat(os.path.join(core.DATA, "Decaps", "combined_decaps.mat"))
    core._lib["d"] = m["decaps"]; core._lib["names"] = [str(x[0][0]) for x in m["names"]]; core._lib["slope"] = m["loglog_slope"].ravel()
    core._lib["freq"] = sio.loadmat(os.path.join(core.DATA, "Decaps", "sp_freq.mat"))["freq"].ravel(); core._lib["lists"] = None
    return core._lib


def tsec(T, key, band):
    """the target section of (pdn, band): PDN 3/4 are stored per CFG"""
    return T[key][band] if key in ("pdn1", "pdn2") else T[key][CFG][band]


class Cell:
    """One PDN ready to run: core configured, problem loaded, bands installed."""
    def __init__(self, key, n_particles=50, n_iters=15, load_problem=True):
        os.environ.setdefault("STUDY_THREADS", "1")
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.setdefault(v, "1")
        os.environ.setdefault("WARM_START_PCT", "20")
        import allcpu_core as core
        self.core = core; self.key = key; self.spec = SPEC[key]; s = self.spec
        core.N_PARTICLES, core.N_ITERATIONS = int(n_particles), int(n_iters)
        core.BANDS.clear(); core.BANDS.update(s["bands"]); core.EXTRAP_START_HZ = s["extrap_hz"]
        load_library(core, s["lib"])
        self.prob = core.Problem(s["name"], s["path"], n_obs=s["n_obs"]) if load_problem else None
        self.bare = None

    def bare_peaks(self):
        if self.bare is None:
            core = self.core; self.bare = {b: core.bare_peaks(self.prob)[b] for b in core.BANDS}
        return self.bare

    def select_band(self, band):
        """Install RANK_BAND / objective weights for `band`; return (levels {name: threshold in cost units}, info). Raises if the floors are not filled yet."""
        core = self.core; s = self.spec; core.RANK_BAND = band
        t = tsec(targets(), self.key, band)
        if "levels_ohm" in t:                                                    # PDN 1 / PDN 2: fixed whole-number targets
            core.OBJ_WEIGHTS = None; levels = dict(t["levels_ohm"]); floor = t["floor_ohm"]
        else:                                                                    # PDN 3 / PDN 4: levels from the band's floor and bare impedance
            if s["ratio"]:
                fl = t.get("floors_ohm")
                if not fl or any(fl.get(r) is None for r in MPHY_RAILS): raise RuntimeError(f"{self.key}/{CFG}/{band}: rail floors missing in {TARGETS_FILE} -- run floor_pass.py first")
                import numpy as np
                core.OBJ_WEIGHTS = np.array([1.0 / fl[r] for r in MPHY_RAILS]); floor = 1.0     # cost = max_k peak_k/floor_k >= 1
            else:
                floor = t.get("floor_ohm")
                if floor is None: raise RuntimeError(f"{self.key}/{CFG}/{band}: floor missing in {TARGETS_FILE} -- run floor_pass.py first")
                core.OBJ_WEIGHTS = None
            bare0 = float(core.cost_of(self.bare_peaks(), band)); T_ = targets()
            if T_.get("levels_mode", "frac") == "frac":                           # floor + g x (bare - floor): always strictly inside (floor, bare)
                levels = {k: floor + g * (bare0 - floor) for k, g in T_["levels_frac"].items()}
            else: levels = {k: floor * m for k, m in T_["levels_mult"].items()}      # DecapStudy2 style
        bare = float(core.cost_of(self.bare_peaks(), band))
        for k, v in levels.items(): assert floor < v < bare, f"{self.key}/{band}: level {k}={v:g} not inside (floor {floor:g}, bare {bare:g})"
        return levels, dict(floor=floor, bare=bare, unit="ratio" if s["ratio"] else "ohm")

    def order(self, levels):
        """easiest level first (largest threshold), as in the earlier studies."""
        return sorted(levels, key=lambda k: -levels[k])


class _SwarmCost(dict):                                                           # evaluations per capacitor count, relative to 50x15 (any "PxI")
    def __missing__(self, sw): P, I = (int(x) for x in sw.split("x")); return P * I / 750.0
SWARM_COST = _SwarmCost()
METHOD_COST = {"solve": 0.5, "numpy": 1.0, "sm": 1.5, "iterative": 1.6}           # rough s/eval ratios from validate.py at N=73/79 (used only to order the queue)


def targets_status(pdns, bands=None):
    """List of '<pdn>/<band>' whose floors are still null (floor_pass.py not run yet) or that are not optimisation bands of that PDN."""
    T = targets(); missing = []
    for k in pdns:
        for b in (bands or SPEC[k]["opt_bands"]):
            if b not in SPEC[k]["opt_bands"]: missing.append(f"{k}/{b} (not an optimisation band of {k} on {CFG})"); continue
            try: t = tsec(T, k, b)
            except KeyError: missing.append(f"{k}/{CFG}/{b} (no entry in targets file)"); continue
            if "levels_ohm" in t: continue
            ok = (t.get("floors_ohm") and all(t["floors_ohm"].get(r) is not None for r in MPHY_RAILS)) if SPEC[k]["ratio"] else t.get("floor_ohm") is not None
            if not ok: missing.append(f"{k}/{CFG}/{b}")
    return missing


def job_cmd(pdn, method, swarm, out, bands=None, runs=None, levels=None):
    cmd = [sys.executable, "-u", os.path.join(COMMON, "job_worker.py"), "--pdn", pdn, "--method", method, "--swarm", swarm, "--out", out]
    if bands: cmd += ["--bands", ",".join(bands)]
    if runs: cmd += ["--runs", str(runs)]
    if levels: cmd += ["--levels", ",".join(levels)]
    return cmd


def thread_env():
    e = dict(os.environ, STUDY_THREADS="1", WARM_START_PCT=os.environ.get("WARM_START_PCT", "20"), ALLCPU_GRID=GRID, ALLCPU_LIB=LIB34)
    for k in ("OMP", "OPENBLAS", "MKL", "NUMEXPR"): e[k + "_NUM_THREADS"] = "1"
    return e
