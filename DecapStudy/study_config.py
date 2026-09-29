"""
Central configuration for the decap-placement study (CPU version).

Everything a reviewer may want to challenge lives here: data paths, rail
voltages / current assumptions -> Ztarget, band definitions, staging budgets.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- paths
# Server defaults; override with env vars (PDN_DATA, STUDY_OUT) -- nothing
# here is tied to a local machine. Expected layout under DATA_ROOT:
#   MPHY/  VDDQ_DDR3/  Decaps/   (as in the Dataset README)
DATA_ROOT = os.environ.get("PDN_DATA", "/DATA/Aurindum/Swarming/Dataset")
OUT_ROOT = os.environ.get("STUDY_OUT", os.path.join(HERE, "results"))

# ---------------------------------------------------------------- seeding
# Paired seeds: every cell uses seed = BASE_SEED + run_id*10000 + n_caps, so
# cells see identical particle initialisation at the same run index (and the
# same stage), independent of scope, band or inversion method.
BASE_SEED = 12345

# ---------------------------------------------------------------- bands
# Upper edge in Hz of the optimisation band. None = full grid (99.66 MHz).
#   B4 = full band, each frequency weighted by how much the best single
#   capacitor can move |Z_kk| there (see study_core.movement_weights), so the
#   optimiser is not scored on bins it cannot influence.
WEIGHTED = "weighted"
BANDS = {"B1": 20e6, "B2": 50e6, "B3": None, "B4": WEIGHTED}
BAND_LABEL = {"B1": "<=20 MHz", "B2": "<=50 MHz", "B3": "full (99.66 MHz)",
              "B4": "full, movement-weighted"}

# Band used to COMPARE cells whose own objectives differ (Stage-1 screening,
# winning-band choice, recommendation). Scoring is always reported on all
# four bands; this only decides which column ranks the cells. Must be a
# physical band (B1-B3): B4's weighted peak is not a real impedance.
CARE_BAND = os.environ.get("CARE_BAND", "B2")

# MAX_CAPS = number of decap pads (every index except the observation ports),
# capped at 20 ONLY for the 21-port DDR3 PDN. Keys are Problem names.
MAX_CAPS_CAP = {"ddr21": 20}

# Threads used to evaluate the particles of one iteration in parallel (numpy
# releases the GIL in LAPACK). Results are identical to serial. Grid 2 always
# runs serially so its timings are clean.
THREADS = int(os.environ.get("STUDY_THREADS", "1"))

# ---------------------------------------------------------------- targets
# Ztarget_k = V_k * RIPPLE / dI_k   (classic target-impedance rule).
# ASSUMPTIONS (not in the datasets): rail voltages from the rail names in the
# MPHY README; MPHY_VDD taken as 1.0 V; 5 % allowed ripple; dI is the
# assumed transient current step per rail. dI values are deliberately small
# (tens to hundreds of mA) so the resulting targets (0.2-1.8 ohm for MPHY)
# sit in the range the library can actually influence -- the README bounds
# show a target below ~0.2 ohm is unreachable on every MPHY rail.
RIPPLE = 0.05

# name, mat file, obs index in y_sp_mphy_full (0-based), first pad index in
# y_sp_mphy_full (0-based), pad count, rail voltage [V], current step [A]
MPHY_RAILS = [
    ("mer1",    "y_sp_mer1.mat",    0,  5, 15, 1.0, 0.25),
    ("mer2",    "y_sp_mer2.mat",    1, 20, 15, 1.0, 0.25),
    ("pll1v0",  "y_sp_pll1v0.mat",  2, 35, 14, 1.0, 0.10),
    ("pll1v8",  "y_sp_pll1v8.mat",  3, 49, 15, 1.8, 0.05),
    ("mphyvdd", "y_sp_mphyvdd.mat", 4, 64,  9, 1.0, 0.10),
]
IMPROVABLE_RAILS = ["mer2", "pll1v0", "mphyvdd"]      # README: MER1 / PLL1V8 have ~no leverage

DDR_VOLTAGE, DDR_DI = 1.35, 1.0          # VDD_DDR_1V35, 1 A step -> 67.5 mohm

# ---------------------------------------------------------------- scopes
SCOPES = ["A1", "A2", "A3", "A4", "A5"]
SCOPE_LABEL = {
    "A1": "MPHY rail-wise",
    "A2": "MPHY whole package",
    "A3": "MPHY improvable only",
    "A4": "DDR3 21-port",
    "A5": "DDR3 79-port",
}

# ---------------------------------------------------------------- PSO
W_MAX, W_MIN = 0.9, 0.4
C1, C2 = 1.5, 1.5
WARM_FRACTION = 0.4          # share of the swarm carrying the previous stage's best
WARM_JITTER = 0.02

STAGES = {
    1: dict(n_particles=20, n_iters=6,  runs=3),      # screening (reduced)
    2: dict(n_particles=50, n_iters=15, runs=10),     # full (matches SB.py)
}
AUTO_CONTINUE_AFTER_STAGE1 = True   # Stage 2 starts by itself after Stage 1
SCREEN_FACTOR = 2.0          # Stage 1: drop cells with median > 2x row best

# Optional stagnation stop: end a run when the best cost has not improved by
# more than PATIENCE_TOL (relative) for PATIENCE consecutive capacitor
# counts. None = off, i.e. always run to MAX_CAPS as specified.
PATIENCE = None
PATIENCE_TOL = 1e-3

# ---------------------------------------------------------------- Grid 2
GRID2_METHODS = ["pure_python", "numpy", "solve", "sm", "iterative"]
GRID2_SIZES = {          # N -> (problem key, description)
    16: "mer2",          # single-rail MPHY (improvable rail)
    21: "ddr21",
    73: "mphy_full",     # whole package, 5 observation ports
    79: "ddr_full",
}
GRID2_RUNS = 5
GRID2_PURE_RUNS = 2
GRID2_PURE_SIZES = [16, 21]        # executed; N=73/79 extrapolated from n^3 scaling
GRID2_TIME_LIMIT_S = 1800          # per-run wall-clock cap (flagged as timed_out)
GRID2_PSO = dict(n_particles=50, n_iters=15)
EVAL_TIMING_N = 12                 # fitness evaluations timed per (method, size)
EVAL_TIMING_CAPS = 10

# ---------------------------------------------------------------- consistency
CONSISTENCY_TOL = 0.01             # <=1 % relative deviation of band peak => "matches"
