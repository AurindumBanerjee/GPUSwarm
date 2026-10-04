"""
Central configuration for the decap-placement study, EXPERIMENT 2 (CPU).

Own copy of the Exp 1 code (DecapStudy/); nothing here imports or writes into
Exp 1. The one scientific change versus Exp 1: target impedances are no longer
derived from assumed voltages/currents -- they are read from targets.json
(written by derive_targets.py from Exp 1's results). The study refuses to
start while any target is still null.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- paths
# Input data is read-only and lives outside this directory:
#   DATA_ROOT/{MPHY,VDDQ_DDR3,Decaps}/   (as in the Dataset README)
DATA_ROOT = os.environ.get("PDN_DATA", "/DATA/Aurindum/Swarming/Dataset")

# Everything the study WRITES stays inside this directory. These are fixed on
# purpose (no env override); _inside() asserts it.
OUT_ROOT = os.path.join(HERE, "results")      # results_long.jsonl, tables, report, cache
FIG_DIR = os.path.join(OUT_ROOT, "figs")
LOG_DIR = os.path.join(HERE, "logs")          # study.log, run_exp2.out
TARGETS_FILE = os.path.join(HERE, "targets.json")


def _inside(path):
    p = os.path.abspath(path)
    assert p == HERE or p.startswith(HERE + os.sep), f"{p} is outside {HERE}"
    return p


for _p in (OUT_ROOT, FIG_DIR, LOG_DIR, TARGETS_FILE):
    _inside(_p)

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

# Threads used to evaluate the particles of one iteration in parallel (numpy
# releases the GIL in LAPACK). Results are identical to serial. Grid 2 always
# runs serially so its timings are clean.
THREADS = int(os.environ.get("STUDY_THREADS", "1"))
# BLAS layers must stay single-threaded (stacked on the particle threads they
# oversubscribe the cores: ~35x slower measured on the server). run_exp2.sh
# exports these before python starts; setdefault here is the backstop and
# only works because study_config is imported before numpy.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# ---------------------------------------------------------------- port rules
# Observation ports come FIRST in every dataset and never take a capacitor.
# Decap pads are every remaining index, and the capacitor limit of a run is
# the number of pads: MAX_CAPS = N - n_obs, derived from the loaded matrix
# (study_core.Problem), never hardcoded.
# The observation-port COUNT cannot be read from a matrix, so it is declared
# here: 1 everywhere except y_sp_mphy_full.mat, which has five (indices 0-4).
DEFAULT_N_OBS = 1
N_OBS_BY_FILE = {"y_sp_mphy_full.mat": 5}

# Layout of the five rails inside y_sp_mphy_full (Dataset/MPHY/README, 0-based):
# name, rail file, observation index, first pad index, pad count.
# --validate cross-checks every row against the loaded matrices.
MPHY_RAILS = [
    ("mer1",    "y_sp_mer1.mat",    0,  5, 15),
    ("mer2",    "y_sp_mer2.mat",    1, 20, 15),
    ("pll1v0",  "y_sp_pll1v0.mat",  2, 35, 14),
    ("pll1v8",  "y_sp_pll1v8.mat",  3, 49, 15),
    ("mphyvdd", "y_sp_mphyvdd.mat", 4, 64,  9),
]
IMPROVABLE_RAILS = ["mer2", "pll1v0", "mphyvdd"]      # README: MER1 / PLL1V8 have ~no leverage

# ---------------------------------------------------------------- targets
# Targets live in targets.json (see derive_targets.py). One entry per rail
# (mer1 ... mphyvdd) and per DDR3 PDN size (ddr21, ddr_full -- their capacitor
# budgets differ, so their floors do too), one value per threshold level.
# A level is a multiplier on the Exp 1 floor; levels are reported against, and
# the run stops when the TIGHTEST level (smallest multiplier) is met.
DEFAULT_MULTIPLIERS = [1.1, 1.25, 1.5]
DDR_TARGET_KEYS = ["ddr21", "ddr_full"]
EXP1_RESULTS = os.path.join(os.path.dirname(HERE), "DecapStudy", "results", "results_long.jsonl")  # read-only

# Objective per problem. "raw": minimise max_k peak |Z_kk| in ohm (targets only
# set the stop condition). "ratio": minimise max_k peak |Z_kk| / Ztarget_k
# (rails compete fairly); this is A2, the whole-package problem.
RATIO_OBJECTIVE_PROBLEMS = ["mphy_full"]

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
SCREEN_FACTOR = 2.0          # Stage 1: drop cells with median > 2x row best

# Optional stagnation stop: end a run when the best cost has not improved by
# more than PATIENCE_TOL (relative) for PATIENCE consecutive capacitor
# counts. None = off, i.e. always run to MAX_CAPS as specified.
PATIENCE = None
PATIENCE_TOL = 1e-3

# ---------------------------------------------------------------- Grid 2
GRID2_METHODS = ["pure_python", "numpy", "solve", "sm", "iterative"]
GRID2_SIZES = {          # N -> problem key
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
