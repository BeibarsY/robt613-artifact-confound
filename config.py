"""Project-wide configuration: run mode, paths, seeds and dataset constants.

Every constant tagged ``verified: data`` was checked by loading the data
(see scripts/01_audit.py). Constants tagged ``source: Brunner2008`` come from
the dataset description (report/refs/Brunner2008_desc_2a.pdf).
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- run mode
# SMOKE = 1 subject, 1 session (fast check). FULL = all 9 subjects, both sessions.
MODE: str = os.environ.get("ROBT613_MODE", "SMOKE").upper()
CONFIGS: dict[str, dict] = {
    # n_permutations: rows used by the H1 decision rule (eog_only, both classifiers, every analysis);
    # n_permutations_other: every other row (chance reported, not used in a decision rule).
    # Reduced from 100 for the 30 min Colab budget, as PLAN.md Phase 8 allows (NOTES.md).
    "SMOKE": {"subjects": [1], "sessions": ["0train"], "n_permutations": 20, "n_permutations_other": 10},
    "FULL": {"subjects": list(range(1, 10)), "sessions": ["0train", "1test"], "n_permutations": 100,
             "n_permutations_other": 30},
}
CFG: dict = CONFIGS[MODE]

# ---------------------------------------------------------------- paths
ROOT: Path = Path(__file__).resolve().parent
RESULTS: Path = ROOT / "results"
TABLES: Path = RESULTS / "tables"
FIGURES: Path = RESULTS / "figures"
CACHE: Path = RESULTS / "cache"
for _p in (TABLES, FIGURES, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- reproducibility
SEED: int = 42

# ---------------------------------------------------------------- dataset facts
SFREQ: float = 250.0                     # verified: data; source: Brunner2008
N_EEG: int = 22                          # verified: data; source: Brunner2008
N_EOG: int = 3                           # verified: data; source: Brunner2008
EOG_CHANNELS: list[str] = ["EOG1", "EOG2", "EOG3"]  # verified: data (names)
# EOG positions from Brunner2008 Fig. 3 (right), read from the rendered figure: #1 at the
# outer canthus on the viewer's right, #2 above the nasion between the eyebrows, #3 at the
# outer canthus on the viewer's left. Assuming the face is drawn facing the reader:
# EOG1 = subject's LEFT outer canthus, EOG2 = glabella, EOG3 = subject's RIGHT outer canthus.
# Consistent with data: blinks are largest on EOG2 (fig_1a, fig_3b).
# HEOG = EOG1 - EOG3 (> 0 when the cornea moves toward EOG1, i.e. gaze to the subject's left,
# under that assumption). VEOG = EOG2 - mean(EOG1, EOG3) (no infra-orbital electrode).
# Both are reference-free (the shared mastoid reference cancels).
EOG_POSITIONS: dict[str, str] = {"EOG1": "left outer canthus", "EOG2": "glabella", "EOG3": "right outer canthus"}
EOG_HEOG: tuple[str, str] = ("EOG1", "EOG3")
EOG_VEOG: tuple[str, tuple[str, str]] = ("EOG2", ("EOG1", "EOG3"))
REFERENCE: str = "left mastoid"          # source: Brunner2008
GROUND: str = "right mastoid"            # source: Brunner2008
ACQ_BANDPASS_HZ: tuple[float, float] = (0.5, 100.0)  # source: Brunner2008
ACQ_NOTCH_HZ: float = 50.0               # source: Brunner2008 ("additional 50 Hz notch")
SESSIONS: list[str] = ["0train", "1test"]  # verified: data (MOABB keys)
SESSION_MAT_SUFFIX: dict[str, str] = {"0train": "T", "1test": "E"}  # source: Brunner2008 Table 1
N_RUNS_PER_SESSION: int = 6              # verified: data
N_TRIALS_PER_RUN: int = 48               # verified: data; source: Brunner2008

# ---------------------------------------------------------------- event timing
# Trial timing (source: Brunner2008): fixation cross at t=0, cue arrow at t=2 s
# for 1.25 s, MI until t=6 s.
# MOABB puts the class annotation at STI trigger + interval[0] = trial start + 2 s,
# i.e. at CUE ONSET (verified: data, scripts/01_audit.py checks a 500-sample shift
# for every trial). So epochs locked to the annotation are cue-locked.
# Confirmed in Phase 4: the class-dependent HEOG deflection starts 0.19-0.25 s after the
# annotation and peaks at 0.36 s, with nothing before 0 (neuro_eog_latency.csv).
T_TRIAL_START_TO_CUE: float = 2.0        # source: Brunner2008; verified: data
ANNOT_IS_CUE_ONSET: bool = True          # verified: data (sample shift + EOG latency)
CUE_DURATION: float = 1.25               # source: Brunner2008
MI_END_AFTER_CUE: float = 4.0            # source: Brunner2008 (t=6 s)
EPOCH_TMIN: float = -1.0                 # relative to cue (fixation baseline)
EPOCH_TMAX: float = 4.0
BASELINE: tuple[float, float] = (-1.0, 0.0)
MI_WINDOW: tuple[float, float] = (0.5, 2.5)  # relative to cue, for band power

EVENT_ID: dict[str, int] = {"feet": 1, "left_hand": 2, "right_hand": 3, "tongue": 4}  # verified: data
CLASSES_4: list[str] = ["left_hand", "right_hand", "feet", "tongue"]
CLASSES_LR: list[str] = ["left_hand", "right_hand"]

# ---------------------------------------------------------------- channel sets
# 2a has no Fp, T7/T8 or P7/P8 electrodes (verified: data). See NOTES.md.
PHYSIO_CHANNELS: list[str] = ["C3", "Cz", "C4", "FC3", "FC4", "CP3", "CP4"]
ARTIFACT_PRONE_CHANNELS: list[str] = ["Fz", "FC1", "FC2", "FC3", "FC4", "C5", "C6"]
CENTRAL_CHANNELS: list[str] = ["C3", "C1", "Cz", "C2", "C4"]

BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0), "theta": (4.0, 8.0), "mu": (8.0, 13.0),
    "beta": (13.0, 30.0), "high": (30.0, 45.0),
}

# ---------------------------------------------------------------- cleaning (Phase 3)
CONDITIONS: list[str] = ["none", "eog_reg", "ica"]
# Where EOG regression and ICA are fitted. "calibration" = the session's EOG calibration
# runs (eyes open / closed / movements, read from the .mat), applied to the MI runs.
# This keeps every MI trial out of the cleaning fit (course notebook 04 "fit on
# calibration"). "session" = fit on the 6 MI runs themselves (PLAN.md original). See NOTES.md.
CLEAN_FIT_ON: str = "calibration"
# 20 components: CAR leaves rank 21. n_components=0.99 kept only 9-14 components, and with
# n components the largest reachable z-score is (n-1)/sqrt(n) (2.85 for n=10), so the default
# z=3 rule excluded nothing in 9/18 recordings (see NOTES.md, Phase 3).
ICA_KW: dict = {"n_components": 20, "method": "fastica", "max_iter": "auto", "random_state": SEED}
ICA_DECIM: int = 1                       # A04T calibration is only 80 s; fits take ~2 s each
# Exclude a component if |Pearson r| with any EOG channel on the fitting data >= threshold
# (mne ICA.find_bads_eog, measure="correlation"). Chosen from fig_3f.
ICA_EOG_MEASURE: str = "correlation"
ICA_EOG_THRESHOLD: float = 0.5
# Peak-to-peak rejection on EEG channels of the `none` condition, applied identically to all
# conditions. Set after inspecting fig_3c (see NOTES.md). None = reject nothing.
PTP_REJECT_UV: float | None = None

# ---------------------------------------------------------------- features (Phase 5)
WELCH_NPERSEG: int = 250                 # 1 s segments -> 1 Hz bins; MI window gives 3 segments (50 % overlap)
AMP_WINDOW: tuple[float, float] = (0.0, 1.0)  # s after cue, mean amplitude bins (gaze-shift capture)
AMP_BIN: float = 0.1
FEATURE_SETS: list[str] = ["eog_only", "eeg_all", "eeg_physio", "eeg_artifact_prone"]
TASKS: list[str] = ["left_vs_right", "four_class"]
CLASSIFIERS: list[str] = ["logreg", "slda"]
# 4-class = multinomial logistic regression only (PLAN.md 1); sLDA is the binary robustness check
TASK_CLASSIFIERS: dict[str, list[str]] = {"left_vs_right": ["logreg", "slda"], "four_class": ["logreg"]}
N_JOBS: int = int(os.environ.get("ROBT613_NJOBS", "-1"))  # permutations in parallel (Colab CPU = 2)
FRONTAL_AMP_CHANNELS: list[str] = ["Fz", "FC1", "FC2", "FC3", "FC4"]  # exploratory feature set
