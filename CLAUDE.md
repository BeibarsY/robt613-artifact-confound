# ROBT613 midterm: artifacts as confounds in MI decoding (BNCI2014-001)

Working spec: `PLAN.md` (phases, hypotheses, acceptance criteria). Read it, this file,
`NOTES.md` and `results/tables/` before starting work. Do not redo finished phases.

## Scope (hard rule)
Only methods from ROBT613-BCI-Workshop notebooks 00-06: MNE Raw/Epochs, MOABB, FIR/IIR
filters, CAR, EOG regression, ICA, PTP rejection, window means, Welch log band power,
LDA (shrinkage), L2 logistic regression, GroupKFold by run, AUC, balanced accuracy,
permutation chance. No CSP/Riemann/wavelets/SVM/NN/ASR in the core result. If something
out of scope seems necessary, write it in `NOTES.md` instead of implementing it.

## How to run
```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
ROBT613_MODE=SMOKE .venv/bin/python scripts/01_audit.py   # 1 subject, 1 session
ROBT613_MODE=FULL  .venv/bin/python scripts/01_audit.py   # 9 subjects, 2 sessions
ROBT613_MODE=FULL  .venv/bin/python scripts/02_signal_inspection.py
ROBT613_MODE=FULL  .venv/bin/python scripts/03_artifacts.py      # caches ICA in results/cache/ica
ROBT613_MODE=FULL  .venv/bin/python scripts/04_neurophys_check.py  # also writes feature cache
ROBT613_MODE=FULL  .venv/bin/python scripts/05_decode_all_conditions.py   # ROBT613_NJOBS=2 to mimic Colab
ROBT613_MODE=FULL  .venv/bin/python scripts/06_stats.py
.venv/bin/python scripts/build_notebook.py     # regenerate notebooks/final.ipynb after any src/ change
.venv/bin/python slide_pdf/report/make_numbers.py        # regenerate slide_pdf/report/numbers.tex after any results change
.venv/bin/python -m pytest tests -q
```
Data are cached in `~/mne_data` (~83 MB per subject).

## Conventions
- All constants, seeds (`SEED=42`) and channel sets live in `config.py`.
- Shared code goes in `src/`, scripts in `scripts/NN_*.py` write only to `results/`.
- Docstrings + type hints on every function. Course-derived code gets
  `# adapted from ROBT613-BCI-Workshop notebook 0X`.
- Every script is wrapped in `src.timing.timed(...)` (appends to `results/timing.csv`).
- Figures: `results/figures/fig_<stage><letter>_*.png|pdf` via `src.plots.save`.
- Never fit anything on test data. Never type a number into text that is not read from `results/`.
- Uncertain dataset fact → check the data or mark `TODO-verify`. Never guess.
- Commit messages start with the phase number. Add a dated line to `CONTRIBUTIONS.md` each session.

## Verified dataset facts (Phase 1)
- MOABB annotations sit at **cue onset** (STI trial-start trigger + 2.0 s, checked for every trial).
- Expert artifact flags and EOG calibration runs are not in MOABB's Raw; `src/data.py` reads them
  from the cached source `.mat` files.

## Cleaning conditions (Phase 3) — use `preprocess.build_conditions` + `preprocess.make_epochs`
- `none` = 1-45 Hz FIR + CAR; `eog_reg` / `ica` fitted on the EOG **calibration runs**, applied to MI runs.
- ICA: 20 comps, exclude |r| >= 0.5 with any EOG channel. No PTP rejection. Same 288 epochs in all conditions.

## Status (2026-10-27)
Phases 1-8 done locally (Colab run still to do). Phase 9 drafted (slide_pdf/report/main.tex, slide_pdf/slides/slides.tex).
H1 supported, H2 and H3 not supported. See NOTES.md for every decision and deviation from PLAN.md.
