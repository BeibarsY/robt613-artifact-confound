# Contribution log

ROBT613 midterm project "Artifacts as confounds in motor-imagery decoding" (BNCI2014-001).

## Team and roles

| Role | Member | Leads (PLAN.md phases) | Reviews | Presents (slides) |
| --- | --- | --- | --- | --- |
| Data and provenance lead | Chidiadi Bethel Mba | Loading, data audit, event coding, subject selection (1); notebook assembly and Colab run (8) | Evaluation (4, 7) | 1-3: title, hypotheses, data audit |
| Signal processing lead | Sungkar Bolat | Filtering, re-referencing, artifact handling, quality control (2, 3) | Data and provenance (1) | 4-5: filtering, artifact handling |
| Decoding lead | Beibars Ybraiakhyn | Feature extraction, classifier pipeline, cross-validation (5, 6) | Signal processing (2, 3) | 6-7: stage 4 check, features and evaluation |
| Evaluation and reporting lead | Ramazan Zypayev | Neurophysiological check, chance levels, statistics, figures, report integration (4, 7, 9) | Decoding (5, 6) | 8-11: results, exploratory, limits, conclusion |

Review = re-run the phase script on SMOKE, read the matching NOTES.md section, sign off below with a date.

## Log

| Date | Teammate (directing) | Phase | What |
| --- | --- | --- | --- |
| 2026-10-27 | Beibars Ybraiakhyn | 0-1 | Repo scaffold, config, `src/data.py`, `scripts/01_audit.py`; SMOKE + FULL audit run (Claude Code session) |
| 2026-10-27 | Beibars Ybraiakhyn | 2 | `src/preprocess.py` (concat, band-pass, spectral QC), `scripts/02_signal_inspection.py`; SMOKE + FULL run, fig_2a-c (Claude Code session) |
| 2026-10-27 | Beibars Ybraiakhyn | 3 | CAR, EOG regression, ICA fitted on calibration runs, cue-locked epochs, QC; `scripts/03_artifacts.py`, fig_3a-3f, 2 new tests (Claude Code session) |
| 2026-10-27 | Beibars Ybraiakhyn | 4 | `src/neurophys.py`, `src/features.py`, `scripts/04_neurophys_check.py`; ERD, cue-locked EOG and frontal-EEG checks, fig_4a-4e (Claude Code session) |
| 2026-10-27 | Beibars Ybraiakhyn | 5-6 | `src/decode.py`, `scripts/05_decode_all_conditions.py`; full grid, permutation chance, cross-session, nested-C, expert-excluded sensitivity, leakage tests (Claude Code session) |
| 2026-10-27 | Beibars Ybraiakhyn | 7 | `src/stats.py`, `scripts/06_stats.py`; H1-H3, robustness, exploratory frontal amplitude (Claude Code session) |
| 2026-10-27 | Beibars Ybraiakhyn | 8 | `scripts/build_notebook.py`, notebooks/final.ipynb executed locally FULL (Claude Code session) |
| 2026-10-27 | Beibars Ybraiakhyn | 9 | Draft report (`slide_pdf/report/main.tex`), refs.bib, numbers.tex generator, slides (`slide_pdf/slides/slides.tex`) (Claude Code session) |
| 2026-10-31 | Beibars Ybraiakhyn | team | Roles and responsibilities, per-member study guides in `team/` (RU + EN), 12-minute defense script (Claude Code session) |
| 2026-11-01 | Beibars Ybraiakhyn | 8-9 | Author names in report, slides and notebook; report folder moved to `slide_pdf/`; paths and numbers generator updated (Claude Code session) |
