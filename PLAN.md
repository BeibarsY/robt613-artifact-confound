# ROBT613 Midterm Project — Plan for Claude Code

**Project:** Artifacts as confounds in motor-imagery decoding
**Dataset:** BNCI2014-001 (BCI Competition IV, dataset 2a) via MOABB
**Team:** 4 students, 3 weeks. Submission: one `.ipynb` + Colab link, IEEE 2-column PDF (6–8 pp), slides PDF, contribution log, peer evaluations.

This file is the working spec. Work through it phase by phase. Do not skip the acceptance criteria. Put this file in the repo root next to `CLAUDE.md`.

---

## 0. Hard constraints (from the assignment — never violate)

1. **Scope.** Only methods from BCI Workshop notebooks 00–06 (https://github.com/berdakh/ROBT613-BCI-Workshop):
   - Data: MNE `Raw`/`Epochs`/`Evoked`, MOABB loader, annotations, metadata.
   - Preprocessing: FIR/IIR band-pass and notch, resampling, re-referencing (CAR), EOG regression, ICA, bad-channel marking, baseline correction, amplitude (peak-to-peak) rejection.
   - Features: mean amplitude in time windows, decimated waveforms, Welch PSD, log band power, channel/band selection.
   - Classifiers: LDA (shrinkage), logistic regression (L2).
   - Evaluation: `GroupKFold` / leave-one-group-out, ROC AUC, balanced accuracy, permutation chance level, confusion matrices.
   - **Forbidden in the core result:** CSP, Riemannian, wavelets/time-frequency, Xdawn, CCA, SVM, NN, boosting, ASR, source localization, random epoch splits, tuning on test folds.
   - CSP is allowed only as an *optional* +5% extension, compared against the finished in-scope baseline. Do it last, if at all.
2. **Compute.** The final notebook must run top-to-bottom on a free Colab CPU runtime in under 30 minutes, download included. Keep a `SMOKE` config (1 subject, 1 session) and a `FULL` config (all 9 subjects, both sessions). If FULL exceeds budget, reduce ICA cost (decimate, fewer components) or document a subject subset — never silently drop subjects.
3. **Reproducibility.** Pin versions in `requirements.txt`, set `random_state=42` everywhere (numpy, sklearn, ICA), print package versions and subject IDs in the notebook. Shared code lives in functions, not copied cells.
4. **Integrity.** Every number in the report must come from a file in `results/` produced by the notebook. Never invent dataset facts — verify them by loading the data and from the dataset paper. Code reused from the course notebooks gets a comment `# adapted from ROBT613-BCI-Workshop notebook 0X`. External code gets a citation at the line where it is used. AI tool usage goes in the report appendix.
5. **Topic registration** in the shared spreadsheet by day 3 of Week 1 (dataset = BNCI2014-001, idea = Artifacts as confounds). This is a team action, not a code task.

---

## 1. Research question and pre-registered hypotheses

**Question.** How much of motor-imagery (MI) decoding performance on dataset 2a comes from ocular/muscular artifacts rather than sensorimotor rhythms, and does artifact removal change *which* features win?

**Why it matters.** The cue in 2a is an arrow (left / right / down / up) shown for 1.25 s at t = 2 s of each trial. Subjects were told to fixate, but an arrow cue invites a reflexive saccade or gaze shift in the cued direction, and MI effort invites jaw/neck tension. Any decoder that picks this up works in the lab and fails on a locked-in user who cannot move their eyes — or inflates reported accuracies. 2a is the most used MI benchmark, so a measurable confound there matters for the field.

**Hypotheses (write the predictions before running the full pipeline):**

- **H1 — EOG carries class information.** A decoder trained on the 3 EOG channels alone decodes left vs right hand above the permutation chance level in a majority of subjects. (If H1 is false, that is a clean negative result and still earns full marks — report it as such.)
- **H2 — Cleaning lowers EEG decoding where H1 holds.** Removing ocular activity from EEG (EOG regression; ICA) reduces mu/beta decoding AUC, and the per-subject drop correlates positively with that subject's EOG-only AUC.
- **H3 — Cleaning changes which features win.** Before cleaning, discriminative weight is concentrated on frontal channels and low-frequency (<4 Hz) or high-frequency (>30 Hz) bands; after cleaning it shifts to central channels (C3, Cz, C4) in mu (8–13 Hz) and beta (13–30 Hz). A decoder restricted to the "physiological" feature set (C3/Cz/C4, 8–30 Hz) loses less from cleaning than one restricted to the "artifact-prone" set (frontal/temporal, <4 Hz and >30 Hz).

**Primary task:** left hand vs right hand (binary; lateral cue → lateral eye movement is the most plausible confound). **Secondary:** 4-class (left, right, feet, tongue) with multinomial logistic regression, reported as balanced accuracy and macro one-vs-rest AUC. Tongue imagery is the most EMG-prone class — note this in the interpretation.

**Decision rules (pre-specified):**
- H1 supported if group-level EOG-only AUC exceeds 0.5 by a one-sided Wilcoxon test across the 9 subjects (p < 0.05) **and** at least 5/9 subjects exceed the 95th percentile of their own permutation null.
- H2 supported if paired Wilcoxon (no-cleaning vs cleaning) across subjects gives p < 0.05 and Spearman ρ(EOG-only AUC, AUC drop) > 0.
- H3 supported if the weight-map centroid moves toward central channels and the artifact-prone feature set shows a larger AUC drop than the physiological set (paired Wilcoxon on the drops).
- With n = 9, state the power limitation explicitly in the report.

---

## 2. Repository layout

```
robt613-artifact-confound/
├── CLAUDE.md                  # short: project summary, scope rules, how to run, conventions
├── PLAN.md                    # this file
├── requirements.txt           # pinned versions
├── config.py                  # SMOKE / FULL configs, paths, seeds, constants
├── src/
│   ├── data.py                # MOABB loading, data audit, event coding, trial counts
│   ├── preprocess.py          # filter, CAR, EOG regression, ICA, epoching, rejection, QC report
│   ├── neurophys.py           # ERD check (C3/C4 mu/beta), EOG evoked-by-class check
│   ├── features.py            # Welch log band power, window-mean amplitudes, feature-set selectors
│   ├── decode.py              # sklearn Pipelines, grouped CV, permutation chance, per-subject tables
│   ├── stats.py               # Wilcoxon, Spearman, binomial chance, summary tables
│   └── plots.py               # all figures, consistent style, saved to results/figures/
├── scripts/
│   ├── 01_audit.py
│   ├── 02_signal_inspection.py
│   ├── 03_artifacts.py
│   ├── 04_neurophys_check.py
│   ├── 05_decode_all_conditions.py
│   ├── 06_stats.py
│   └── build_notebook.py      # assembles src/ + scripts/ into notebooks/final.ipynb
├── notebooks/
│   └── final.ipynb            # generated; the submission notebook
├── results/
│   ├── tables/                # csv: audit, rejection counts, AUC per subject×condition, stats
│   └── figures/               # png/pdf, numbered by project stage (fig_1_*, fig_2_*, ...)
├── report/                    # IEEE LaTeX template, figures copied from results/
├── slides/
├── CONTRIBUTIONS.md           # dated log: who did what (also visible in git history)
└── tests/
    └── test_smoke.py          # SMOKE run end-to-end on 1 subject, asserts shapes and no leakage
```

**Conventions:** every function has a docstring and type hints; every script writes its outputs to `results/` and nothing else; no global state except `config.py`; figures use the same channel order and colour map throughout; file names of figures/tables map to the 7 required project stages.

---

## 3. Environment

`requirements.txt` (pin exact versions after `pip install` on Colab; record them with `pip freeze | grep -iE "mne|moabb|scikit|numpy|scipy|pandas|matplotlib"`):

```
mne==<pin>
moabb==<pin>
scikit-learn==<pin>
numpy==<pin>
scipy==<pin>
pandas==<pin>
matplotlib==<pin>
```

Seeds: `SEED = 42` in `config.py`; pass to `np.random.default_rng`, `GroupKFold` shuffling (if used), `LogisticRegression`, `ICA(random_state=SEED)`, `permutation_test_score(random_state=SEED)`.

Data cache: MOABB downloads into `~/mne_data`. In Colab this is re-downloaded each session (~700 MB, several minutes) — include it in the 30-minute budget and print download time.

---

## 4. Phases, tasks and acceptance criteria

Each phase maps to one of the seven required project stages. Run everything on `SMOKE` first, then `FULL`.

### Phase 1 — Data audit (stage 1) · `src/data.py`, `scripts/01_audit.py`

Tasks:
1. Load with `moabb.datasets.BNCI2014_001().get_data(subjects=[...])` → nested dict `subject → session → run → Raw`. Print and **verify** (do not assume): subject IDs 1–9; session keys (recent MOABB: `'0train'`, `'1test'`); 6 runs per session; channel names and types (22 EEG, 3 EOG, stim); sampling rate (expect 250 Hz); units (MOABB rescales to Volts — check `raw.get_data().std()` is ~1e-5, not ~10); montage (set `standard_1005` if MOABB did not); reference (left mastoid, ground right mastoid — from the dataset paper); the acquisition band-pass (0.5–100 Hz) and 50 Hz notch applied at recording.
2. Event coding: dump `raw.annotations` descriptions and `mne.events_from_annotations`. **Verify where the event marker sits** — MOABB's `interval = [2, 6]` implies the marker is at trial start (fixation), cue at +2 s, MI window +3 to +6 s. Confirm from the paper (Brunner et al. 2008) and from the EOG evoked response (cue-locked eye activity should appear after +2 s). Record the final convention in `config.py` (`T_CUE = 2.0`, `MI_WINDOW = (0.5, 2.5)` relative to cue, etc.).
3. Check whether MOABB keeps the expert artifact-rejection flags from the original GDF (event `1023`, "rejected trial") and the EOG calibration runs (eyes open / closed / eye movements). If the rejected-trial flags are available, store them per trial — they are a useful external reference for Phase 3. If not, say so in the audit.
4. Trial counts per class per session per subject → `results/tables/audit_trial_counts.csv` (expect 72 per class per session, 288 per session).
5. Figures: `fig_1a_raw_segment.png` (10 s raw, EEG + EOG, one subject), `fig_1b_sensor_layout.png` (`raw.plot_sensors(show_names=True)`), `results/tables/audit_summary.csv` (one row per recording: subject, session, n_runs, n_eeg, n_eog, sfreq, duration, n_trials, units check).

Acceptance: all facts in the table come from loaded data or are tagged `source: Brunner2008`; the event-timing convention is confirmed by the EOG evoked plot in Phase 4, not assumed.

### Phase 2 — Signal inspection and filtering (stage 2) · `src/preprocess.py`, `scripts/02_signal_inspection.py`

Tasks:
1. PSD (`raw.compute_psd(method='welch')`) before any filtering: one subject, EEG channels averaged and C3/C4 separately, 0–125 Hz. Check: is there a 50 Hz line peak (should be mostly removed at acquisition — decide notch on evidence); is there a low-frequency drift; what does the EOG PSD look like.
2. Choose and justify: FIR band-pass **1–45 Hz** for the "analysis" raw (keeps delta for the artifact-band features and gamma/EMG up to 45 Hz; removes drift). Document why 1 Hz high-pass also helps ICA. Notch only if the PSD shows residual 50 Hz. No resampling (250 Hz is already modest; keep it and say why). Record the filter parameters MNE prints (filter length, transition bands).
3. Figures: `fig_2a_psd_before_after.png` (before/after filter, log scale), `fig_2b_filter_response.png` (optional, `mne.filter.create_filter` + `plot_filter`).

Acceptance: every cut-off is tied to a sentence pointing at a feature in `fig_2a`.

### Phase 3 — Artifact handling and QC (stage 3) · `src/preprocess.py`, `scripts/03_artifacts.py`

This phase produces the **three cleaning conditions** used in every later comparison. Apply each at the session level (concatenate the 6 runs with `mne.concatenate_raws`, keeping run index as metadata for grouping).

| Condition | What it does | Notebook |
| --- | --- | --- |
| `none` | 1–45 Hz filter + CAR only | 03, 04 |
| `eog_reg` | `none` + `mne.preprocessing.EOGRegression(picks='eeg', picks_artifact='eog')` fitted on the session's raw, applied to EEG | 04 |
| `ica` | `none` + ICA (`method='fastica'` or `'picard'`, `n_components=22` or 0.99 variance, fit on 1 Hz high-passed, `decim=3` data), components flagged by `ica.find_bads_eog(raw, ch_name=['EOG1','EOG2','EOG3'])` excluded; also inspect `find_bads_muscle` only as a *diagnostic* figure (do not use its output for the core result unless it is in notebook 04 — check) | 04 |

Re-referencing: CAR for all conditions, applied before regression/ICA. Justify with a figure (mastoid-referenced vs CAR topography of the mean MI-window power). Keep it constant across conditions — it is not the manipulated factor.

Epoching (notebook 05): epochs locked to **cue onset**, `tmin=-1.0` (fixation baseline), `tmax=+4.0`, baseline `(-1.0, 0.0)` for amplitude features; for band-power features use the MI window `(+0.5, +2.5)` relative to cue (document the choice; 2a's standard window is 2–6 s after trial start, i.e. 0–4 s after cue). Peak-to-peak rejection threshold chosen from the amplitude distribution (plot the per-epoch max PTP histogram, pick a threshold at a documented percentile or absolute µV, e.g. 150 µV EEG), applied **identically** in all conditions, computed on the `none` condition so rejection does not become a hidden factor. Alternative to decide and document: reject nothing and let the comparison be purely about cleaning. Pick one, state why.

Outputs: `results/tables/qc_rejection_counts.csv` (per subject×session×condition: n_ica_components_excluded, which components, n_epochs_rejected, n_epochs_kept per class), figures `fig_3a_ica_components_topo.png`, `fig_3b_eog_regression_before_after.png` (a frontal channel around a blink, three traces: none / eog_reg / ica), `fig_3c_ptp_histogram.png`.

Acceptance: same epoch set in all three conditions for a given subject×session (assert it in `tests/test_smoke.py`); rejection counts reported per class so class imbalance after rejection is visible.

### Phase 4 — Neurophysiological check (stage 4) · `src/neurophys.py`, `scripts/04_neurophys_check.py`

Two checks, both **before any decoding**:

1. **The expected signal exists.** Mu/beta ERD over C3/C4: log band power (Welch, notebook 06) in the MI window vs the fixation baseline, per class, per hemisphere → bar plot + topomap of log(MI/baseline) for left vs right (expect contralateral decrease). Also class-wise PSD at C3 and C4. Do this on the `ica` condition (cleanest) and on `none`; if ERD is absent in some subjects, flag them as candidate "BCI-inefficient" and keep them in the analysis.
2. **The suspected confound exists.** Cue-locked evoked EOG (notebook 01 style `Epochs.average()`), horizontal component (EOG1 − EOG3 or whatever the montage implies — verify positions in the paper) and vertical, left vs right vs feet vs tongue, grand average and per subject. A lateral deflection after cue onset that differs by class is the direct evidence for H1. Also an evoked EEG at Fp-like frontal channels (Fz, FC1, FC2 — 2a has no Fp) to see if the same deflection leaks into EEG, and whether `eog_reg`/`ica` remove it.

Figures: `fig_4a_erd_c3c4.png`, `fig_4b_erd_topomap.png`, `fig_4c_eog_evoked_by_class.png`, `fig_4d_frontal_eeg_evoked_before_after_cleaning.png`.

Acceptance: a reader can see ERD and the EOG class difference (or their absence) without any classifier.

### Phase 5 — Features and classifier (stage 5) · `src/features.py`, `src/decode.py`

Features (all inside one sklearn `Pipeline`; scaling fitted on training folds only):

- **Band power features** (notebook 06): Welch PSD on the MI window, log mean power in bands `delta 1–4`, `theta 4–8`, `mu 8–13`, `beta 13–30`, `high 30–45` Hz, per channel → flattened vector. Implemented as a `FunctionTransformer` or a small custom transformer with `fit/transform`.
- **Amplitude features** (notebook 02): mean amplitude in consecutive windows (e.g. 100 ms bins, 0–1 s after cue) on EOG channels and on frontal EEG — this is what captures a gaze shift, which is a DC-like step, not a rhythm.

Feature-set selectors (channel × band masks) — these are the "which features" factor:

| Set | Channels | Bands / features |
| --- | --- | --- |
| `eog_only` | EOG1–3 (+ derived H/V) | amplitude windows + delta/theta band power |
| `eeg_all` | all 22 EEG | all 5 bands |
| `eeg_physio` | C3, Cz, C4 (+ neighbours FC3/FC4/CP3/CP4 — list the exact 2a names) | mu + beta |
| `eeg_artifact_prone` | frontal (Fz, FC1, FC2, FC3, FC4) + lateral temporal/periphery (T7, T8, P7, P8 — whichever exist in 2a) | delta + high |

Classifier: `LogisticRegression(C=..., penalty='l2', max_iter=2000, class_weight='balanced')`; also `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')` as a check that conclusions do not depend on the classifier. Pipeline: `StandardScaler → classifier`. `C` tuned by nested CV (`GridSearchCV` with inner `GroupKFold` by run) over a small grid, or fixed and justified.

Acceptance: `pipeline.fit` is only ever called on training indices; assert in `test_smoke.py` that no transformer is fitted outside `cross_validate`.

### Phase 6 — Honest evaluation (stage 6) · `src/decode.py`, `scripts/05_decode_all_conditions.py`

Evaluation protocol (identical for all conditions):

- **Within-session CV:** `GroupKFold(n_splits=6)` with run as group, per subject per session → mean ± SD AUC across folds; report both sessions, use session `0train` as primary and `1test` as replication.
- **Cross-session:** train on `0train`, test on `1test` per subject — a second, independent estimate.
- **Chance level:** `sklearn.model_selection.permutation_test_score` with the same `GroupKFold`, `n_permutations=100` (raise to 200 in FULL if budget allows), scoring `roc_auc`; store the 95th percentile of the null per subject×condition. Also compute the binomial 95% upper bound for 144 trials (binary) / 288 trials (4-class) as a sanity reference.
- **Metrics:** ROC AUC (binary), balanced accuracy + macro OVR AUC (4-class), confusion matrices for the 4-class case.

Full condition grid (one factor varies per comparison):

```
task        ∈ {left_vs_right, four_class}
cleaning    ∈ {none, eog_reg, ica}
feature_set ∈ {eog_only, eeg_all, eeg_physio, eeg_artifact_prone}
classifier  ∈ {logreg, slda}
```

`eog_only` is evaluated with `cleaning=none` only (EOG channels are never cleaned). Total: 9 subjects × 2 sessions × ≤ (3×3+1) × 2 ≈ 360 CV runs plus permutations — cheap (features are ~100-dim, 288 trials). Time it in SMOKE and extrapolate; cache per-subject feature matrices to `.npz` so permutations do not recompute Welch.

Outputs: `results/tables/auc_per_subject.csv` (long format: subject, session, task, cleaning, feature_set, classifier, auc_mean, auc_sd, chance_p95, n_trials), `results/tables/auc_cross_session.csv`, `fig_6a_auc_per_subject_by_condition.png`, `fig_6b_confusion_4class.png`.

Acceptance: no accuracy from a random epoch split anywhere; every reported AUC has a chance value next to it; the table has one row per subject (no subject-averaged numbers without the per-subject rows behind them).

### Phase 7 — Hypothesis tests (stage 7) · `src/stats.py`, `scripts/06_stats.py`

1. **H1:** EOG-only AUC per subject vs permutation null; one-sided Wilcoxon of AUC − 0.5 across subjects; count subjects above their p95. Figure: `fig_7a_eog_only_auc_vs_null.png` (per-subject bars with null p95 markers).
2. **H2:** paired Wilcoxon `eeg_all/none` vs `eeg_all/eog_reg`, and vs `eeg_all/ica` (two tests, report both, mention multiple comparisons); Spearman ρ between EOG-only AUC and ΔAUC. Figure: `fig_7b_auc_drop_vs_eog_auc.png` (scatter, one dot per subject).
3. **H3:** (a) paired Wilcoxon on ΔAUC(none→ica) between `eeg_physio` and `eeg_artifact_prone`; (b) weight maps: refit `eeg_all` logreg on the full session (for visualisation only, flagged as such) and plot |coef| as channel×band heatmap and as topomap per band, before and after cleaning; compute a centroid (mean |coef| over central vs non-central channels). Figures: `fig_7c_feature_weights_before_after.png`, `fig_7d_delta_auc_by_feature_set.png`.
4. Robustness: repeat H1/H2 with sLDA and on session `1test`; put in a supplementary table.
5. Interpretation checklist for the report: offline-only; n = 9; EOG regression cannot remove EMG; ICA component selection is semi-automatic (document thresholds); cleaning may also remove genuine signal (state what the `eeg_physio` drop implies); cue-locked eye movement is not the same as the subject "cheating".

Outputs: `results/tables/stats_h1.csv`, `stats_h2.csv`, `stats_h3.csv`, with test statistic, p-value, effect size (median Δ and IQR), n.

### Phase 8 — Notebook assembly and Colab validation

1. `scripts/build_notebook.py` uses `nbformat` to generate `notebooks/final.ipynb`: markdown header per stage (copy the hypotheses and the decision rules verbatim from this file), a `!pip install` cell with pinned versions, one cell per `src/` module (contents inlined, so the notebook has no repo dependency), then the scripts' bodies as cells in stage order, then a final cell that prints `pip freeze` of the key packages, subject IDs, seed and total wall time.
2. Run it on a **fresh** Colab CPU runtime (Runtime → Disconnect and delete runtime, then Run all). Record wall time per stage. If > 30 min: reduce `n_permutations`, ICA `decim`, or cache features — never reduce subjects without documenting.
3. Save with outputs. Upload, get the Colab link.

Acceptance: a teammate who did not write it runs it from the link and gets the same `results/tables/stats_h*.csv` values (seeded).

### Phase 9 — Report and slides

Report (IEEE two-column, 6–8 pages, references not counted), section → stage mapping:

1. Introduction and significance (2–3 cited papers; see §6) — why artifact confounds in 2a matter.
2. Dataset and data audit (stage 1) — table from `audit_summary.csv`.
3. Methods: signal inspection (2), artifact handling (3), neurophysiological check (4), features/classifier (5), evaluation (6) — one figure each, every choice justified by a figure or a number.
4. Results: hypothesis tests (7) — the three H figures and the stats tables.
5. Discussion: what the result shows and does not show; limitations; what it means for 2a benchmarks.
6. Appendix: AI-tool disclosure; contribution summary; supplementary tables.

Writing style for the report: no colons or semicolons in body text. Every number is copied from `results/tables/`, never typed from memory.

Slides: 12 minutes, one slide per stage plus hypothesis/result/limits; each member presents the stage they led and must be able to answer on any stage.

---

## 5. Team roles mapped to phases

| Role | Leads | Reviews |
| --- | --- | --- |
| Data and provenance lead | Phases 1, 8 (notebook assembly) | Phase 6–7 |
| Signal processing lead | Phases 2, 3 | Phase 1 |
| Decoding lead | Phases 5, 6 | Phases 2–3 |
| Evaluation and reporting lead | Phases 4 (checks), 7, 9 | Phases 5–6 |

Reviews are PRs: the reviewer re-runs the phase script on SMOKE and signs off in `CONTRIBUTIONS.md` with a date. Every commit message starts with the phase number.

---

## 6. Literature to ground the question (verify every entry — titles, years, DOIs — before citing; do not add references you have not opened)

- Brunner, Leeb, Müller-Putz, Schlögl, Pfurtscheller — "BCI Competition 2008 – Graz data set A" (dataset description: protocol, cue timing, EOG electrode positions, artifact flags).
- Tangermann et al. 2012 — "Review of the BCI Competition IV", Frontiers in Neuroscience.
- Fatourechi, Bashashati, Ward, Birch 2007 — survey of EMG and EOG artifacts in BCI systems, Clinical Neurophysiology.
- McFarland, Sarnacki, Vaughan, Wolpaw 2005 — signal and noise during early BCI training (EMG contamination of MI control), Clinical Neurophysiology.
- Goncharova, McFarland, Vaughan, Wolpaw 2003 — EMG contamination of EEG, spectral and topographical characteristics, Clinical Neurophysiology.
- Pfurtscheller & Lopes da Silva 1999 — ERD/ERS basic principles, Clinical Neurophysiology.

---

## 7. Timeline (3 weeks)

| When | Done |
| --- | --- |
| W1 d1–d2 | Repo set up, `requirements.txt`, SMOKE config, Phase 1 on 1 subject; everyone has run notebooks 00–06 |
| W1 d3 | Topic registered in the spreadsheet; hypotheses and decision rules frozen in PLAN.md |
| W1 d4–d7 | Phases 2–4 on SMOKE, then FULL; checkpoint: figures 1–4 reviewed |
| W2 d1–d4 | Phase 5–6 on SMOKE → FULL; `auc_per_subject.csv` complete |
| W2 d5–d7 | Phase 7 stats; first look at H1–H3; decide whether the optional CSP extension fits the budget |
| W3 d1–d3 | Phase 8 notebook build + fresh-Colab run (twice, by two different people) |
| W3 d4–d6 | Report draft, figures finalised, slides, contribution log |
| W3 d7 | Submission; peer evaluations |

---

## 8. Working rules for Claude Code sessions

- Start each session by reading `CLAUDE.md`, this file, and `results/tables/` to see what already exists. Do not redo finished phases.
- Always run SMOKE before FULL. Time every script and append the time to `results/timing.csv`.
- When a dataset fact is uncertain (event timing, EOG positions, units, session names), load the data and check; if it cannot be checked from the data, mark it `TODO-verify` and stop rather than guess.
- Never fit any transformer or classifier on data that will be tested on. If a shortcut is tempting, write the test first in `tests/test_smoke.py`.
- Stay inside the scope table. If a method outside it seems necessary, write a note in `NOTES.md` explaining why and what the in-scope alternative is — do not implement it.
- No results in markdown or report text that are not read from `results/`. When writing report text, load the CSV and print the numbers in the same cell.
- Keep `CONTRIBUTIONS.md` current: every session that changes code adds a dated line with the human teammate's name who directed it.
