# Notes, deviations and open questions

## Deviations from PLAN.md (with reason)
- **Artifact-prone channel set.** PLAN.md lists T7/T8/P7/P8 "whichever exist". None exist in 2a
  (22 channels, Fz to POz, see fig_1b). Used: frontal Fz, FC1, FC2, FC3, FC4 + most lateral C5, C6.
  FC3/FC4 are also in the physio set. The two sets do not share *features* because the bands are
  disjoint (physio = mu+beta, artifact-prone = delta+high). Team to confirm.
- **Rejected-trial flags.** MOABB 1.7.2 can annotate them (`artifact_handling` option), but we read
  `run.artifacts` directly from the .mat so we can keep them as a per-trial column without letting
  them reject anything. Stored in `results/tables/audit_trials.csv` (`expert_artifact`).

## Dataset facts and their sources
- NEMAR README (`~/mne_data/NEMAR/nm000139/README.md`) says online filter 0.05-200 Hz and ground
  "unknown". Brunner2008 says 0.5-100 Hz + 50 Hz notch, ground right mastoid. We use Brunner2008.
- Calibration-run durations in the .mat (e.g. S1 0train 118.7 / 80.7 / 165.9 s) do not match the
  nominal 2 / 1 / 1 min in Brunner2008. Order of the three blocks in the .mat is not labelled.
  TODO-verify block order before using them (e.g. from the eye-movement block's EOG signature).
- A04T has 1 calibration run (79.6 s), consistent with Brunner2008 ("only the eye movement condition").
- Brunner2008: "The EOG channels ... must not be used for classification" (motivation for H1).

## Phase 2 decisions (evidence: fig_2a, fig_2b, fig_2c, signal_psd_qc.csv, signal_filter_params.csv)
- **Band-pass 1-45 Hz, zero-phase FIR (firwin, Hamming), 825 taps (3.3 s).** Lower transition 1 Hz
  (-6 dB at 0.5 Hz). Upper transition set to **5 Hz** instead of MNE's default 11.25 Hz. With the
  default, 50 Hz is only -4.5 dB, with 5 Hz it is -51 dB at the same filter length.
- **High-pass 1 Hz.** Every recording has more power below 1 Hz than at 1-4 Hz (EEG drift column in
  signal_psd_qc.csv, all positive, EOG larger). 1 Hz also helps ICA (drift is non-stationary and
  dominates variance, which hurts ICA unmixing) and keeps delta (1-4 Hz) for the artifact-band features.
- **No separate notch.** Residual 50 Hz above the 40-60 Hz flanks is up to about +20 dB (S5 1test) in
  the unfiltered data despite the acquisition notch, but the analysis filter gives -51 dB at 50 Hz.
  The 100 Hz harmonic (S1 C3, EOG3) is also in the stop band.
- **No resampling.** 250 Hz gives Nyquist 125 Hz, well above 45 Hz. The data are small (288 trials x
  25 ch per session), so resampling would save little time and add another filter step.
- **Filter EEG and EOG identically** so EOG regression / ICA see the same band on both.
  Open point for Phase 5: a sustained gaze shift is a slow step. The 1 Hz high-pass turns it into a
  transient. EOG amplitude features in 100 ms bins over 0-1 s after cue should survive this (the step
  onset is kept), but check fig_4c before fixing the EOG feature filter.
- **High-band narrowband peaks.** Two subjects have a peak inside the 30-45 Hz 'high' feature band
  (S1 about 35-36 Hz at FC4, S9 about 31-32 Hz at C4, both sessions). All others are below 1 dB over
  the 1/f fit. Origin not known (do not guess). Matters for H3 because 'high' is part of the
  artifact-prone set. Phase 4 should check whether these peaks differ by class.

## Phase 3 decisions (evidence: fig_3a-3f, qc_*.csv)
- **Cleaning models are fitted on the EOG calibration runs, not on the MI runs (deviation from
  PLAN.md).** PLAN.md says fit EOG regression and ICA on the session's raw. That puts every test-fold
  trial inside the fit (label-free, but it breaks the PLAN.md 8 rule "never fit any transformer on
  data that will be tested on"). Course notebook 04 teaches "fit on calibration, audit on held-out".
  2a ships per-session EOG calibration runs for this purpose (Brunner2008). They are filtered and
  CAR'd exactly like the MI runs, then the fitted models are applied to the MI runs. Switch:
  `config.CLEAN_FIT_ON` ("calibration" | "session"). A04T calibration = 80 s eye-movement block only.
  Check that the fit transfers. The excluded ICA components still correlate with EOG on the MI runs.
  Fz-EOG |r| falls on the MI runs in all 18 recordings (fig_3e, qc_eog_correlation.csv).
- **CAR before regression and ICA, EEG only.** fig_3d: with the recorded left-mastoid reference, C5
  (closest to the reference) shows artificially low power. That is a left/right asymmetry that
  would bias a left-vs-right decoder. CAR removes it. EOG channels keep the recorded reference.
- **ICA = FastICA, 20 components, decim 1, random_state 42.** First attempt used
  n_components=0.99 (PLAN.md) and the MNE default z-score rule (z >= 3). 0.99 kept only 9-14
  components, and with n components the largest reachable |z| is (n-1)/sqrt(n) = 2.85 for n = 10.
  So the rule excluded nothing in 9/18 recordings and Fz-EOG2 median |r| stayed at 0.67 (none 0.73).
  CAR leaves rank 21, so 20 components. decim 1 because A04T has only 80 s of calibration data
  (samples / components^2 = 50, others 170-270). Fits take about 2 s.
- **ICA exclusion rule: |r| >= 0.5 between the component and any EOG channel on the fitting data**
  (`find_bads_eog(measure="correlation")`). fig_3f shows a clear gap between about 0.25 and 0.55
  over all 360 components. Result: 2 components excluded in 17 recordings, 3 in S5 1test.
  Median Fz-EOG2 |r| on the MI runs: none 0.73, eog_reg 0.08, ica 0.15.
  `find_bads_muscle` is not used anywhere. It is not in course notebook 04.
- **Peak-to-peak rejection: none (`config.PTP_REJECT_UV = None`).** After 1-45 Hz + CAR the largest
  per-epoch EEG PTP over all 5184 epochs is 142 µV (99.5th pct 107 µV). fig_3c shows no gross
  outliers, so the 150 µV example threshold would reject nothing anyway. The secondary bump at 70-80 µV
  in `none` shrinks after cleaning (it is ocular). A threshold that cut it would remove the trials
  whose effect we are measuring. The same 288 epochs per recording are used in every condition
  (asserted in tests/test_smoke.py).
- **EOG regression can inject artifacts.** Regression subtracts a scaled copy of the EOG channels.
  When an EOG channel has a non-ocular event (fig_3b2: a -900 µV step on all three EOG channels with
  slow recovery, absent from the EEG) the regression writes it into the EEG. 6 epochs have eog_reg
  PTP above the largest `none` PTP (up to 395 µV). All 6 are expert-flagged artifact trials.
  -> Phase 6/7 sensitivity analysis: repeat with expert-flagged trials excluded.
- No epochs straddle run boundaries (make_epochs raises otherwise).

## Phase 4 findings (evidence: fig_4a-4e, neuro_*.csv)
- **EOG positions** read from Brunner2008 Fig. 3 (rendered with pymupdf): EOG1 left outer canthus,
  EOG2 glabella, EOG3 right outer canthus, assuming the face is drawn facing the reader. HEOG = EOG1 - EOG3,
  VEOG = EOG2 - mean(EOG1, EOG3). Left-arrow cues give HEOG > 0, i.e. gaze toward the cued side under
  this assumption. Only the sign interpretation depends on it. The class difference does not.
- **Event timing confirmed.** The grand-average left-right HEOG difference starts 0.19-0.25 s after
  the annotation and peaks at 0.36 s. Zero samples cross the threshold before 0 s (neuro_eog_latency.csv).
- **The confound exists before any classifier.** Mean HEOG 0.2-1.0 s after cue, left minus right:
  positive in 17/18 recordings, p < 0.001 (Welch t) in 15/18 (neuro_eog_evoked_stats.csv). S3 is the
  exception. The vertical arrows (feet = down, tongue = up) give the matching VEOG deflections (fig_4c).
  So the 4-class task has the same confound. A return eye movement follows at about 1-1.3 s (arrow
  off at 1.25 s), inside the MI band-power window (0.5-2.5 s).
- **Leak into frontal EEG.** Left-right difference at FC3 (+1.9 µV) and FC4 (-1.5 µV) at about 0.3 s in
  `none`. Both cleanings reduce it but about 1 µV remains (fig_4d). Cleaning is partial.
- **ERD exists but is weak in some subjects.** Weak-ERD flag = lateralization above -0.5 dB in both mu
  and beta: S2 0train, S5 both sessions, S7 0train (neuro_erd_lateralization.csv). They are kept in the
  analysis as candidate BCI-inefficient subjects.
- **High-band peaks (Phase 2 follow-up).** S1 FC4 36 Hz does not differ by class (Kruskal p = 0.73 / 0.96).
  S9 C4 31-32 Hz differs strongly by class (p about 1e-30, left vs right 1e-19 / 1e-31). It is lateralized
  over the right hemisphere, so it is probably physiological high beta. So 'high' (30-45 Hz) is not a pure
  artifact band. Keep this in mind when interpreting H3 for S9.

## Phase 5-6 design (src/features.py, src/decode.py, scripts/05)
- **Features are computed once per recording x condition and cached.** This is safe because every
  extractor is per trial and stateless (Welch log band power, baseline-corrected window means).
  Nothing is estimated across trials (tests: test_features_are_per_trial, test_no_leakage_in_cv).
  StandardScaler and the classifier are fitted inside each training fold.
- Band power: Welch, 1 s Hamming segments with 50 % overlap on the MI window 0.5-2.5 s after cue,
  1 Hz bins, band edges [lo, hi). Amplitude: 10 bins of 100 ms over 0-1 s after cue, baseline -1..0 s.
- eog_only = EOG1-3 amplitude bins (30) + EOG delta/theta band power (6). The derived HEOG/VEOG
  are linear combinations of EOG1-3, so they add nothing for a linear classifier and are left out.
- **Logistic regression C fixed at 1.0.** Nested leave-one-run-out GridSearch over
  {0.01, 0.1, 1, 10} (decode_nested_C.csv) changes AUC by a median of 0.008 on S1 (FULL value in
  the table). Nested tuning inside every permutation would cost about 20x more than the Colab budget allows.
- **Chance level for every row.** Within-run label permutations (keeps per-run class counts), the same
  leave-one-run-out CV, 95th percentile stored as chance_p95. One child seed per permutation, so the null
  does not depend on n_jobs. **Budget (PLAN.md Phase 8 allows reducing n_permutations).** 100 permutations
  for every row the H1 decision rule uses (left vs right, eog_only, both classifiers, all robustness
  analyses), 30 for every other row (`n_permutations_other`). Cross-session null = refit on permuted
  training labels, same counts. With 2 workers the decoding script takes 374 s locally (10-core Apple
  laptop, results/timing.csv). Colab is estimated at 9-12 min, notebook total about 22-26 min including
  the ~9 min download. **Not yet measured on Colab.** If it exceeds 30 min, lower `n_permutations_other`
  first (H1 rows unaffected).
- 4-class task = multinomial logistic regression only (PLAN.md 1 names it). sLDA is used for the binary
  robustness check. The nested-C check is run on eeg_all only.
- Confusion matrices: 4-class, logreg, leave-one-run-out predictions summed over subjects.
- **Weight maps (H3b)** come from a full-session refit for visualisation only (flagged in the figure).
  They are never used as a performance estimate.

## Phase 7 results (stats_h1/h2/h3.csv, stats_robustness.csv, stats_exploratory.csv)
- H1 supported. H2 not supported. H3 not supported. Robustness: H1 holds in all 5 analyses, H2 in 0 of 10.
- **Exploratory, not pre-registered, chosen after seeing H2:** time-domain amplitude bins at Fz, FC1-4
  (`eeg_frontal_amp`). Median AUC does not drop significantly after cleaning, but the per-subject drop
  correlates strongly with EOG-only AUC (rho 0.98 regression, 0.85 ICA, 0train). Must be labelled
  exploratory in report and slides.
- Interpretation: the gaze-shift confound is a time-domain event. The pre-registered EEG feature sets are
  band power in the MI window, and they do not use it in a measurable way.

## Phase 9 report
- slide_pdf/report/main.tex (IEEEtran) + refs.bib, slide_pdf/slides/slides.tex (beamer). No LaTeX on the dev machine, so compile on
  Overleaf (upload slide_pdf/report/ with figures/ and numbers.tex). Every number is a macro from slide_pdf/report/numbers.tex,
  generated by `python slide_pdf/report/make_numbers.py` from results/tables. Re-run it after any re-run of the notebook.
- References: brunner2008 opened in full. The other five were checked for title / venue / pages / year by web
  search only. DOIs confirmed for tangermann2012 and pfurtscheller1999. A teammate must open each paper (PLAN.md 6).
- TODO team: names, affiliation, AI-disclosure wording, page count (draft about 2500 words + 5 figures).

## Phase 8 notebook
- Colab install cell pins mne==1.13.2 and moabb==1.7.2 (same as the course notebooks). numpy/scipy/
  sklearn/pandas/matplotlib are Colab defaults, and their versions are printed in the last cell.
  requirements.txt pins the local dev environment.

## TODO-verify
- EOG orientation assumption (face drawn facing reader): a teammate should look at Fig. 3 and confirm.
- Re-pin requirements.txt on Colab.
