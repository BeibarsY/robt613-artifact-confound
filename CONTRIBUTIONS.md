# Contribution log

ROBT613 Brain Machine Interface, midterm project "Artifacts as confounds in motor-imagery decoding" (BNCI2014-001).

## Team and roles

| Role | Member | Leads (PLAN.md phases) | Reviews | Presents (slides) |
| --- | --- | --- | --- | --- |
| Data and provenance lead | Chidiadi Bethel Mba | Loading, data audit, event coding, subject selection (1); notebook and Colab run (8) | Evaluation (4, 7) | 1-3: title, hypotheses, data audit |
| Signal processing lead | Sungkar Bolat | Filtering, re-referencing, artifact handling, quality control (2, 3) | Data and provenance (1) | 4-5: filtering, artifact handling |
| Decoding lead | Beibars Ybraiakhyn | Feature extraction, classifier pipeline, cross-validation (5, 6) | Signal processing (2, 3) | 6-7: stage 4 check, features and evaluation |
| Evaluation and reporting lead | Ramazan Zypayev | Neurophysiological check, chance levels, statistics, figures, report integration (4, 7, 9) | Decoding (5, 6) | 8-11: results, exploratory, limits, conclusion |

## Stage ownership and review

Review = re-run the stage script on SMOKE, read the matching `NOTES.md` section, then fill in the date.

| Stage | Lead | Reviewer | Review date |
| --- | --- | --- | --- |
| 1. Data audit | Chidiadi Bethel Mba | Sungkar Bolat | |
| 2. Filtering | Sungkar Bolat | Beibars Ybraiakhyn | |
| 3. Artifact handling | Sungkar Bolat | Beibars Ybraiakhyn | |
| 4. Neurophysiological check | Ramazan Zypayev | Chidiadi Bethel Mba | |
| 5-6. Features, classifiers, evaluation | Beibars Ybraiakhyn | Ramazan Zypayev | |
| 7. Hypothesis tests | Ramazan Zypayev | Chidiadi Bethel Mba | |

## Remaining tasks

Fill in the date when the task is done.

| Task | Owner | Date done |
| --- | --- | --- |
| Confirm EOG electrode orientation (Brunner 2008, Fig. 3) | Chidiadi Bethel Mba | |
| Fresh Colab run, record stage timings (30 min limit) | Chidiadi Bethel Mba | |
| Re-pin `requirements.txt` from Colab versions | Chidiadi Bethel Mba | |
| Second independent Colab run, compare `stats_h*.csv` | Sungkar Bolat | |
| Inspect excluded ICA components in all 18 recordings | Sungkar Bolat | |
| Confirm artifact-prone channel set, decide on optional CSP extension | Beibars Ybraiakhyn | |
| Compile report and slides PDF in Overleaf | Beibars Ybraiakhyn | |
| Open and verify the 5 web-checked references, add missing DOIs | Ramazan Zypayev | |
| Extend report to 6-8 pages, final read for style rules | Ramazan Zypayev | |
| Full rehearsal of the 12-minute talk | All | |

## Log

| Date | Member | Phase | What |
| --- | --- | --- | --- |
| 2026-10-27 | Beibars Ybraiakhyn | 0-9 | Initial implementation of all stages (`src/`, `scripts/`, tests), FULL run, executed notebook, report and slide drafts |
| 2026-10-31 | Beibars Ybraiakhyn | team | Roles and responsibilities, study guide per member in `team/` (RU + EN), 12-minute defense script |
| 2026-11-01 | Beibars Ybraiakhyn | 8-9 | Author names, affiliation and course name in report, slides and notebook; GitHub repository |
