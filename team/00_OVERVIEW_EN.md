# For everyone: the project in 5 minutes

All four of us read this file. At the defense anyone can be asked about any part, so everyone needs the big picture. Your own part is in your personal file in this folder:

- `Bethel_EN.md` — data and reproducibility
- `Ramazan_EN.md` — evaluation, statistics, report and slides
- `Beibars_EN.md` — features and classifiers
- `Sungkar_EN.md` — filtering and artifact cleaning

Russian versions have the same names without `_EN` (the overview is `00_ОБЩЕЕ_для_всех.md`).

Roles go from the most approachable to the most technical: Bethel, Ramazan, Beibars, Sungkar.

---

## 1. The project in one paragraph

The dataset is **BCI Competition IV 2a**. 9 people sit in front of a screen. An **arrow** appears: left, right, down or up. Following the arrow, the person **imagines** moving the left hand, right hand, feet or tongue, without actually moving. We record the brain (EEG) and train a computer to guess what was imagined. This is a brain–computer interface (BCI).

**The suspicion:** the arrow points toward the task. The eyes **reflexively** look where the arrow points, and the electrodes around the eyes (EOG) record this. On top of that, eye movement leaks a little into the EEG. The question is whether the computer guesses the class **from the eyes instead of the brain**. If it does, such a BCI is useless for a paralysed person who cannot move the eyes, and all published accuracies on this dataset are inflated.

## 2. Three hypotheses (written down BEFORE the main analysis)

| | In plain words | Result |
| --- | --- | --- |
| **H1** | The 3 eye channels (EOG) alone can tell left from right hand better than chance | **Supported.** AUC 0.86, 8 of 9 people above chance (p = 0.002) |
| **H2** | Removing the eyes from the EEG lowers EEG accuracy, more so in people whose eyes "give more hints" | **Not supported.** Accuracy barely changed (p = 0.63) |
| **H3** | After cleaning, the classifier relies on the proper motor channels and rhythms instead of frontal, suspicious ones | **Not supported.** Small, non-significant shift (p = 0.75) |

**Main conclusion:** the eyes really do follow the arrow and predict the class very well on their own. But the standard classifier based on **rhythm power** (band power) in the motor-imagery window does **not** rely on them. The risk is in methods that look at the **shape of the signal over time** at frontal electrodes. We checked this with an extra analysis that was not planned in advance (exploratory).

> **A negative result is fine.** H2 and H3 were not supported, and that is an honest scientific result. Marks are not lost for it. Tuning the analysis until a hypothesis "works" would be bad. We did not do that: the decision rules were written in `PLAN.md` in advance.

## 3. The seven stages and who is responsible

| Stage | What we do | Lead | Reviewer |
| --- | --- | --- | --- |
| 1. Data audit | Load the data, check every fact | Bethel | Sungkar |
| 2. Filtering | Remove drift and 50 Hz power-line noise | Sungkar | Beibars |
| 3. Artifacts | Three versions of the data: `none`, `eog_reg`, `ica` | Sungkar | Beibars |
| 4. Neurophysiological check | Plots: is the brain signal there, is the eye signal there | Ramazan | Bethel |
| 5. Features and classifier | What we feed the classifier | Beibars | Ramazan |
| 6. Honest evaluation | Cross-validation, chance level | Beibars | Ramazan |
| 7. Hypothesis tests | Statistics for H1–H3 | Ramazan | Bethel |
| + Colab notebook | Everything runs with one button | Bethel | — |
| + Report and slides | Overleaf, 6–8 pages | Ramazan | everyone |

## 4. Speaking order (12 minutes)

| Slides | Speaker | Time |
| --- | --- | --- |
| Title, "Question and hypotheses", "Stage 1 Data audit" | Bethel | ~3:05 |
| "Stage 2 Signal inspection", "Stage 3 Artifact handling" | Sungkar | ~2:35 |
| "Stage 4 The confound exists", "Stage 5–6 Features, classifiers" | Beibars | ~2:50 |
| "Stage 7 H1", "Stage 7 H2 and H3", "Exploratory", "Limits and take-home" | Ramazan | ~2:35 |

Ramazan leads stage 4, but Beibars presents it so that everyone speaks for about the same time.

The slides are in `slide_pdf/slides/slides.tex` and are compiled in Overleaf.

**Full talk text:** `SCRIPT_defense_EN.md` (we speak from it). The "What you say at the defense" section in each personal file is a short cheat sheet for the same text.

## 5. Glossary (the essential words)

- **EEG**: electrodes on the head that record the brain's electrical activity. We have 22.
- **EOG**: 3 electrodes around the eyes. They record eye movements and blinks. EOG1 next to the left eye, EOG2 above the nose, EOG3 next to the right eye.
- **Trial**: one attempt, an arrow and then 4 seconds of imagined movement. A session has 288 trials (72 per class).
- **Run**: a block of 48 trials. A session has 6 runs. Each person has 2 sessions on different days, `0train` and `1test`.
- **Artifact**: anything in the signal that is not the brain, such as blinks, eye movements, muscle tension, power-line noise.
- **ERD**: when someone imagines moving a hand, brain rhythms (mu 8–13 Hz, beta 13–30 Hz) **weaken** over the opposite hemisphere. The left hand weakens the right side (electrode C4), the right hand the left side (C3). This is the "proper" brain signal for a BCI.
- **AUC**: classifier quality from 0 to 1. 0.5 is random guessing, 1.0 is perfect. Better than accuracy because it does not depend on a threshold.
- **Chance level, permutation test**: we randomly shuffle the class labels and retrain the classifier many times. This shows what AUC "pure luck" gives. If the real AUC is above 95% of those, the result is not luck.
- **Leave-one-run-out cross-validation**: train on 5 runs, test on the 6th, repeat 6 times. The test is always on data the classifier has not seen.
- **Data leakage**: when information from test data accidentally gets into training. It makes results unfairly high. We have an automated test that checks there is no leakage.
- **Wilcoxon**: a statistical test for small samples (we have 9 people). It does not need a normal distribution.

## 6. Questions ANY of us may get

**"Why does this project matter?"**
BCIs are built for paralysed people. If a classifier works on eye movements, it will not work for such a person. Dataset 2a is the most widely used one for this task, so the check matters.

**"Why was H2 not supported if the eyes predict the class so well?"**
An eye movement is a short jump in the signal about 0.2–0.4 s after the arrow. It is an event in **time**. Our EEG features are **rhythm power** (band power) in the 0.5–2.5 s window. Rhythm power barely notices that jump, so removing the eyes does not change it.

**"So all published results on 2a are fine?"**
No, we cannot say that. We tested only band-power classifiers. Methods that look at the signal shape at frontal electrodes, or use the EOG channels, may rely on the eyes. Our exploratory analysis shows this. Our recommendation is to report what the eyes alone achieve next to any result on 2a.

**"Only 9 people? That is few."**
The dataset has exactly 9 people. The report says plainly that small effects can be missed with n = 9. This is a limitation.

**"Did you cheat? Did test data get into training?"**
No, and it is checked:
1. The cleaning (regression and ICA) is trained on **separate eye calibration recordings**, not on the trials.
2. The classifier is trained on 5 runs only and tested on the 6th.
3. `tests/test_smoke.py` has an automated leakage test.

**"Why not CSP, neural networks, SVM?"**
The assignment allows only methods from course notebooks 00–06. CSP is allowed only as an optional extension after the main result.

**"Can your numbers be reproduced?"**
Yes. One notebook, `notebooks/final.ipynb`, does everything from scratch. We ran it twice and the results matched to the last digit. Every number in the report is filled in automatically from `results/tables/`, none is typed by hand.

**"Did the subjects cheat by moving their eyes?"**
No. It is a reflex to the arrow, not an attempt to cheat. The subjects did nothing wrong.

## 7. Where things are

- `PLAN.md` — the original plan and hypotheses.
- `NOTES.md` — **every** decision and why it was made. The main file for preparing answers.
- `results/figures/` — all figures. The name starts with the stage number: `fig_3...` is stage 3.
- `results/tables/` — all numbers.
- `notebooks/final.ipynb` — the notebook for submission.
- `slide_pdf/report/main.tex` — the report.
- `slide_pdf/slides/slides.tex` — the slides.
- `CONTRIBUTIONS.md` — who did what. Add a dated line whenever you do or review something.
