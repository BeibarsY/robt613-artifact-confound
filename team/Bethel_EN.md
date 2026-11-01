# Bethel — Data and reproducibility

**Role:** Data and provenance lead
**Stages:** 1 (data audit) and the Colab notebook (stage 8 in `PLAN.md`)
**You review:** stages 4 and 7 (Ramazan)
**At the defense:** you open the presentation, about 3 minutes

Russian version: `Bethel.md`. The full overview is `00_OVERVIEW_EN.md`. Section 0 below repeats its key points.

This is the most approachable part of the project. There is almost no heavy maths. The job is to check facts carefully and run the notebook cleanly.

---

## 0. The project in short (from the shared overview)

**Dataset:** BCI Competition IV 2a. 9 people see an **arrow** (left, right, down, up) and **imagine** moving the left hand, right hand, feet or tongue. We record the brain (EEG, 22 electrodes) and train a computer to guess the imagined movement.

**The suspicion:** the arrow points toward the task, and the eyes **reflexively** follow it. Three electrodes around the eyes (EOG) record this, and some of it leaks into the EEG. Does the classifier guess from the eyes instead of the brain? If so, it would fail for a paralysed user who cannot move the eyes.

| Hypothesis | Plain words | Result |
| --- | --- | --- |
| **H1** | The 3 eye channels alone can tell left from right hand better than chance | **Supported.** AUC 0.86, 8 of 9 people above chance (p = 0.002) |
| **H2** | Removing the eyes from the EEG lowers EEG accuracy | **Not supported.** Accuracy barely changes (p = 0.63) |
| **H3** | After cleaning, the classifier relies on proper motor features | **Not supported.** Small, non-significant shift (p = 0.75) |

**Main conclusion:** the eyes really do follow the arrow and predict the class very well on their own. But the standard classifier based on **rhythm power** (band power) does not rely on them. A negative result for H2 and H3 is fine: the decision rules were written in advance, and we report honestly.

**Speaking order (12 min):**

| Who | What | Time |
| --- | --- | --- |
| **Bethel** | title, question and hypotheses, stage 1 | ~3.1 min |
| Sungkar | stages 2–3 (filtering, cleaning) | ~2.6 min |
| Beibars | stage 4 (plots before any classifier) and stages 5–6 (features, evaluation) | ~2.8 min |
| Ramazan | results, exploratory, limits, conclusion | ~2.6 min |

**Glossary:**
- **EEG**: brain electrodes (22).
- **EOG**: 3 electrodes around the eyes. EOG1 next to the left eye, EOG2 above the nose, EOG3 next to the right eye.
- **Trial**: one arrow plus 4 s of imagined movement. 288 per session.
- **Run**: a block of 48 trials. 6 runs per session.
- **Session**: 2 per person, on different days, named `0train` and `1test`.
- **AUC**: 0.5 = guessing, 1.0 = perfect.

## 1. Your part in plain words

Your part is the **foundation**. If we misread the data (wrong units, wrong arrow time, wrong electrodes), everything else is wrong, even with perfect code. So we did not take any fact on trust. Each one was checked by loading the data or by opening the dataset description paper (Brunner 2008, in `slide_pdf/report/refs/Brunner2008_desc_2a.pdf`).

The second half is **reproducibility**: anyone presses "Run all" in Colab and gets exactly the same numbers.

## 2. What we learned about the data (and how we checked)

| Fact | Value | How we checked |
| --- | --- | --- |
| Recordings | 9 people × 2 sessions = **18 recordings** | loaded all of them |
| Trials per session | **288** (6 runs of 48, 72 per class) | counted |
| Channels | 22 EEG + 3 EOG, **250 Hz** | from the files |
| Units | volts (MOABB converts from microvolts) | signal spread about 0.00001, so volts |
| Reference electrode | left mastoid (behind the ear) | Brunner 2008 paper |
| Recording filter | 0.5–100 Hz + 50 Hz notch | paper. **The NEMAR README says 0.05–200 Hz, which is wrong.** We trust the paper |

### The most important check: where the time marker sits

In the original files the marker is at the **start of the trial** (a fixation cross appears). The arrow appears **2 seconds later**. The MOABB library shifts the marker by those 2 seconds, so **the marker sits exactly on the arrow**. We checked this for **every one** of the 5184 trials: the shift is always exactly 500 samples (2.0 s).

A second, independent check (stage 4): the eyes start moving **0.19 s** after the marker, never before it. Nobody can look at an arrow before it appears, so the marker is right.

### What MOABB drops, and we recovered

When loading, MOABB throws away two useful things. We read them directly from the original `.mat` files (code in `src/data.py`):

1. **Expert artifact flags.** A specialist marked "dirty" trials by hand: 5 to 73 per session, 488 in total (9.4%). We do **not** remove them. We keep them as a separate column and use them for a robustness check.
2. **Eye calibration recordings.** At the start of each session the person looked with eyes open, with eyes closed, and moved the eyes. Stage 3 trains the eye cleaning **on these recordings**, so test trials never enter the cleaning. Subject A04T has only one such recording (80 seconds), as stated in the paper.

## 3. Your tasks (step by step)

### Task 1. Get the project and run it on your computer
1. Beibars will send the GitHub repository link. Download it (`git clone <link>`).
2. In the project folder, set up the environment (once):
   ```bash
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```
3. Check that it works (about a minute, it downloads one person's data):
   ```bash
   ROBT613_MODE=SMOKE .venv/bin/python scripts/01_audit.py
   ```
   At the end you should see a table with 288 trials. If there is an error, post it in the group chat.

### Task 2. Confirm which eye electrode is where
1. Open the paper (https://www.bbci.de/competition/iv/desc_2a.pdf; locally it is `slide_pdf/report/refs/Brunner2008_desc_2a.pdf`), page 2, Figure 3 on the right (the face).
2. In the figure, #1 is on the viewer's right, #2 above the nose, #3 on the viewer's left.
3. We assumed the face is drawn **facing us**. Then EOG1 is next to the person's **left** eye and EOG3 next to the **right** eye.
4. If you agree, write "confirmed, Bethel, date" in `NOTES.md` under TODO-verify.

This only affects the **sign** (whether the eye moves left or right). It does not change the finding that the eyes move differently for different classes.

### Task 3. Run the notebook on Colab (main task)
1. Upload `notebooks/final.ipynb` to Google Colab.
2. Menu Runtime → "Disconnect and delete runtime" (clean start).
3. Runtime → "Run all".
4. Wait. The data download takes about 9 minutes, and by our estimate the computation takes another 13–17 minutes.
5. The last cell prints the time of each stage and the total. **Write it down.**
6. **Limit: 30 minutes.** If it is over, in the `config.py` cell change `"n_permutations_other": 30` to 20. This does not change the numbers for the main hypothesis H1.
7. Save the notebook with outputs (File → Save) and get the share link for submission.

### Task 4. Library versions
The last notebook cell prints the library versions on Colab. Copy them into `requirements.txt`.

### Task 5. A second run by someone else
Ask a teammate to repeat Task 3, then compare `results/tables/stats_h1.csv`, `stats_h2.csv` and `stats_h3.csv`. The numbers must match. The plan requires this.

### Task 6. Report section
In `slide_pdf/report/main.tex`, read the "Dataset and data audit" section and improve the wording if you like. **Never type numbers by hand.** They come from `slide_pdf/report/numbers.tex` (commands like `\NTrialsSession`).

### Task 7. Review stages 4 and 7 (Ramazan)
```bash
ROBT613_MODE=SMOKE .venv/bin/python scripts/04_neurophys_check.py
ROBT613_MODE=SMOKE .venv/bin/python scripts/06_stats.py
```
SMOKE mode uses only one person, so it is quick. Read "Phase 4 findings" and "Phase 7 results" in `NOTES.md`, then add a line like `| date | Bethel | 4,7 review | ... |` to `CONTRIBUTIONS.md`.

## 4. What you say at the defense (about 3 minutes)

**Slide 1 (title):** "We tested whether a classifier on the most popular BCI dataset guesses the class from eye movements instead of the brain."

**Slide 2 (Question and hypotheses):**
- "In dataset 2a the arrow points toward the task: left for the left hand, right for the right hand. The eyes reflexively look there too."
- "We wrote down three hypotheses and the rules for deciding them before running the analysis." Name H1, H2 and H3 briefly (table in section 0).
- "The rules were fixed before the analysis, so we could not tune the analysis to the result."

**Slide 3 (Stage 1 Data audit):**
- "18 recordings, 288 trials each, 22 EEG and 3 eye channels."
- "We took no fact on trust. Most importantly, we checked that the time marker sits exactly on the arrow, for every one of the 5184 trials."
- "MOABB drops the expert artifact flags and the eye calibration recordings. We recovered both from the original files. The calibration recordings are used in stage 3."
- "The whole project is one notebook. We ran it twice and the numbers matched to the last digit." Hand over to Sungkar.

## 5. Questions you may get

**"How did you check that the marker is on the arrow?"**
Two ways. First, we compared it with the original trigger: the shift is exactly 2.0 s for every trial. Second, the eyes start moving 0.19 s after the marker and never before it. The eyes cannot react to the arrow before it appears.

**"Why didn't you remove the trials the expert marked as dirty?"**
Those trials may contain exactly the eye movements we are studying. Removing them would hide what we are looking for. Instead we repeated the analysis without them, and the conclusions did not change (H1 supported, H2 not).

**"Why do the README and your report disagree about the filter?"**
The primary source is the dataset authors' paper (Brunner 2008). The NEMAR README is a secondary summary, and it contains an error (0.05–200 Hz instead of 0.5–100 Hz).

**"How did you ensure reproducibility?"**
- A fixed seed (42) everywhere.
- All code in functions in `src/`, no copy-paste.
- One notebook runs everything from scratch.
- Every number in the report is read automatically from the results tables.
- Two independent runs gave identical numbers.

**"How long does the notebook take?"**
Give the time you measure on Colab in Task 3. Locally it takes about 6 minutes with the data already downloaded.

**"Why the calibration recordings?"**
The eye cleaning has to be trained on something. If it were trained on the same trials we later test, the test would have "seen" the training. The calibration recordings are separate minutes at the start of the session where the person moves the eyes on purpose. Training on them is cleaner, and that is exactly why the authors recorded them.

## 6. Your files
- `src/data.py` — loading, trial table, expert flags, calibration recordings.
- `scripts/01_audit.py` — the audit.
- `results/tables/audit_*.csv` — audit results.
- `results/figures/fig_1a_raw_segment.png` — 10 seconds of raw signal. Blinks are visible on EOG2.
- `results/figures/fig_1b_sensor_layout.png` — electrode layout.
- `scripts/build_notebook.py` — builds the notebook.
- `notebooks/final.ipynb` — the notebook.
