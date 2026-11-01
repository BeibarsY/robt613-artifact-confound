# Beibars — Decoding

**Role:** Decoding lead
**Stages:** 5 (features and classifier) and 6 (honest evaluation)
**You review:** stages 2–3 (Sungkar)
**At the defense:** you speak third, two slides in a row — stage 4 and stages 5–6, about 2 minutes 50 seconds

Read `00_OVERVIEW_EN.md` first. It covers the whole project. Russian version: `Beibars.md`.

---

## 1. Your part in plain words

A classifier cannot read a signal directly. It needs **numbers (features)**. Your part answers three questions:
1. **Which numbers** we take from each trial.
2. **What** guesses the class.
3. **How to check honestly** that it is guessing and not memorising.

## 2. Features: which numbers we take

### Rhythm power (band power)
We take the signal **0.5–2.5 s after the arrow**, when the person is already imagining the movement. For each electrode we compute how much "energy" is in each frequency band (Welch method) and take the logarithm.

| Band | Frequencies | What is there |
| --- | --- | --- |
| delta | 1–4 Hz | slow waves and **eye movements** |
| theta | 4–8 Hz | |
| mu | 8–13 Hz | **motor rhythm**, weakens when a hand movement is imagined |
| beta | 13–30 Hz | **motor rhythm** |
| high | 30–45 Hz | **muscles**, sometimes brain |

### Amplitude over time
We take **0–1 s after the arrow**, cut it into 10 pieces of 100 ms, and take the mean signal in each. This catches a **glance to the side**: a "jump" in the signal, not a rhythm.

### Four feature sets (the main "switch" of the experiment)

| Set | Contents | Why |
| --- | --- | --- |
| `eog_only` | 3 eye channels: amplitudes + delta/theta (36 numbers) | How much the eyes alone "know" (H1) |
| `eeg_all` | All 22 EEG electrodes × 5 bands (110 numbers) | The standard classifier |
| `eeg_physio` | C3, Cz, C4, FC3, FC4, CP3, CP4 × mu + beta (14 numbers) | The "proper" brain signal |
| `eeg_artifact_prone` | Fz, FC1–FC4, C5, C6 × delta + high (14 numbers) | The "suspicious" signal: forehead, edges, eye and muscle bands |

**A deviation from the plan that you need to confirm:** the plan proposed the temporal electrodes T7/T8/P7/P8 for the suspicious set. **This dataset does not have them** (`fig_1b`). We took the most frontal and most lateral electrodes that exist: Fz, FC1–FC4, C5, C6.

### Why features can be computed in advance
Each trial is computed **on its own**. Nothing "learns" from other trials. So computing features once before cross-validation is honest. What **does** learn (the StandardScaler normalisation and the classifier itself) is trained only inside the training part. An automated test checks this.

## 3. Classifiers

- **Logistic regression (L2)** — the main one. It draws a "boundary" between classes.
- **LDA with shrinkage** — a backup, to check that the conclusion does not depend on the classifier.
- A **StandardScaler** always comes first. It puts all features on the same scale.

**The parameter C = 1 is fixed.** We checked: tuning C honestly with nested cross-validation changes AUC by only **0.007** on average. Tuning C inside each of thousands of permutations would not fit into 30 minutes on Colab.

## 4. Honest evaluation

### Leave-one-run-out cross-validation
A session has 6 runs (blocks). Train on 5, test on the 6th, repeat 6 times. **Why not shuffle trials randomly:** neighbouring trials in one run are similar (fatigue, electrode position). With random shuffling the test is almost a copy of the training, and the result is inflated. The plan forbids it.

### Chance level (permutation test)
- We shuffle the labels **within each run** (so each run keeps the same number of trials per class), then train and test the same way.
- Repeat many times and see what AUC "pure luck" gives.
- The 95th percentile of those random AUCs is the **chance threshold**. Every AUC in our table sits next to its own threshold.
- **100 permutations** for the rows used to decide H1, **30** for the rest. Otherwise we would not fit into 30 minutes on Colab. The plan allows this.

### Two more checks
- **Across sessions:** train on day 1 (`0train`), test on day 2 (`1test`). An independent estimate.
- **4 classes** (left, right, feet, tongue): logistic regression, balanced accuracy, chance level about 0.29.

## 5. Results (left vs right hand, session 0train, median of 9 people)

| Feature set | no cleaning | after regression | after ICA | above chance |
| --- | --- | --- | --- | --- |
| `eog_only` (eyes only) | **0.86** | — | — | 8 of 9 |
| `eeg_all` (all EEG) | 0.85 | 0.83 | 0.80 | 8 of 9 |
| `eeg_physio` (motor) | 0.77 | 0.77 | 0.77 | 8 of 9 |
| `eeg_artifact_prone` | 0.57 | 0.58 | 0.56 | 3 of 9 |

How to read it:
- **The eyes alone guess as well as the whole brain** (0.86 vs 0.85).
- Cleaning changes almost nothing. The differences between columns are not statistically significant.
- The "suspicious" set barely works, so frontal slow waves and muscle bands carry almost no class information.
- For S5 the eyes alone give **AUC 1.00** (perfect).

Per person: `fig_6a`. For 4 classes (`fig_6b`): all EEG 0.59, eyes only 0.56, chance 0.29. The eyes also "know" about feet and tongue, because the down and up arrows move the eyes too.

### Exploratory analysis (not planned in advance, always say so)
After the H2 result we added one set: **amplitude over time** at the frontal electrodes Fz, FC1–FC4. Without cleaning AUC is 0.70. After cleaning the median barely changed, **but** the better the eyes alone work for a person, the more this classifier drops for them after cleaning (correlation 0.98 and 0.85). So **time-domain** frontal features do use the eyes, and rhythm power does not. Ramazan presents the "Exploratory" slide, but you may get questions about it.

## 6. Your tasks (step by step)

1. **Read** "Phase 5-6 design" in `NOTES.md`.
2. **Look at** `fig_6a` and `fig_6b` in `results/figures/` and the table `results/tables/auc_per_subject.csv`.
3. **Confirm or challenge** the channel choice for `eeg_artifact_prone` (Fz, FC1–FC4, C5, C6). Write the decision in `NOTES.md`.
4. **Decide about CSP.** CSP is an optional extension worth +5% of the grade. Rules:
   - only after the main result is finished (it is);
   - only as a comparison with our result, not instead of it;
   - only if Colab still fits into 30 minutes (Bethel will tell us after the run).
   If we do it, it is your part: a separate CSP + LDA feature set on the same 6 folds with the same chance level. If not, write one line in `NOTES.md` explaining why.
5. **Report section:** in `slide_pdf/report/main.tex` read "Features and classifiers" and "Evaluation". Do not type numbers by hand.
6. **Review stages 2–3 (Sungkar):**
   ```bash
   ROBT613_MODE=SMOKE .venv/bin/python scripts/03_artifacts.py
   ```
   Read the Phase 2 and Phase 3 sections in `NOTES.md` and add a dated line to `CONTRIBUTIONS.md`.

## 7. What you say at the defense (about 2 minutes 50 seconds)

Full text: `SCRIPT_defense_EN.md`, slides 6 and 7. Below is a cheat sheet.

### First, understand stage 4 (Ramazan leads it, you present it)
Stage 4 is a check **before** any classifier: do the data contain (a) the normal brain signal and (b) the suspicious eye signal? Without them, no classifier result would prove anything.
- **Brain signal (ERD).** When a person imagines moving the **left** hand, mu and beta rhythms **weaken** over the **right** hemisphere (electrode C4), and the reverse for the right hand. After ICA this goes the expected way in **all 18 recordings** (`fig_4a`, `fig_4b`). S2, S5 and S7 have a weak signal. We keep them.
- **Eye signal (key figure `fig_4c`).** The top-left plot is the horizontal eye signal (HEOG). The **blue** line is the left arrow: the eyes go left (line goes up). The **red** line is the right arrow: the eyes go right (line goes down). The down (feet) and up (tongue) arrows give matching movements on the vertical signal (VEOG). The bottom row is the "left minus right" difference for each person.
- **Numbers:** the movement starts **0.19 s** after the arrow (normal eye reaction time), peaks at 0.37 s, nothing before the arrow. The difference goes the expected way in **17 of 18** recordings, highly significant (p < 0.001) in 15.

### Slide "Stage 4 The confound exists before any classifier" (about 1 minute 15 seconds)
- "Before training any classifier, we checked on plots that the data contain both the brain signal and the eye signal."
- "The brain signal is there. Imagining the left hand weakens the rhythm over the right hemisphere, and vice versa, in all 18 recordings."
- Point at the top-left plot: "Blue is the left arrow, the eyes go left. Red is the right arrow. It starts 0.19 seconds after the arrow, a normal eye reaction."
- Point at the bottom row: "We see this in 17 of 18 recordings. So the suspicion is confirmed before any classifier." No hand-over, go straight to the next slide: "Now, how we decoded."

### Slide "Stage 5–6 Features, classifiers, honest evaluation" (about 1 minute 35 seconds)
- "From each trial we take two kinds of features: rhythm power in the imagery window and signal amplitude over time right after the arrow."
- "We build four sets: eyes only, all EEG, motor channels and rhythms, suspicious frontal channels and bands."
- "The classifier is logistic regression, with LDA as a check."
- "The evaluation is honest: we train on 5 blocks and test on the 6th. The chance level for every row comes from shuffling labels. An automated test checks that test data never entered training."
- Point at `fig_6a`: "The eyes alone, in the first panel, guess as well as the whole EEG for most people. And cleaning, the orange and green bars, changes almost nothing." Hand over to Ramazan.

## 8. Questions you may get

**"How do you know it is eye movement and not a brain signal?"** (stage 4)
It is recorded by electrodes around the eyes, it is huge compared with EEG, and its direction matches the arrow: left for the left arrow, right for the right arrow, up and down for the vertical arrows. It starts at 0.19 s, a typical eye reaction time.

**"What is ERD and why check it?"** (stage 4)
ERD is the weakening of mu and beta rhythms over the opposite hemisphere when a hand movement is imagined. It is the "proper" brain signal for a BCI. We checked that it is there, otherwise comparing "brain versus eyes" would make no sense.

**"Why is ERD weak in some people?"** (stage 4)
This happens in every BCI study (it is called BCI inefficiency). S2, S5 and S7 have weak ERD. We flagged them but kept them, because removing people with poor results would be cherry-picking.

**"Why AUC and not accuracy?"**
AUC does not depend on a threshold and is equally clear for any class balance: 0.5 is guessing, 1.0 is perfect. For 4 classes we report balanced accuracy and one-vs-rest AUC.

**"Why leave-one-run-out and not a random 80/20 split?"**
Neighbouring trials are similar. With a random split the test "peeks" at the training data and accuracy is inflated. Whole blocks are more honest, and the assignment requires it.

**"What is a permutation test?"**
We shuffle the labels so there is surely no link between signal and class, and see what AUC comes out "by itself". We repeat this 100 times. If our real AUC is above 95% of those random ones, it is not chance.

**"Why shuffle within runs?"**
So each run keeps the same number of trials per class as in the real data. Otherwise the chance level would be computed under different conditions.

**"Why didn't you tune C?"**
We checked: honest tuning changes AUC by 0.007 on average, which is nothing. Tuning inside every permutation would take many times longer than Colab allows.

**"Features computed before cross-validation — isn't that leakage?"**
No. Each trial is computed separately, with no information from other trials. Everything that learns (normalisation and classifier) is trained only on the training folds. The test `test_no_leakage_in_cv` checks this.

**"Why is the artifact_prone AUC below 0.5 for some people?"**
With random guessing, AUC fluctuates around 0.5 in both directions. Values slightly below 0.5 are within random variation, not "reverse" guessing.

**"Why not CSP?"**
The assignment allows the main result only with course methods from notebooks 00–06. CSP is an optional extension (see task 4).

**"Why only logistic regression for 4 classes?"**
That is what the plan says (multinomial logistic regression). We use LDA as a check on the left-versus-right task.

## 9. Your files
- `src/features.py` — features and feature sets.
- `src/decode.py` — classifiers, cross-validation, permutations.
- `scripts/05_decode_all_conditions.py`.
- `results/tables/auc_*.csv`, `decode_*.csv`.
- Figures `fig_6a`, `fig_6b`.
- `tests/test_smoke.py` — leakage tests.
