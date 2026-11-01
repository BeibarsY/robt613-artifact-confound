# Sungkar — Signal processing

**Role:** Signal processing lead
**Stages:** 2 (filtering) and 3 (artifacts and quality control)
**You review:** stage 1 (Bethel)
**At the defense:** you speak second, about 2.5 minutes

Read `00_OVERVIEW_EN.md` first. It covers the whole project. Russian version: `Sungkar.md`.

---

## 1. Your part in plain words

Raw EEG is dirty. It contains:
- **slow drift** — the signal "floats" up and down over seconds (sweat, electrodes);
- **50 Hz power-line noise** — sockets radiate, electrodes pick it up;
- **blinks and eye movements** — the eye works like a small battery, and when it moves it changes the voltage on the forehead.

Your part turns this into clean data, in **three versions**. Comparing these three versions is the main experiment of the project.

| Version | What we do | Analogy |
| --- | --- | --- |
| `none` | 1–45 Hz filter + CAR | Noise removed, the eyes are still there |
| `eog_reg` | `none` + subtract the eye channels from each EEG channel | We know how much "eye" reaches each electrode and subtract it |
| `ica` | `none` + split the signal into 20 sources, drop the eye ones | Like isolating one voice from a recording of a noisy room |

## 2. Stage 2: filtering

**What we did:** a **1–45 Hz** band-pass filter (FIR, 825 coefficients, 3.3 s).

**Why 1 Hz at the bottom:** all 18 recordings have strong drift below 1 Hz in the spectrum (`fig_2a`). ICA also works poorly on drift.

**Why 45 Hz at the top:** above it are power-line and muscle noise, and the rhythms we need (mu 8–13, beta 13–30 Hz) are below it.

**A detail you need to be able to explain:** a filter has a "transition band" where it goes smoothly from "pass" to "block". MNE's default for 45 Hz is very wide, so at **50 Hz** the filter reduces the noise by only **−4.5 dB**, which is almost nothing. We narrowed the transition band to 5 Hz, and at 50 Hz it became **−51 dB**: the noise is suppressed hundreds of times. The filter length stayed the same.

**So a separate 50 Hz notch filter is not needed.** Some recordings have leftover line noise up to +20 dB (S5, session 1test, `fig_2c`), but our filter removes it anyway.

**No resampling:** 250 Hz is already modest, and the data are small.

**A finding:** S1 and S9 have a narrow peak in the 30–45 Hz band. S1's (36 Hz, electrode FC4) does **not** depend on the class, it is just noise. S9's (31–32 Hz, electrode C4) depends **strongly** on the class and sits over the motor area, so it is probably a real brain rhythm (high beta). Conclusion: the 30–45 Hz band is not pure "junk".

## 3. Stage 3: artifacts

### CAR (common average reference)
Originally all electrodes are recorded relative to one electrode **behind the left ear**. Because of that, electrode C5 (closest to the left ear) shows artificially **low** power, so the left and right sides of the head are not symmetric (`fig_3d`). That is bad for a left-versus-right task. CAR subtracts the mean of all electrodes from each one, and the asymmetry disappears.

### The key decision: what the cleaning is trained on
Regression and ICA must be **trained** on something. The plan suggested training on the trials themselves. But then test trials enter the training, which breaks the rule "never train on test data".

**We train the cleaning on separate calibration recordings** (a few minutes at the start of each session where the person deliberately blinks and moves the eyes), then apply it to the trials. The course (notebook 04) teaches exactly this. The dataset authors recorded those minutes for this purpose.

### ICA: the first attempt failed, and you need to be able to tell this
- The plan said: number of components = enough for 99% of variance, exclude by MNE's standard rule (z-score ≥ 3).
- That gave 9–14 components. With 10 components the **largest possible** z-score is (10−1)/√10 = **2.85**, so it **mathematically cannot reach** 3. In 9 of 18 recordings ICA removed nothing.
- **The fix:** use **20 components** (after CAR no more than 21 are possible) and remove a component if its correlation with any eye channel is **|r| ≥ 0.5**.
- Why 0.5: across all 360 components the correlations split into two groups, with nothing between ~0.25 and ~0.55 (`fig_3f`). The 0.5 threshold falls into that gap.
- Result: **2 components** removed in 17 recordings and 3 in one (S5, 1test). In `fig_3a` one component is a frontal blob (blinks), the other is "left-right" (horizontal eye movements).

### Does the cleaning work
Correlation of electrode Fz with eye channel EOG2 (median over 18 recordings):

| `none` | `eog_reg` | `ica` |
| --- | --- | --- |
| 0.73 | **0.08** | **0.15** |

This is computed on trials the cleaning did **not** see during training, so the cleaning transfers to new data.

`fig_3b` shows it clearly: a blink gives +50 µV at Fz in `none`, and in `eog_reg` and `ica` it is gone, while the rest of the signal is untouched.

### Why we do not reject "dirty" trials
- The largest signal range in any trial is 142 µV. The usual threshold is 150 µV, so it would reject nothing (`fig_3c`).
- The histogram has a bump at 70–80 µV. It **shrinks** after cleaning, so it is the eyes. Rejecting those trials would remove exactly what we study.
- All three versions use **the same 288 trials**. An automated test checks this.

### A regression trap we found
If an eye electrode records a non-eye jump (for example the electrode shifted, −900 µV, `fig_3b2`), regression **copies** it into the EEG. This happened in 6 trials (up to 395 µV). The expert had flagged all 6 as dirty, so we also repeated the analysis without flagged trials. The conclusions did not change.

## 4. Your tasks (step by step)

1. **Read** "Phase 2 decisions" and "Phase 3 decisions" in `NOTES.md`.
2. **Look at** `fig_2a`–`fig_2c` and `fig_3a`–`fig_3f` in `results/figures/`.
3. **Check ICA in all 18 recordings.** Right now the maps are drawn only for S1. Draw and inspect the rest:
   ```python
   import sys; sys.path.insert(0, ".")
   import mne
   for s in range(1, 10):
       for sess in ("0train", "1test"):
           ica = mne.preprocessing.read_ica(f"results/cache/ica/S{s:02d}_{sess}_calibration-ica.fif")
           fig = ica.plot_components(picks=ica.exclude, show=False)
           fig.savefig(f"/tmp/ica_S{s}_{sess}.png")
   ```
   For each removed component, make sure it is the eyes: a frontal blob (blink) or "plus on the left, minus on the right" (looking sideways). If one looks like brain (a blob over C3 or C4 in the middle of the head), post it in the team chat.
4. **Report section:** in `slide_pdf/report/main.tex` read "Signal inspection and filtering" and "Artifact handling". Do not type numbers by hand.
5. **Review stage 1 (Bethel):**
   ```bash
   ROBT613_MODE=SMOKE .venv/bin/python scripts/01_audit.py
   ```
   Read the "Dataset facts" section of `NOTES.md` and add a dated line to `CONTRIBUTIONS.md`.

## 5. What you say at the defense (about 2.5 minutes)

Full text: `SCRIPT_defense_EN.md`, slides 4 and 5. Below is a cheat sheet.

**Slide "Stage 2 Signal inspection and filtering":**
- "The spectra show two problems: drift below 1 Hz and leftover 50 Hz power-line noise, up to +20 dB in places."
- "We use a 1–45 Hz filter. An important detail: with default settings it barely reduces 50 Hz, only −4.5 dB. We narrowed the transition band and got −51 dB. So no separate 50 Hz filter is needed."

**Slide "Stage 3 Artifact handling":**
- "We make three versions of the data: no cleaning, eye-channel regression, and ICA. Everything after is compared between them."
- "The key decision: the cleaning is trained on separate eye calibration recordings, not on the trials. So test data never enter training."
- "The first ICA setting from the plan failed: with 10 components the z ≥ 3 rule is mathematically impossible. We took 20 components and a correlation threshold of 0.5."
- "Result: the correlation of the frontal electrode with the eyes drops from 0.73 to 0.08 and 0.15."
- "We do not reject trials: the large amplitudes are exactly the eye movements we study." Hand over to Beibars.

## 6. Questions you may get

**"Why 1 Hz and not 0.5 Hz?"**
Below 1 Hz all recordings have strong drift, and ICA works poorly on drift. We still keep the delta band (1–4 Hz).

**"Doesn't the 1 Hz filter distort eye movements?"**
Partly. A long look to the side is a "step", and the filter turns it into a short spike. But the onset of the movement is kept: `fig_4c` shows a strong effect both at 1–45 Hz and in a wider band.

**"How is regression different from ICA?"**
Regression simply subtracts the eye channels from each EEG electrode with a fitted weight. It is simple, but it subtracts everything seen on the eye channels, including non-eye jumps. ICA splits the signal into independent sources and removes only those that look like the eyes. It is more flexible, but the component choice is semi-automatic.

**"Doesn't ICA remove brain signal?"**
There is always a risk. That is why we chose the threshold from the empty gap in the correlation distribution (0.25–0.55), not at random, and remove only 2 components out of 20. Also, accuracy on the motor features did not drop after ICA (0.77 and 0.77).

**"Why CAR and not the original reference?"**
The original reference is one electrode behind the left ear. Because of it, left-side electrodes, especially C5, look "quieter" (`fig_3d`). For a left-versus-right comparison that is a bias. CAR removes it.

**"Why didn't you reject bad trials?"**
The largest range is 142 µV, below the usual 150 µV threshold. And the large amplitudes are the eyes we are studying. We checked the expert-flagged trials separately, and the conclusions are the same.

**"Why didn't you remove muscle artifacts (EMG)?"**
Neither regression nor ICA with our rule targets muscles, and the course gives no method for it. This is a limitation, and it is in the report. It matters most for the tongue class.

## 7. Your files
- `src/preprocess.py` — filter, CAR, regression, ICA, epochs.
- `scripts/02_signal_inspection.py`, `scripts/03_artifacts.py`.
- `results/tables/signal_*.csv`, `results/tables/qc_*.csv`.
- `results/cache/ica/` — saved ICA for each recording.
- Figures `fig_2*` and `fig_3*`.
