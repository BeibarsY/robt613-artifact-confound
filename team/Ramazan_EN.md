# Ramazan — Evaluation and reporting

**Role:** Evaluation and reporting lead
**Stages:** 4 (neurophysiological check), 7 (hypothesis tests), report and slides
**You review:** stages 5–6 (Beibars)
**At the defense:** you speak last — hypothesis results, exploratory, limits and conclusion, about 2 minutes 35 seconds

Read `00_OVERVIEW_EN.md` first. It covers the whole project. Russian version: `Ramazan.md`.

---

## 1. Your part in plain words

You own three things:
1. **Stage 4:** **before** any classifier, show on plots whether the data contain (a) the normal brain signal and (b) the suspicious eye signal. If neither existed, no classifier would prove anything.
2. **Stage 7:** answer H1, H2 and H3 statistically, using the rules written down in advance.
3. **Report and slides:** turn everything into a PDF and make sure every number is honest.

## 2. Stage 4: what we see without any classifier

### (a) The brain signal is there
When someone imagines moving the **left** hand, mu and beta rhythms **weaken** over the **right** hemisphere (electrode C4), and vice versa. This is called ERD.
- After ICA the left/right beta difference goes in the expected direction in **all 18 recordings** (`fig_4a`, `fig_4b`).
- In some it is weak: S2 (0train), S5 (both sessions), S7 (0train). We **keep** them and flag them as "BCI may work poorly for them". Such people exist in every study.

### (b) The eye signal is there too, and it is strong (key figure `fig_4c`)
- **Left** arrow → eyes look left. **Right** arrow → right. On the horizontal EOG (HEOG) these are two bumps in opposite directions.
- **Down** (feet) and **up** (tongue) arrows give matching bumps on the vertical EOG (VEOG). So the eyes "give hints" in the 4-class task too.
- The left-minus-right difference goes the expected way in **17 of 18** recordings, highly significant (p < 0.001) in 15. Effect size (Cohen's d) up to 3.2, which is huge.
- The movement starts **0.19 s** after the arrow (normal eye reaction time) and peaks at **0.37 s**. Nothing happens before the arrow, which also proves the time marker is correct.
- When the arrow disappears (at 1.25 s) the eyes return to the centre. This falls inside the window where we compute features.
- This signal leaks into the frontal EEG electrodes FC3 and FC4 (`fig_4d`). Cleaning removes it **only partly**.

## 3. Stage 7: hypothesis results

| Hypothesis | Rule (written in advance) | Result | Verdict |
| --- | --- | --- | --- |
| **H1** the eyes predict left vs right | Wilcoxon p < 0.05 **and** at least 5 of 9 above their own chance level | AUC 0.86, p = 0.002, **8 of 9** | **Supported** |
| **H2** cleaning lowers EEG accuracy | Wilcoxon p < 0.05 **and** the drop tracks eye strength (ρ > 0) | change −0.002, p = 0.63 | **Not supported** |
| **H3** after cleaning, motor features become important | the "suspicious" set loses more than the "motor" set (p < 0.05) | p = 0.75, weights shift slightly (24.8% → 27.6%, p = 0.13) | **Not supported** |

**Why p = 0.002 is the limit:** with 9 people the Wilcoxon test cannot give p below 1/512 ≈ 0.002. It is the strongest result possible with 9 people.

**Robustness checks:** H1 holds in **all 5** variants (other classifier, other session, expert-flagged trials removed, train on one session and test on the other). H2 holds in **0 of 10**.

**Exploratory (NOT planned in advance, say so every time):**
- We added time-domain amplitudes at frontal electrodes (Fz, FC1–FC4). Without cleaning AUC is 0.70.
- After cleaning the median did not drop significantly.
- **But** the stronger a person's eye signal, the more they lose after cleaning (correlation 0.98 and 0.85).
- Conclusion: **time-domain** frontal features use the eyes, and rhythm power (our main features) does not.

**Main message for the conclusions:** the eye problem is real and large, but it is an event **in time** (a short jump after the arrow). Standard rhythm-power features do not notice it. The risky methods are those that look at the signal shape at the forehead or use the EOG.

## 4. Your tasks (step by step)

### Task 1. Understand
Read "Phase 4 findings" and "Phase 7 results" in `NOTES.md`. Look at `fig_4a`–`fig_4e` and `fig_7a`–`fig_7e` in `results/figures/`.

### Task 2. Check the references
`slide_pdf/report/refs.bib` has 6 references. Brunner 2008 has been opened and read. The other 5 were only checked by web search, and the plan requires someone to **open each one**. For each:
1. Find the paper (Google Scholar, journal site, PubMed).
2. Check authors, title, journal, year, volume and pages.
3. Where there is a `% TODO-verify DOI` line, find the DOI, add `doi = {...}` and delete that line.

### Task 3. Compile the report in Overleaf
1. Overleaf → New Project → Upload Project. Upload the **whole** `slide_pdf/report/` folder: `main.tex`, `refs.bib`, `numbers.tex` and the `figures/` folder.
2. Press Recompile. If there is an error, post it in the chat.
3. Target **6–8 pages**. It is about 5–6 now. You can add figures (for example `fig_6a`, `fig_4a`) or expand the discussion.

### Task 4. Three report rules
1. **No number typed by hand.** Every number in the text is a command like `\HoneP` or `\AucEogNoneT`. They come from `numbers.tex`, which is generated from the results tables. If you need a new number, ask for a command to be added to `slide_pdf/report/make_numbers.py`.
2. **No colons or semicolons** in the body text (assignment rule).
3. The exploratory analysis is labelled **exploratory, not pre-registered** everywhere.

### Task 5. Slides
`slide_pdf/slides/slides.tex` compiles in Overleaf the same way. It needs `slide_pdf/report/figures/` and `slide_pdf/report/numbers.tex`, because the slides load them from `../report/`. The easiest way is to upload `slide_pdf/report/` and `slide_pdf/slides/` into one project.

### Task 6. End of the report (with everyone)
Replace the `TODO`s at the end of `slide_pdf/report/main.tex`: names, affiliation, AI-use statement (Claude Code).

### Task 7. Review stages 5–6 (Beibars)
```bash
ROBT613_MODE=SMOKE .venv/bin/python scripts/05_decode_all_conditions.py
```
Read "Phase 5-6 design" in `NOTES.md` and add a dated line to `CONTRIBUTIONS.md`.

## 5. What you say at the defense

Full text: `SCRIPT_defense_EN.md`, slides 8–11. Below is a cheat sheet.

> Beibars presents the stage 4 slide. But **you lead stage 4**, so you may get questions about it. Keep section 2 of this file in mind.

### Slides "Stage 7 H1", "H2 and H3", "Exploratory", "Limits" (about 2 minutes 35 seconds)
- **H1:** "The three eye channels alone separate left from right hand with AUC 0.86. 8 of 9 people are above their own chance level, p = 0.002, the smallest possible with 9 people. H1 is supported and holds in all five robustness checks."
- **H2 and H3:** "But removing the eyes from the EEG does not lower the standard classifier. The change is −0.002, p = 0.63. The classifier weights do not shift in a measurable way. H2 and H3 are not supported."
- **Exploratory:** "Why? The eye movement is a short event in time, and our features are rhythm power. As an extra analysis, not planned in advance, we tested time-domain frontal features. There, the stronger a person's eye signal, the more they lose after cleaning."
- **Limits:** "Only 9 people, offline only, we did not remove muscle artifacts, and frontal cleaning is incomplete. And it is a reflex to the arrow, not subjects cheating."
- **Conclusion:** "Our recommendation for anyone using this dataset: report what the eyes alone achieve next to your score."

## 6. Questions you may get

**"Why is a negative result for H2 acceptable?"**
We wrote the rules before the analysis. H2 failed honestly. That is useful knowledge: standard band-power classifiers on 2a do not rely on the eyes. Tuning the analysis to "confirm" it would be dishonest.

**"Why Wilcoxon and not a t-test?"**
9 people are too few to check normality. Wilcoxon does not need it.

**"Did you correct for multiple comparisons?"**
Yes. H2 has two comparisons (regression and ICA), Holm correction. Corrected p = 1.0, so the same conclusion.

**"How did you get the H3 weights, if training on all data is leakage?"**
The weights in `fig_7c` come from a fit on the **whole** session **for the picture only**, and the figure says so. No accuracy is computed from them. All accuracies come from cross-validation.

**"Isn't the exploratory analysis cherry-picking?"**
We label it honestly as unplanned and chosen after seeing H2. So it is not proof, it is a hypothesis for the next study.

**"Why does S9 have a peak at 31 Hz?"**
It sits over the motor area (C4) and depends strongly on the class, so it is probably a real brain rhythm (high beta), not an artifact. So the 30–45 Hz band is not pure noise. We note this as a limit for H3.

**"Why keep people with weak ERD?"**
Removing people with poor results would be cherry-picking. This happens in every BCI study ("BCI inefficiency"). We flagged them and kept them.

**"How do you know the numbers in the report are right?"**
No number is typed by hand. `slide_pdf/report/make_numbers.py` reads the results tables and writes LaTeX commands. If the data are recomputed, the report updates itself.

## 7. Your files
- `src/neurophys.py`, `scripts/04_neurophys_check.py` — stage 4.
- `src/stats.py`, `scripts/06_stats.py` — stage 7.
- `results/tables/neuro_*.csv`, `results/tables/stats_*.csv`.
- Figures `fig_4*` and `fig_7*`.
- `slide_pdf/report/main.tex`, `slide_pdf/report/refs.bib`, `slide_pdf/report/make_numbers.py`, `slide_pdf/report/numbers.tex`.
- `slide_pdf/slides/slides.tex`.
