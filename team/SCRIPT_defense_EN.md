# Defense script (English) — 12 minutes

Russian translation for understanding: `СЦЕНАРИЙ_защиты.md`. **We speak this English version at the defense.**

**How to use this file**
- Read your part out loud with a timer at least 3 times. Target speed is about 130 words per minute: calm, not rushed.
- You do not have to memorise it word for word. Keep the key numbers and the order of ideas exactly.
- Text in *[italic brackets]* is a stage direction (what to point at), not something to say.
- The script is about 1300 words: 10 minutes of speech plus pauses, so about 11 minutes, which leaves 1 minute of buffer. If we are running late, skip the lines marked **(cut if late)**.
- Slides: `slide_pdf/slides/slides.tex`. Slide numbers below follow the order in that file.

| # | Slide | Speaker | Time | Clock at end |
| --- | --- | --- | --- | --- |
| 1 | Title | Bethel | 0:20 | 0:20 |
| 2 | Question and pre-registered hypotheses | Bethel | 1:25 | 1:45 |
| 3 | Stage 1 Data audit | Bethel | 1:20 | 3:05 |
| 4 | Stage 2 Signal inspection and filtering | Sungkar | 0:50 | 3:55 |
| 5 | Stage 3 Artifact handling | Sungkar | 1:45 | 5:40 |
| 6 | Stage 4 The confound exists before any classifier | Beibars | 1:15 | 6:55 |
| 7 | Stage 5–6 Features, classifiers, honest evaluation | Beibars | 1:35 | 8:30 |
| 8 | Stage 7 H1 supported | Ramazan | 0:35 | 9:05 |
| 9 | Stage 7 H2 and H3 not supported | Ramazan | 0:45 | 9:50 |
| 10 | Exploratory: where the risk lies | Ramazan | 0:40 | 10:30 |
| 11 | Limits and take-home | Ramazan | 0:35 | 11:05 |

Per speaker: Bethel 3:05, Sungkar 2:35, Beibars 2:50, Ramazan 2:35. Times include short pauses for pointing at figures.
---

## Slide 1 — Title · Bethel · 0:20

Good morning. We are Bethel, Sungkar, Beibars and Ramazan. Our project asks a simple question about the most popular motor-imagery dataset. When a classifier decodes imagined movements, is it reading the brain, or is it reading the eyes?

## Slide 2 — Question and pre-registered hypotheses · Bethel · 1:25

In BCI Competition IV dataset 2a, nine people imagine moving the left hand, the right hand, the feet or the tongue. The task is shown by an arrow. The arrow points left for the left hand, right for the right hand, down for the feet and up for the tongue. And people's eyes reflexively look where an arrow points.

This matters because motor-imagery BCIs are built for people who cannot move. If a decoder works because of eye movements, it will fail for exactly those users, and published accuracies on this benchmark would be inflated.

*[point at the three hypotheses]*
We wrote down three hypotheses before running the full analysis.
H1. The three eye channels alone can tell left from right hand better than chance.
H2. Removing eye activity from the EEG lowers EEG decoding, and more so in people whose eyes carry more information.
H3. After cleaning, the classifier shifts its weight from frontal and artifact-prone features to central motor rhythms.

We also fixed the statistical decision rules in advance, so we could not tune the analysis to the result.

## Slide 3 — Stage 1 Data audit · Bethel · 1:20

The dataset has nine subjects with two sessions each, so eighteen recordings. Each recording has 288 trials, 22 EEG channels and 3 eye channels, at 250 hertz.

*[point at the raw signal; the red traces are the eye channels]*
We did not take any fact on trust. The most important check was timing. In the original files the marker sits at the start of the trial, and the arrow appears two seconds later. We confirmed that the library shifts the marker exactly onto the arrow, for every one of the 5184 trials.

The library also drops two things we needed, and we recovered both from the original files. The first is the expert's artifact flags, which mark about nine percent of trials. The second is short eye calibration recordings made at the start of each session. Those become important in stage 3.

**(cut if late)** The whole project runs from one notebook. We ran it twice and got identical numbers.

Sungkar will now explain how we cleaned the signal.

## Slide 4 — Stage 2 Signal inspection and filtering · Sungkar · 0:50

*[point at the solid lines on the left panel]*
These are the raw spectra. We see two problems. There is slow drift below one hertz in every recording, and there is leftover power-line noise at fifty hertz, up to twenty decibels above its neighbours.

*[point at the dashed lines]*
We use a band-pass filter from one to forty-five hertz. There is one detail worth mentioning. With the default settings, this filter reduces fifty hertz by only four and a half decibels, which is almost nothing. We narrowed the transition band and got minus fifty-one decibels at the same filter length. So we did not need a separate notch filter.

## Slide 5 — Stage 3 Artifact handling · Sungkar · 1:45

We create three versions of the data, and every later comparison is between them. "None" is the filter plus a common average reference. "EOG regression" subtracts a scaled copy of the eye channels from each EEG channel. "ICA" splits the signal into twenty independent components and removes the ones that look like the eyes.

The key decision is where we train the cleaning. If we trained it on the trials, the test trials would leak into it. So we train it on the separate eye calibration recordings and then apply it to the trials.

**(cut if late)** Our first ICA setting from the plan failed. With only ten components, the standard z-score rule cannot even reach its threshold. So we used twenty components and removed those correlating with the eyes at 0.5 or more.

*[point at the Fz trace]*
Here is a blink. In black, without cleaning, it leaks fifty microvolts into the frontal channel. In orange and green it is gone, and the rest of the signal is untouched. Overall, the correlation between the frontal channel and the eyes drops from 0.73 to 0.08 and 0.15. We do not reject any trials, because the large ones are exactly the eye movements we study.

Beibars will now show what the data look like before any classifier.

## Slide 6 — Stage 4 The confound exists before any classifier · Beibars · 1:15

Before training any classifier, we checked two things on plots. The expected brain signal had to be there, and so did the suspected eye signal.

The brain signal is there. When people imagine the left hand, the motor rhythm weakens over the right hemisphere, and the reverse for the right hand. After cleaning, this goes in the expected direction in all eighteen recordings.

*[point at the top-left panel: blue line and red line]*
This is the horizontal eye signal after the arrow. Blue is the left arrow, and the eyes move left. Red is the right arrow, and the eyes move right. The movement starts 0.19 seconds after the arrow, a normal eye reaction time, and nothing happens before it. The up and down arrows give the matching vertical movements.

*[point at the bottom row]*
We see this left-right difference in seventeen of eighteen recordings. So the suspected confound is real, before any classifier is involved.

*[next slide, same speaker]*
Now, how we decoded.

## Slide 7 — Stage 5–6 Features, classifiers, honest evaluation · Beibars · 1:35

From each trial we extract two kinds of features. The first is band power, the strength of each brain rhythm in the window where people imagine the movement. The second is amplitude, the average signal in short time bins right after the arrow. Amplitude is what captures a quick eye movement.

We compare four feature sets: the eye channels only, all EEG, the central motor channels and rhythms, and the frontal, artifact-prone channels and bands. The classifier is logistic regression, with shrinkage LDA as a check.

The evaluation is strict. We train on five runs and test on the sixth, six times over, so the test data are never seen in training. For every result we estimate the chance level by shuffling the labels. **(cut if late)** An automated test confirms that no test trial ever enters training.

*[point at the first panel, then at the second]*
Here is the main picture. The eye channels alone, on the left, decode most people as well as the whole EEG does, here in the second panel. And the three colours, meaning no cleaning, regression and ICA, are almost the same height.

Ramazan will now present the statistics.

## Slide 8 — Stage 7 H1 supported · Ramazan · 0:35

*[point at the bars and the red markers]*
H1 is supported. The three eye channels alone separate left from right hand with a median AUC of 0.86, where 0.5 is guessing. Eight of nine people are above their own chance level. The p-value is 0.002, the smallest possible with nine people. H1 also holds in all five robustness checks, including the other session and with the expert-flagged trials removed.

## Slide 9 — Stage 7 H2 and H3 not supported · Ramazan · 0:45

But H2 is not supported. Removing the eyes from the EEG does not lower the standard classifier. The median change is minus 0.002, with p equal to 0.63.

*[point at the left scatter plot]*
And the drop is not larger in people whose eyes carry more information.

*[point at the right plot]*
H3 is not supported either. The artifact-prone features do not lose more than the motor features, and the classifier weights do not shift in a measurable way.

These are honest negative results. The rules were fixed in advance.

## Slide 10 — Exploratory: where the risk lies · Ramazan · 0:40

So why do the eyes not matter here? The eye movement is a short event in time, right after the arrow, and our EEG features measure rhythm power. This next analysis was not planned in advance. We tested time-domain features on the frontal channels instead.

*[point at the lines going down]*
Their median did not drop significantly. But the stronger a person's eye signal, the more they lost after cleaning, with a correlation of 0.98. Time-domain features do pick up the eyes.

## Slide 11 — Limits and take-home · Ramazan · 0:35

Our study has limits. It has only nine people and is offline only. We did not remove muscle artifacts, and frontal cleaning is incomplete. And the eye movement is a reflex to the arrow, not subjects cheating.

Our take-home message is simple. Anyone reporting results on this dataset should show what the eyes alone achieve, next to their score.

Thank you. We are happy to take questions.

---

## Rehearsal checklist

- [ ] Each speaker reads their part with a timer three times.
- [ ] One full run-through together, with the slides, and a timer for the whole talk.
- [ ] If the full run is over 11:30, drop the **(cut if late)** lines.
- [ ] Each speaker knows the 3–4 numbers in their part by heart. The numbers are listed below.
- [ ] Each speaker reads the "Questions you may get" section in their own file, because anyone can be asked about any stage.

## Key numbers per speaker

- **Bethel:** 18 recordings, 288 trials each, 22 EEG + 3 eye channels, marker checked on 5184 trials, about 9% expert-flagged.
- **Sungkar:** filter 1–45 Hz, 50 Hz noise up to +20 dB, default −4.5 dB versus ours −51 dB, 20 ICA components, threshold 0.5, eye correlation 0.73 → 0.08 / 0.15.
- **Ramazan:** H1 AUC 0.86, 8 of 9, p = 0.002, H2 change −0.002 with p = 0.63, exploratory correlation 0.98.
- **Beibars:** eyes move 0.19 s after the arrow, 17 of 18 recordings, 4 feature sets, train on 5 runs and test on the 6th, chance by label shuffling, eyes-only AUC 0.86 versus all EEG 0.85.
