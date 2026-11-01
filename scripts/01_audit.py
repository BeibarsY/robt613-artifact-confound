"""Stage 1 - data audit. Writes results/tables/audit_*.csv and fig_1a/fig_1b."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

import config
from src import data, plots
from src.timing import timed

mne.set_log_level("ERROR")
print(f"MODE={config.MODE} subjects={config.CFG['subjects']} seed={config.SEED}")

summaries, trials_all, calib_all = [], [], []
with timed("01_audit"):
    for s in config.CFG["subjects"]:
        sessions = data.load_subject(s)
        print(f"S{s}: sessions={list(sessions)}")
        for sess in config.CFG["sessions"]:
            runs = sessions[sess]
            tt = data.trial_table(s, sess, runs)
            trials_all.append(tt)
            calib_all += data.calibration_summary(s, sess)
            summaries.append(data.recording_summary(s, sess, runs, tt))

    summary = pd.DataFrame(summaries)
    trials = pd.concat(trials_all, ignore_index=True)
    counts = (trials.groupby(["subject", "session", "label"]).size().unstack("label")
              .reindex(columns=config.CLASSES_4).reset_index())
    counts["total"] = counts[config.CLASSES_4].sum(axis=1)
    art = (trials.groupby(["subject", "session", "label"])["expert_artifact"].sum()
           .unstack("label").reindex(columns=config.CLASSES_4).add_prefix("artifact_").reset_index())
    counts = counts.merge(art, on=["subject", "session"])

    summary.to_csv(config.TABLES / "audit_summary.csv", index=False)
    counts.to_csv(config.TABLES / "audit_trial_counts.csv", index=False)
    trials.to_csv(config.TABLES / "audit_trials.csv", index=False)
    pd.DataFrame(calib_all).to_csv(config.TABLES / "audit_calibration_runs.csv", index=False)
    print(summary.T.to_string()); print(counts.to_string(index=False))

    # fig_1a: 10 s of raw EEG + EOG around the first cue of subject 1, run 0
    s0 = config.CFG["subjects"][0]
    raw = data.load_subject(s0)[config.CFG["sessions"][0]]["0"]
    first_cue = trials.query("subject == @s0")["cue_time_s"].iloc[0]
    t0 = max(first_cue - 3.0, 0.0)
    seg = raw.copy().crop(t0, t0 + 10.0).pick(["eeg", "eog"])
    x = seg.get_data() * 1e6
    fig, ax = plt.subplots(figsize=(7.5, 8))
    step = 60.0
    for i, (name, tr) in enumerate(zip(seg.ch_names, x)):
        ax.plot(seg.times + t0, tr - tr.mean() - i * step, lw=0.5,
                color="C3" if name.startswith("EOG") else "k")
    ax.set_yticks(-np.arange(len(seg.ch_names)) * step, seg.ch_names)
    for c in trials.query("subject == @s0 and run == 0 and cue_time_s >= @t0 and cue_time_s <= @t0 + 10")["cue_time_s"]:
        ax.axvline(c, color="C0", ls="--", lw=0.8)
    ax.set_xlabel("time in run (s); dashed = cue onset (annotation)")
    ax.set_title(f"S{s0} {config.CFG['sessions'][0]} run 0, unfiltered (offset {step:.0f} µV per channel)")
    plots.save(fig, "fig_1a_raw_segment")

    fig = raw.plot_sensors(show_names=True, show=False)
    plots.save(fig, "fig_1b_sensor_layout")
