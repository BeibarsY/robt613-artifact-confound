"""Stage 2 - signal inspection and filter choice. Writes fig_2a/fig_2b and signal_*.csv."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

import config
from src import data, plots, preprocess
from src.timing import timed

mne.set_log_level("ERROR")
print(f"MODE={config.MODE} subjects={config.CFG['subjects']}")

with timed("02_signal_inspection"):
    # spectral QC on every recording (evidence for the notch / high-pass decision)
    rows = []
    for s in config.CFG["subjects"]:
        sessions = data.load_subject(s)
        for sess in config.CFG["sessions"]:
            raw = preprocess.concat_session(sessions[sess])
            rows.append({"subject": s, "session": sess, **preprocess.spectral_qc(raw)})
    qc = pd.DataFrame(rows)
    qc.to_csv(config.TABLES / "signal_psd_qc.csv", index=False)
    print(qc.round(2).to_string(index=False))

    fdesc = preprocess.filter_description()
    pd.DataFrame([fdesc]).to_csv(config.TABLES / "signal_filter_params.csv", index=False)
    print(fdesc)

    # fig_2a: PSD before / after filtering, one subject, first configured session
    s0, sess0 = config.CFG["subjects"][0], config.CFG["sessions"][0]
    raw = preprocess.concat_session(data.load_subject(s0)[sess0])
    filt = preprocess.bandpass(raw)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    panels = [("EEG mean (22 ch)", "eeg"), ("C3 / C4", ["C3", "C4"]), ("EOG1-3", "eog")]
    for ax, (title, picks) in zip(axes, panels):
        for r, lab, ls in ((raw, "unfiltered", "-"), (filt, "1-45 Hz FIR", "--")):
            p, f = preprocess.psd(r, picks)
            p = 10 * np.log10(p * 1e12)  # µV^2/Hz in dB
            if picks == "eeg":
                ax.plot(f, p.mean(0), ls=ls, color="k", label=lab, lw=0.9)
            else:
                names = picks if isinstance(picks, list) else ["EOG1", "EOG2", "EOG3"]
                for i, n in enumerate(names):
                    ax.plot(f, p[i], ls=ls, color=f"C{i}", lw=0.9, label=f"{n} {lab}")
        for x in (1.0, 45.0, 50.0):
            ax.axvline(x, color="grey", lw=0.5, ls=":")
        ax.set_title(title); ax.set_xlabel("frequency (Hz)"); ax.set_xlim(0, 125)
        ax.legend(fontsize=6, frameon=False)
    axes[0].set_ylabel("PSD (dB re 1 µV²/Hz)")
    fig.suptitle(f"S{s0} {sess0}, Welch 2 s windows; dotted lines at 1, 45, 50 Hz", fontsize=9)
    plots.save(fig, "fig_2a_psd_before_after")

    # fig_2b: analysis filter response
    kw = {k: v for k, v in preprocess.FILTER_KW.items() if k not in ("picks", "method")}
    h = mne.filter.create_filter(None, config.SFREQ, verbose=False, **kw)
    fig = mne.viz.plot_filter(h, config.SFREQ, freq=None, gain=None, fscale="linear",
                              flim=(0, 70), show=False)
    plots.save(fig, "fig_2b_filter_response")

    # fig_2c: per-recording evidence for the notch decision and the high-band peak check
    lab = [f"S{r.subject}{'T' if r.session == '0train' else 'E'}" for r in qc.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    x = np.arange(len(qc))
    axes[0].bar(x - 0.2, qc["eeg_line_50Hz_dB_mean"], 0.4, label="EEG mean", color="k")
    axes[0].bar(x + 0.2, qc["eog_line_50Hz_dB_mean"], 0.4, label="EOG mean", color="C3")
    axes[0].axhline(fdesc["gain_dB_50Hz"], color="C0", ls="--", lw=0.8,
                    label=f"analysis filter gain at 50 Hz ({fdesc['gain_dB_50Hz']:.0f} dB)")
    axes[0].set_ylabel("50 Hz bin above 40-60 Hz flanks (dB)")
    axes[0].set_title("Residual line noise, unfiltered data")
    axes[1].bar(x, qc["eeg_high_peak_dB"], color="C2")
    for i, r in enumerate(qc.itertuples()):
        axes[1].text(i, r.eeg_high_peak_dB + 0.1, f"{r.eeg_high_peak_ch}\n{r.eeg_high_peak_Hz:g}",
                     ha="center", fontsize=5)
    axes[1].set_ylabel("max excess over 1/f fit, 30-45 Hz (dB)")
    axes[1].set_title("Narrowband peaks in the 'high' feature band")
    for ax in axes:
        ax.set_xticks(x, lab, rotation=90, fontsize=7)
    axes[0].legend(fontsize=6, frameon=False)
    plots.save(fig, "fig_2c_line_noise_and_high_band_peaks")
