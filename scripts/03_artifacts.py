"""Stage 3 - artifact handling and QC. Three cleaning conditions, one shared epoch set.

Writes qc_ptp_per_epoch.csv, qc_rejection_counts.csv, qc_cleaning_fit.csv,
qc_eog_correlation.csv and fig_3a-3e.
"""
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
print(f"MODE={config.MODE} subjects={config.CFG['subjects']} fit_on={config.CLEAN_FIT_ON} "
      f"ptp_reject_uV={config.PTP_REJECT_UV}")

s0, sess0 = config.CFG["subjects"][0], config.CFG["sessions"][0]
ptp_rows, fit_rows, corr_rows, rej_rows = [], [], [], []
with timed("03_artifacts"):
    for s in config.CFG["subjects"]:
        sessions = data.load_subject(s)
        for sess in config.CFG["sessions"]:
            runs = sessions[sess]
            tt = data.trial_table(s, sess, runs)
            conds, info = preprocess.build_conditions(s, sess, runs)
            fit_rows.append({"subject": s, "session": sess, **{k: v for k, v in info.items()
                             if k not in ("ica", "fit_raw", "ica_eog_scores", "ica_excluded")},
                             "ica_excluded": ";".join(map(str, info["ica_excluded"])),
                             "n_ica_excluded": len(info["ica_excluded"]),
                             "ica_max_eog_score_excluded": float(info["ica_eog_scores"].max(0)[info["ica_excluded"]].min())
                             if info["ica_excluded"] else np.nan})
            epochs = {c: preprocess.make_epochs(r, tt) for c, r in conds.items()}
            ptp = pd.DataFrame({f"ptp_{c}_uV": preprocess.ptp_uv(e) for c, e in epochs.items()})
            ptp_rows.append(pd.concat([tt.reset_index(drop=True), ptp], axis=1))
            for c, r in conds.items():
                corr_rows.append({"subject": s, "session": sess, "condition": c, **preprocess.eog_corr(r)})
            if s == s0 and sess == sess0:
                keep_s0 = (conds, info, epochs)
            print(f"S{s} {sess}: ICA {info['n_ica_components']} comps, excluded {info['ica_excluded']}")

    ptp_all = pd.concat(ptp_rows, ignore_index=True)
    thr = config.PTP_REJECT_UV
    ptp_all["rejected"] = (ptp_all["ptp_none_uV"] > thr) if thr is not None else False
    ptp_all.to_csv(config.TABLES / "qc_ptp_per_epoch.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(config.TABLES / "qc_cleaning_fit.csv", index=False)
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(config.TABLES / "qc_eog_correlation.csv", index=False)

    # rejection counts per subject x session x condition (identical across conditions by design)
    fit = pd.DataFrame(fit_rows).set_index(["subject", "session"])
    for (s, sess), g in ptp_all.groupby(["subject", "session"]):
        for c in config.CONDITIONS:
            row = {"subject": s, "session": sess, "condition": c,
                   "n_ica_components": fit.loc[(s, sess), "n_ica_components"] if c == "ica" else np.nan,
                   "ica_excluded": fit.loc[(s, sess), "ica_excluded"] if c == "ica" else "",
                   "n_epochs_rejected": int(g["rejected"].sum()),
                   "n_rejected_also_expert": int((g["rejected"] & (g["expert_artifact"] == 1)).sum())}
            for lab in config.CLASSES_4:
                row[f"kept_{lab}"] = int(((g["label"] == lab) & ~g["rejected"]).sum())
            rej_rows.append(row)
    rej = pd.DataFrame(rej_rows)
    rej.to_csv(config.TABLES / "qc_rejection_counts.csv", index=False)
    print(pd.DataFrame(fit_rows)[["subject", "session", "fit_duration_s", "fit_samples_per_comp2",
                                  "n_ica_components", "ica_excluded", "ica_max_eog_score_excluded"]].to_string(index=False))
    print(corr.groupby("condition")[[c for c in corr if c.startswith("absr")]].median().round(3))
    for c in config.CONDITIONS:
        pc = np.percentile(ptp_all[f"ptp_{c}_uV"], [50, 90, 99, 99.5, 100])
        print(f"PTP {c} (µV) pct 50/90/99/99.5/max:", np.round(pc, 1))
    worse = ptp_all["ptp_eog_reg_uV"] > ptp_all["ptp_none_uV"].max()
    print("epochs where eog_reg PTP exceeds the largest `none` PTP:", int(worse.sum()),
          ptp_all.loc[worse, ["subject", "session", "run", "trial_in_run", "label", "expert_artifact",
                              "ptp_none_uV", "ptp_eog_reg_uV"]].round(0).to_string(index=False))

    conds, info, epochs = keep_s0
    # fig_3a: ICA component maps, excluded ones titled in red
    ica = info["ica"]
    figs = ica.plot_components(inst=None, show=False, title=f"S{s0} {sess0} ICA (fit on {config.CLEAN_FIT_ON})")
    fig = figs[0] if isinstance(figs, list) else figs
    sc = info["ica_eog_scores"].max(0)
    for k, ax in enumerate([a for a in fig.axes if a.get_title()][: ica.n_components_]):
        ax.set_title(f"ICA{k:03d}\n|r|={sc[k]:.2f}", fontsize=7,
                     color="C3" if k in ica.exclude else "k")
    plots.save(fig, "fig_3a_ica_components_topo")

    # fig_3b / fig_3b2: Fz and FC2 in all three conditions around two events in the MI runs.
    #   3b  = typical blink: largest |Fz| in `none` among samples where |EOG| < 300 µV
    #   3b2 = largest |EOG| deflection overall (inspect: may be non-ocular, e.g. an electrode step)
    none = conds["none"]
    eog = none.get_data(picks=config.EOG_CHANNELS, reject_by_annotation="NaN")
    eog = np.nanmax(np.abs(eog - np.nanmedian(eog, axis=1, keepdims=True)), axis=0)
    fz = np.abs(none.get_data(picks="Fz", reject_by_annotation="NaN")[0])
    events_3b = {"fig_3b_eog_regression_before_after": int(np.nanargmax(np.where(eog < 300e-6, fz, np.nan))),
                 "fig_3b2_largest_eog_event": int(np.nanargmax(eog))}
    for fname, i_peak in events_3b.items():
        t_peak = none.times[i_peak]
        tmin, tmax = max(t_peak - 2.0, 0), min(t_peak + 2.0, none.times[-1])
        fig, axes = plt.subplots(3, 1, figsize=(7, 5.5), sharex=True)
        seg = none.copy().crop(tmin, tmax)
        for e in config.EOG_CHANNELS:
            axes[0].plot(seg.times + tmin, seg.get_data(picks=e)[0] * 1e6, lw=0.8, label=e)
        axes[0].set_ylabel("EOG (µV)"); axes[0].legend(fontsize=6, frameon=False, ncol=3)
        for ax, ch in zip(axes[1:], ["Fz", "FC2"]):
            for c, col in zip(config.CONDITIONS, ["k", "C1", "C2"]):
                x = conds[c].copy().crop(tmin, tmax).get_data(picks=ch)[0] * 1e6
                ax.plot(seg.times + tmin, x, lw=0.8, color=col, label=c)
            ax.set_ylabel(f"{ch} (µV)"); ax.legend(fontsize=6, frameon=False, ncol=3)
        axes[-1].set_xlabel("time in session (s)")
        what = "typical blink" if "3b_" in fname else "largest EOG deflection"
        fig.suptitle(f"S{s0} {sess0}: {what} in MI runs, 1-45 Hz + CAR", fontsize=9)
        plots.save(fig, fname)

    # fig_3c: per-epoch max EEG peak-to-peak (all configured recordings), none condition
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    bins = np.logspace(np.log10(ptp_all[[f"ptp_{c}_uV" for c in config.CONDITIONS]].min().min()),
                       np.log10(ptp_all[[f"ptp_{c}_uV" for c in config.CONDITIONS]].max().max()), 60)
    ok = ptp_all["expert_artifact"] == 0
    axes[0].hist([ptp_all.loc[ok, "ptp_none_uV"], ptp_all.loc[~ok, "ptp_none_uV"]], bins=bins,
                 stacked=True, color=["0.6", "C3"], label=["expert: clean", "expert: artifact"])
    for q, ls in ((99, ":"), (99.5, "--")):
        axes[0].axvline(np.percentile(ptp_all["ptp_none_uV"], q), color="k", ls=ls, lw=0.8, label=f"{q}th pct")
    if thr is not None:
        axes[0].axvline(thr, color="C0", lw=1.2, label=f"threshold {thr:g} µV")
    axes[0].set_xscale("log"); axes[0].set_xlabel("max EEG peak-to-peak per epoch, `none` (µV)")
    axes[0].set_ylabel("epochs"); axes[0].legend(fontsize=6, frameon=False)
    for c, col in zip(config.CONDITIONS, ["k", "C1", "C2"]):
        axes[1].hist(ptp_all[f"ptp_{c}_uV"], bins=bins, histtype="step", color=col, label=c)
    axes[1].set_xscale("log"); axes[1].set_xlabel("max EEG peak-to-peak per epoch (µV)")
    axes[1].legend(fontsize=6, frameon=False)
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    for ax in axes:
        ax.xaxis.set_major_locator(FixedLocator([20, 30, 50, 100, 200, 400]))
        ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    fig.suptitle(f"{len(ptp_all)} epochs, {ptp_all.groupby(['subject','session']).ngroups} recordings, "
                 "window -1 to +4 s around cue", fontsize=9)
    plots.save(fig, "fig_3c_ptp_histogram")

    # fig_3d: reference choice. Mastoid (as recorded) vs CAR, mean log MI-window 8-30 Hz power
    from src import data as _d  # noqa: F401
    mast = preprocess.bandpass(preprocess.concat_session(data.load_subject(s0)[sess0]))
    tt0 = data.trial_table(s0, sess0, data.load_subject(s0)[sess0])
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))
    for ax, (name, r) in zip(axes, (("left mastoid (recorded)", mast), ("CAR", conds["none"]))):
        ep = preprocess.make_epochs(r, tt0).pick("eeg").crop(*config.MI_WINDOW)
        sp = ep.compute_psd(method="welch", fmin=8, fmax=30, n_fft=250, verbose=False)
        lp = np.log10(sp.get_data().mean(-1).mean(0) * 1e12)
        im, _ = mne.viz.plot_topomap(lp, ep.info, axes=ax, show=False, cmap=plots.CMAP_SEQ, contours=0)
        ax.set_title(name, fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.7, label="log10 µV²/Hz")
    fig.suptitle(f"S{s0} {sess0}: mean 8-30 Hz power, MI window {config.MI_WINDOW} s after cue", fontsize=8)
    plots.save(fig, "fig_3d_reference_mastoid_vs_car")

    # fig_3e: residual |r| between Fz and each EOG channel, per recording and condition
    fig, ax = plt.subplots(figsize=(8, 3))
    corr["rec"] = corr["subject"].map("S{}".format) + corr["session"].map({"0train": "T", "1test": "E"})
    corr["absr_max"] = corr[[f"absr_Fz_{e}" for e in config.EOG_CHANNELS]].max(axis=1)
    piv = corr.pivot(index="rec", columns="condition", values="absr_max")[config.CONDITIONS]
    piv = piv.loc[sorted(piv.index, key=lambda r: (int(r[1:-1]), r[-1] != "T"))]
    piv.plot.bar(ax=ax, color=["k", "C1", "C2"], width=0.8)
    ax.set_ylabel("max |r| (Fz, EOG1-3)"); ax.set_xlabel("")
    ax.legend(fontsize=7, frameon=False)
    plots.save(fig, "fig_3e_residual_eog_correlation")

    # fig_3f: evidence for the ICA exclusion threshold. Max |r| (any EOG channel) of every
    # component on the fitting data, all configured recordings.
    sc_all = []
    for s_, sess_ in [(r["subject"], r["session"]) for r in fit_rows]:
        sp = config.CACHE / "ica" / f"S{s_:02d}_{sess_}_{config.CLEAN_FIT_ON}-ica.scores.npy"
        sc_all.append(np.load(sp).max(0))
    sc_all = np.concatenate(sc_all)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(sc_all, bins=np.linspace(0, 1, 41), color="0.5")
    ax.axvline(config.ICA_EOG_THRESHOLD, color="C3", lw=1, label=f"threshold |r| = {config.ICA_EOG_THRESHOLD}")
    ax.set_xlabel("max |r| with EOG1-3, ICA component time course (fitting data)")
    ax.set_ylabel("components"); ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"{len(sc_all)} components, {len(fit_rows)} recordings", fontsize=8)
    plots.save(fig, "fig_3f_ica_eog_score_distribution")
