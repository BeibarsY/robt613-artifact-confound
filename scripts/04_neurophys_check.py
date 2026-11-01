"""Stage 4 - neurophysiological checks before any decoding.

(1) Mu/beta ERD over C3/C4 (does the expected signal exist?)
(2) Cue-locked EOG and frontal EEG by class (does the suspected confound exist?)
Also writes the per-recording feature cache used by Phase 5.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats

import config
from src import data, features, neurophys, plots, preprocess
from src.timing import timed

mne.set_log_level("ERROR")
print(f"MODE={config.MODE} subjects={config.CFG['subjects']}")
BANDS = list(config.BANDS)
FRONTAL = ["Fz", "FC1", "FC2", "FC3", "FC4"]
erd_rows, eog_rows, hb_rows = [], [], []
evk = {}          # (s, sess, cond, kind) -> {class: mean time course}
psd_c34 = {}      # (s, sess, cond) -> {class: (2, n_freq)}
with timed("04_neurophys_check"):
    qc2 = pd.read_csv(config.TABLES / "signal_psd_qc.csv").set_index(["subject", "session"]) \
        if (config.TABLES / "signal_psd_qc.csv").exists() else None
    for s in config.CFG["subjects"]:
        sessions = data.load_subject(s)
        for sess in config.CFG["sessions"]:
            runs = sessions[sess]
            tt = data.trial_table(s, sess, runs)
            conds, _ = preprocess.build_conditions(s, sess, runs)
            eps = {c: preprocess.make_epochs(r, tt) for c, r in conds.items()}
            features.save_recording(s, sess, eps)
            labels = tt["label"].to_numpy()
            times = eps["none"].times

            # wide-band copy for the EOG evoked check: no extra high-pass (acquisition 0.5 Hz)
            wide = preprocess.concat_session(runs).filter(None, 45.0, h_trans_bandwidth=5.0,
                                                         picks=["eog"], verbose=False)
            ep_wide = preprocess.make_epochs(wide, tt)

            for c in ("none", "eog_reg", "ica"):
                ep = eps[c]
                ch = ep.copy().pick(["eeg", "eog"]).ch_names
                x = ep.get_data(picks=["eeg", "eog"])
                f = features.epoch_features(ep)
                erd = neurophys.erd_db(f["bp"], f["bp_base"])
                erd_mean = {k: erd[labels == k].mean(0) for k in config.CLASSES_4}
                for k in config.CLASSES_4:
                    for ci, chn in enumerate(ch):
                        if chn.startswith("EOG"):
                            continue
                        for bi, b in enumerate(BANDS):
                            erd_rows.append({"subject": s, "session": sess, "condition": c, "label": k,
                                             "channel": chn, "band": b, "erd_dB": erd_mean[k][ci, bi]})
                # class-wise PSD at C3/C4, MI window
                mi = x[:, [ch.index("C3"), ch.index("C4")]][..., (times >= config.MI_WINDOW[0]) & (times < config.MI_WINDOW[1])]
                fr, pw = __import__("scipy.signal", fromlist=["welch"]).welch(mi * 1e6, fs=config.SFREQ,
                                                                              nperseg=250, noverlap=125, axis=-1)
                psd_c34[(s, sess, c)] = {k: 10 * np.log10(pw[labels == k].mean(0)) for k in config.CLASSES_4}
                # frontal EEG evoked by class (baseline -1..0)
                base = (times >= config.BASELINE[0]) & (times < config.BASELINE[1])
                xb = (x - x[..., base].mean(-1, keepdims=True)) * 1e6
                for chn in FRONTAL:
                    evk[(s, sess, c, chn)] = {k: xb[labels == k, ch.index(chn)].mean(0) for k in config.CLASSES_4}
                if c == "none":
                    for band, xb_src in (("1-45", xb), ("wide", None)):
                        if band == "wide":
                            xw = ep_wide.get_data(picks="eog")
                            xb_src = (xw - xw[..., base].mean(-1, keepdims=True)) * 1e6
                            d = neurophys.eog_derivations(xb_src, config.EOG_CHANNELS)
                        else:
                            d = neurophys.eog_derivations(xb_src, ch)
                        for kind, sig in d.items():
                            evk[(s, sess, band, kind)] = {k: sig[labels == k].mean(0) for k in config.CLASSES_4}
                            w = (times >= 0.2) & (times < 1.0)
                            m = sig[:, w].mean(-1)
                            l, r = m[labels == "left_hand"], m[labels == "right_hand"]
                            t_, p_ = stats.ttest_ind(l, r, equal_var=False)
                            sd = np.sqrt((l.var(ddof=1) + r.var(ddof=1)) / 2)
                            eog_rows.append({"subject": s, "session": sess, "band": band, "derivation": kind,
                                             "window_s": "0.2-1.0", **{f"mean_{k}_uV": float(m[labels == k].mean()) for k in config.CLASSES_4},
                                             "left_minus_right_uV": float(l.mean() - r.mean()),
                                             "cohen_d_LR": float((l.mean() - r.mean()) / sd), "welch_t": float(t_), "p_two_sided": float(p_)})
                    # high-band (30-45 Hz) power by class at the channel with the Phase 2 peak
                    if qc2 is not None:
                        pk = qc2.loc[(s, sess)]
                        hch = pk["eeg_high_peak_ch"]
                        hb = f["bp"][:, ch.index(hch), BANDS.index("high")]
                        l, r = hb[labels == "left_hand"], hb[labels == "right_hand"]
                        hb_rows.append({"subject": s, "session": sess, "channel": hch, "peak_Hz": pk["eeg_high_peak_Hz"],
                                        "peak_dB": pk["eeg_high_peak_dB"],
                                        **{f"high_log10P_{k}": float(hb[labels == k].mean()) for k in config.CLASSES_4},
                                        "kruskal_p_4class": float(stats.kruskal(*[hb[labels == k] for k in config.CLASSES_4]).pvalue),
                                        "welch_p_LR": float(stats.ttest_ind(l, r, equal_var=False).pvalue)})
            print(f"S{s} {sess} done")

    erd_df = pd.DataFrame(erd_rows)
    erd_df.to_csv(config.TABLES / "neuro_erd.csv", index=False)
    eog_df = pd.DataFrame(eog_rows)
    eog_df.to_csv(config.TABLES / "neuro_eog_evoked_stats.csv", index=False)
    pd.DataFrame(hb_rows).to_csv(config.TABLES / "neuro_high_band_by_class.csv", index=False)

    # per-subject lateralization of ERD (C3 vs C4) and BCI-inefficiency flag
    lat_rows = []
    for (s, sess, c), g in erd_df.groupby(["subject", "session", "condition"]):
        for b in ("mu", "beta"):
            gb = g[g.band == b].set_index(["label", "channel"])["erd_dB"]
            li = (gb[("right_hand", "C3")] - gb[("right_hand", "C4")]) - (gb[("left_hand", "C3")] - gb[("left_hand", "C4")])
            lat_rows.append({"subject": s, "session": sess, "condition": c, "band": b, "lateralization_dB": li,
                             "erd_C3_right": gb[("right_hand", "C3")], "erd_C4_left": gb[("left_hand", "C4")]})
    lat = pd.DataFrame(lat_rows)
    piv = lat.pivot_table(index=["subject", "session", "condition"], columns="band", values="lateralization_dB").reset_index()
    # weak / absent contralateral ERD: neither band reaches -0.5 dB lateralization
    piv["erd_weak_flag"] = (piv["mu"] > -0.5) & (piv["beta"] > -0.5)
    lat = lat.merge(piv[["subject", "session", "condition", "erd_weak_flag"]], on=["subject", "session", "condition"])
    lat.to_csv(config.TABLES / "neuro_erd_lateralization.csv", index=False)
    print(piv.round(2).to_string(index=False))
    print(eog_df.query("band == '1-45'").round(3)[["subject", "session", "derivation", "left_minus_right_uV", "cohen_d_LR", "p_two_sided"]].to_string(index=False))

    sess0 = config.CFG["sessions"][0]
    subs = config.CFG["subjects"]
    cols = {"left_hand": "C0", "right_hand": "C3", "feet": "C2", "tongue": "C4"}

    # fig_4a: ERD bars at C3/C4, left vs right, mu/beta, none vs ica (primary session)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3), sharey=True)
    for ax, (c, b) in zip(axes, [(c, b) for c in ("none", "ica") for b in ("mu", "beta")]):
        g = erd_df.query("session == @sess0 and condition == @c and band == @b and channel in ['C3','C4'] and label in @config.CLASSES_LR")
        per = g.pivot_table(index="subject", columns=["channel", "label"], values="erd_dB")
        for j, (chn, lab) in enumerate([(a, l) for a in ("C3", "C4") for l in config.CLASSES_LR]):
            v = per[(chn, lab)]
            ax.bar(j, v.mean(), yerr=v.sem() if len(v) > 1 else 0, color=cols[lab], alpha=0.8)
            ax.scatter(np.full(len(v), j) + np.linspace(-0.2, 0.2, len(v)), v, s=6, color="k", zorder=3)
        ax.set_xticks(range(4), ["C3\nleft", "C3\nright", "C4\nleft", "C4\nright"], fontsize=7)
        ax.axhline(0, color="k", lw=0.5); ax.set_title(f"{c}, {b}", fontsize=8)
    axes[0].set_ylabel("ERD (dB, MI vs baseline)\nmean ± SEM, dots = subjects")
    fig.suptitle(f"Session {sess0}: contralateral ERD expected (C3 lower for right hand, C4 lower for left)", fontsize=8)
    plots.save(fig, "fig_4a_erd_c3c4")

    # fig_4b: ERD topomaps, ica condition, grand average over subjects
    info = mne.create_info([c for c in features.load_recording(subs[0], sess0)["none"]["ch_names"] if not c.startswith("EOG")],
                           config.SFREQ, "eeg")
    info.set_montage("standard_1005")
    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    for i, b in enumerate(("mu", "beta")):
        g = erd_df.query("session == @sess0 and condition == 'ica' and band == @b").groupby(["label", "channel"])["erd_dB"].mean()
        maps = {k: g[k].reindex(info.ch_names).to_numpy() for k in config.CLASSES_LR}
        maps["right - left"] = maps["right_hand"] - maps["left_hand"]
        vmax = max(np.abs(maps["left_hand"]).max(), np.abs(maps["right_hand"]).max())
        for j, (name, m) in enumerate(maps.items()):
            v = vmax if j < 2 else np.abs(m).max()
            im, _ = mne.viz.plot_topomap(m, info, axes=axes[i, j], show=False, cmap=plots.CMAP_DIV, vlim=(-v, v), contours=0)
            axes[i, j].set_title(f"{b}: {name}", fontsize=8)
            plt.colorbar(im, ax=axes[i, j], shrink=0.6, label="dB")
    fig.suptitle(f"ERD topography, ica condition, session {sess0}, mean of {len(subs)} subject(s)", fontsize=8)
    plots.save(fig, "fig_4b_erd_topomap")

    # fig_4c: cue-locked HEOG / VEOG by class (grand average) + per-subject HEOG left-right
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.45)
    top = gs[0].subgridspec(1, 4); bot = gs[1].subgridspec(1, 9)
    for j, (band, kind) in enumerate([(b, k) for b in ("wide", "1-45") for k in ("HEOG", "VEOG")]):
        ax = fig.add_subplot(top[j])
        for k in config.CLASSES_4:
            ga = np.mean([evk[(s, sess0, band, kind)][k] for s in subs], 0)
            ax.plot(times, ga, color=cols[k], lw=0.9, label=k)
        ax.axvspan(0, config.CUE_DURATION, color="0.9", zorder=0)
        ax.axvline(0, color="k", lw=0.5); ax.set_xlim(-1, 4)
        ax.set_title(f"{kind}, {'0.5' if band == 'wide' else '1'}-45 Hz", fontsize=8)
        ax.set_xlabel("time from cue (s)", fontsize=7)
        if j == 0:
            ax.set_ylabel("µV (baseline -1..0 s)"); ax.legend(fontsize=6, frameon=False)
    for j, s in enumerate(subs[:9]):
        ax = fig.add_subplot(bot[j])
        e = evk[(s, sess0, "wide", "HEOG")]
        ax.plot(times, e["left_hand"] - e["right_hand"], color="k", lw=0.8)
        ax.axvspan(0, config.CUE_DURATION, color="0.9", zorder=0); ax.axhline(0, color="k", lw=0.4)
        row = eog_df.query("subject == @s and session == @sess0 and band == 'wide' and derivation == 'HEOG'").iloc[0]
        ax.set_title(f"S{s} d={row.cohen_d_LR:.2f}", fontsize=7); ax.set_xlim(-1, 4); ax.tick_params(labelsize=6)
        if j == 0:
            ax.set_ylabel("HEOG left - right (µV)", fontsize=7)
    fig.suptitle(f"Cue-locked EOG, session {sess0}. Grey = arrow on screen. HEOG > 0 = gaze to subject's left "
                 "(EOG1 - EOG3). d = Cohen's d, 0.2-1.0 s", fontsize=8)
    plots.save(fig, "fig_4c_eog_evoked_by_class")

    # onset latency of the class difference (confirms the cue-onset convention):
    # first t > 0 where the grand-average |left - right| HEOG exceeds baseline mean + 5 SD
    lat_rows2 = []
    for band in ("wide", "1-45"):
        for sess in config.CFG["sessions"]:
            dw = np.mean([evk[(s, sess, band, "HEOG")]["left_hand"] - evk[(s, sess, band, "HEOG")]["right_hand"] for s in subs], 0)
            b = (times >= config.BASELINE[0]) & (times < 0)
            thr_ = np.abs(dw[b]).mean() + 5 * np.abs(dw[b]).std()
            post = np.where((times > 0) & (np.abs(dw) > thr_))[0]
            pre = np.where((times < 0) & (np.abs(dw) > thr_))[0]
            lat_rows2.append({"band": band, "session": sess, "threshold_uV": thr_,
                              "onset_s": float(times[post[0]]) if len(post) else np.nan,
                              "peak_s": float(times[np.argmax(np.abs(dw) * (times > 0))]),
                              "peak_uV": float(dw[np.argmax(np.abs(dw) * (times > 0))]),
                              "n_samples_over_threshold_before_cue": int(len(pre))})
    pd.DataFrame(lat_rows2).to_csv(config.TABLES / "neuro_eog_latency.csv", index=False)
    print(pd.DataFrame(lat_rows2).round(3).to_string(index=False))

    # fig_4d: frontal EEG evoked, left - right difference wave, per cleaning condition
    fig, axes = plt.subplots(1, len(FRONTAL), figsize=(12, 2.8), sharey=True)
    for ax, chn in zip(axes, FRONTAL):
        for c, col in zip(config.CONDITIONS, ["k", "C1", "C2"]):
            dw = np.mean([evk[(s, sess0, c, chn)]["left_hand"] - evk[(s, sess0, c, chn)]["right_hand"] for s in subs], 0)
            ax.plot(times, dw, color=col, lw=0.9, label=c)
        ax.axvspan(0, config.CUE_DURATION, color="0.9", zorder=0); ax.axhline(0, color="k", lw=0.4)
        ax.set_title(chn, fontsize=8); ax.set_xlim(-1, 4); ax.set_xlabel("time from cue (s)", fontsize=7)
    axes[0].set_ylabel("left - right (µV)"); axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"Frontal EEG evoked difference (left - right hand cue), 1-45 Hz + CAR, session {sess0}", fontsize=8)
    plots.save(fig, "fig_4d_frontal_eeg_evoked_before_after_cleaning")

    # fig_4e: class-wise PSD at C3 / C4 in the MI window (ica)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for i, (ax, chn) in enumerate(zip(axes, ("C3", "C4"))):
        for k in config.CLASSES_LR:
            ga = np.mean([psd_c34[(s, sess0, "ica")][k][i] for s in subs], 0)
            ax.plot(fr, ga, color=cols[k], label=k)
        ax.set_xlim(1, 45); ax.set_title(chn, fontsize=8); ax.set_xlabel("Hz")
        for b in ("mu", "beta"):
            ax.axvspan(*config.BANDS[b], color="0.93", zorder=0)
    axes[0].set_ylabel("dB re 1 µV²/Hz"); axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"MI-window PSD by class, ica, session {sess0}, mean of {len(subs)} subject(s)", fontsize=8)
    plots.save(fig, "fig_4e_psd_c3c4_by_class")
