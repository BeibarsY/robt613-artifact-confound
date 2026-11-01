"""Stage 7 - pre-registered hypothesis tests H1-H3, robustness, figures fig_7a-7d."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

import config
from src import decode, features, plots
from src import stats as st
from src.timing import timed

mne.set_log_level("ERROR")
auc = pd.read_csv(config.TABLES / "auc_per_subject.csv")
SESS0 = config.CFG["sessions"][0]
subs = config.CFG["subjects"]
BANDS = list(config.BANDS)


def h1(df: pd.DataFrame, session: str, clf: str, label: str) -> dict:
    """H1: EOG-only decodes left vs right. Wilcoxon(AUC - 0.5 > 0) and count above own p95."""
    q = df.query("task == 'left_vs_right' and feature_set == 'eog_only' and cleaning == 'none' "
                 "and session == @session and classifier == @clf").sort_values("subject")
    w = st.wilcoxon_greater(q.auc_mean.to_numpy(), 0.5)
    n_above = int((q.auc_mean > q.chance_p95).sum())
    return {"analysis": label, "session": session, "classifier": clf, "test": "wilcoxon AUC-0.5 > 0",
            **w, "n_above_own_p95": n_above,
            "supported": bool(w["p"] < 0.05 and n_above >= 5)}


def h2(df: pd.DataFrame, session: str, clf: str, label: str) -> list[dict]:
    """H2: cleaning lowers eeg_all AUC, and the drop tracks EOG-only AUC."""
    base = df.query("task == 'left_vs_right' and session == @session and classifier == @clf")
    eog = st.pick(base, feature_set="eog_only", cleaning="none")
    none = st.pick(base, feature_set="eeg_all", cleaning="none")
    out = []
    for c in ("eog_reg", "ica"):
        cl = st.pick(base, feature_set="eeg_all", cleaning=c)
        drop = none - cl
        w = st.wilcoxon_greater(none.to_numpy(), cl.to_numpy())
        sp = st.spearman(eog.to_numpy(), drop.to_numpy())
        out.append({"analysis": label, "session": session, "classifier": clf, "comparison": f"none vs {c}",
                    "test": "wilcoxon none > cleaned (eeg_all)", **w,
                    "spearman_rho_eogAUC_drop": sp["rho"], "spearman_p": sp["p_two_sided"]})
    adj = st.holm([o["p"] for o in out])
    for o, a in zip(out, adj):
        o["p_holm"] = a
        o["supported"] = bool(o["p"] < 0.05 and o["spearman_rho_eogAUC_drop"] > 0)
    return out


def weight_maps(session: str) -> pd.DataFrame:
    """Refit eeg_all logreg on ALL left/right trials of a session (visualisation only, flagged),
    per subject x cleaning. |coef| on standardised features, normalised to sum 1."""
    rows = []
    for s in subs:
        r = features.load_recording(s, session)
        idx, y, _ = decode.task_data(r["label"], r["run"], "left_vs_right")
        for c in config.CONDITIONS:
            X, names = features.select(r[c], "eeg_all")
            m = decode.make_pipeline("logreg").fit(X[idx], y)   # NOT an evaluation: full-session refit
            w = np.abs(m.named_steps["clf"].coef_.ravel())
            w = w / w.sum()
            for n, v in zip(names, w):
                ch, b = n.split(":")
                rows.append({"subject": s, "session": session, "cleaning": c, "channel": ch, "band": b, "abs_coef_norm": v})
    return pd.DataFrame(rows)


with timed("06_stats"):
    # ---------------- H1
    h1_rows = [h1(auc, SESS0, "logreg", "primary")]
    pd.DataFrame(h1_rows).to_csv(config.TABLES / "stats_h1.csv", index=False)

    # ---------------- H2
    h2_rows = h2(auc, SESS0, "logreg", "primary")
    pd.DataFrame(h2_rows).to_csv(config.TABLES / "stats_h2.csv", index=False)

    # ---------------- H3
    base = auc.query("task == 'left_vs_right' and session == @SESS0 and classifier == 'logreg'")
    drops = {fs: st.pick(base, feature_set=fs, cleaning="none") - st.pick(base, feature_set=fs, cleaning="ica")
             for fs in ("eeg_physio", "eeg_artifact_prone", "eeg_all")}
    wm = weight_maps(SESS0)
    wm.to_csv(config.TABLES / "weights_eeg_all_fullsession_refit.csv", index=False)
    share = (wm.assign(central=wm.channel.isin(config.CENTRAL_CHANNELS),
                       mubeta=wm.band.isin(["mu", "beta"]),
                       central_mubeta=wm.channel.isin(config.CENTRAL_CHANNELS) & wm.band.isin(["mu", "beta"]))
             .groupby(["subject", "cleaning"])
             .apply(lambda g: pd.Series({k: g.loc[g[k], "abs_coef_norm"].sum() for k in ("central", "mubeta", "central_mubeta")}),
                    include_groups=False).reset_index())
    share.to_csv(config.TABLES / "weights_share.csv", index=False)
    sp = share.pivot(index="subject", columns="cleaning")
    h3_rows = []
    w = st.wilcoxon_greater(drops["eeg_artifact_prone"].to_numpy(), drops["eeg_physio"].to_numpy())
    h3_rows.append({"analysis": "primary", "session": SESS0, "test": "wilcoxon drop(artifact_prone) > drop(physio), none->ica",
                    **w, "median_drop_physio": float(drops["eeg_physio"].median()),
                    "median_drop_artifact_prone": float(drops["eeg_artifact_prone"].median())})
    for k in ("central", "mubeta", "central_mubeta"):
        w = st.wilcoxon_greater(sp[(k, "ica")].to_numpy(), sp[(k, "none")].to_numpy())
        h3_rows.append({"analysis": "primary", "session": SESS0, "test": f"wilcoxon weight share {k}: ica > none", **w,
                        "median_share_none": float(sp[(k, "none")].median()), "median_share_ica": float(sp[(k, "ica")].median())})
    h3 = pd.DataFrame(h3_rows)
    supported = bool(h3.loc[0, "p"] < 0.05 and h3.loc[1, "delta_median"] > 0)
    h3["H3_supported"] = supported
    h3.to_csv(config.TABLES / "stats_h3.csv", index=False)

    # ---------------- robustness (supplementary)
    rob = []
    sess_list = config.CFG["sessions"]
    for sess in sess_list:
        for clf in config.CLASSIFIERS:
            if sess == SESS0 and clf == "logreg":
                continue
            rob.append({"hypothesis": "H1", **h1(auc, sess, clf, f"{sess}/{clf}")})
            rob += [{"hypothesis": "H2", **r} for r in h2(auc, sess, clf, f"{sess}/{clf}")]
    sens = config.TABLES / "auc_expert_artifacts_excluded.csv"
    if sens.exists():
        se = pd.read_csv(sens)
        rob.append({"hypothesis": "H1", **h1(se, SESS0, "logreg", "expert artifacts excluded")})
        rob += [{"hypothesis": "H2", **r} for r in h2(se, SESS0, "logreg", "expert artifacts excluded")]
    xs = config.TABLES / "auc_cross_session.csv"
    if xs.exists() and len(pd.read_csv(xs)):
        x = pd.read_csv(xs).rename(columns={"auc": "auc_mean", "auc_chance_p95": "chance_p95"}).assign(session="cross")
        rob.append({"hypothesis": "H1", **h1(x, "cross", "logreg", "cross-session 0train->1test")})
        rob += [{"hypothesis": "H2", **r} for r in h2(x, "cross", "logreg", "cross-session 0train->1test")]
    pd.DataFrame(rob).to_csv(config.TABLES / "stats_robustness.csv", index=False)

    # ---------------- secondary: 4-class summary
    fc = auc.query("task == 'four_class' and classifier == 'logreg'")
    sec = (fc.assign(above=lambda d: d.bacc_mean > d.bacc_chance_p95)
             .groupby(["session", "feature_set", "cleaning"])
             .agg(bacc_median=("bacc_mean", "median"), auc_ovr_median=("auc_mean", "median"),
                  n_above_p95=("above", "sum"), n=("above", "size")).reset_index())
    sec.to_csv(config.TABLES / "stats_secondary_4class.csv", index=False)

    print(pd.DataFrame(h1_rows).round(4).T.to_string()); print(pd.DataFrame(h2_rows).round(4).T.to_string())
    print(h3.round(4).T.to_string()); print(sec.round(3).to_string(index=False))
    print(pd.DataFrame(rob)[["hypothesis", "analysis", "comparison", "p", "delta_median", "n_above_own_p95",
                             "spearman_rho_eogAUC_drop", "supported"]].round(3).to_string(index=False))

    # ---------------- figures
    q = auc.query("task == 'left_vs_right' and feature_set == 'eog_only' and session == @SESS0 and classifier == 'logreg'").sort_values("subject")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(np.arange(len(q)), q.auc_mean - 0.5, bottom=0.5, color="0.5", yerr=q.auc_sd, error_kw={"lw": 0.6})
    ax.scatter(np.arange(len(q)), q.chance_p95, marker="_", s=120, color="C3", label="own permutation null, 95th pct", zorder=3)
    ax.axhline(0.5, color="k", lw=0.5)
    ax.set_xticks(range(len(q)), [f"S{s}" for s in q.subject]); ax.set_ylabel("ROC AUC (EOG only)")
    r0 = h1_rows[0]
    ax.set_title(f"H1: EOG-only left vs right, {SESS0}. Wilcoxon p = {r0['p']:.3g}, "
                 f"{r0['n_above_own_p95']}/{len(q)} above own p95", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    plots.save(fig, "fig_7a_eog_only_auc_vs_null")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    eog = st.pick(base, feature_set="eog_only", cleaning="none")
    none = st.pick(base, feature_set="eeg_all", cleaning="none")
    for ax, c, r in zip(axes, ("eog_reg", "ica"), h2_rows):
        d = none - st.pick(base, feature_set="eeg_all", cleaning=c)
        ax.scatter(eog, d, color="k", s=14)
        for s_, xv, yv in zip(eog.index, eog, d):
            ax.annotate(f"S{s_}", (xv, yv), fontsize=6, xytext=(3, 2), textcoords="offset points")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("EOG-only AUC"); ax.set_title(f"none - {c}: Wilcoxon p = {r['p']:.3g}, rho = {r['spearman_rho_eogAUC_drop']:.2f}", fontsize=8)
    axes[0].set_ylabel("AUC drop of eeg_all after cleaning")
    plots.save(fig, "fig_7b_auc_drop_vs_eog_auc")

    ch_order = [c for c in features.load_recording(subs[0], SESS0)["none"]["ch_names"] if not c.startswith("EOG")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), gridspec_kw={"height_ratios": [1.6, 1]})
    mats = {}
    for j, c in enumerate(("none", "ica")):
        m = wm.query("cleaning == @c").groupby(["channel", "band"]).abs_coef_norm.mean().unstack()[BANDS].reindex(ch_order)
        mats[c] = m
    mats["ica - none"] = mats["ica"] - mats["none"]
    for j, (k, m) in enumerate(mats.items()):
        ax = axes[0, j]
        v = np.abs(m.values).max()
        im = ax.imshow(m.values, aspect="auto", cmap=plots.CMAP_DIV if j == 2 else plots.CMAP_SEQ,
                       vmin=-v if j == 2 else 0, vmax=v)
        ax.set_xticks(range(len(BANDS)), BANDS, fontsize=7); ax.set_yticks(range(len(ch_order)), ch_order, fontsize=6)
        ax.set_title(f"mean normalised |coef|: {k}", fontsize=8); plt.colorbar(im, ax=ax, shrink=0.6)
    info = mne.create_info(ch_order, config.SFREQ, "eeg"); info.set_montage("standard_1005")
    for j, b in enumerate(("delta", "mu", "beta")):
        d = mats["ica - none"][b].to_numpy()
        im, _ = mne.viz.plot_topomap(d, info, axes=axes[1, j], show=False, cmap=plots.CMAP_DIV,
                                     vlim=(-np.abs(d).max(), np.abs(d).max()), contours=0)
        axes[1, j].set_title(f"{b}: ica - none", fontsize=8); plt.colorbar(im, ax=axes[1, j], shrink=0.6)
    fig.suptitle(f"eeg_all logistic-regression weights, full-session refit (visualisation only, not an evaluation), "
                 f"{SESS0}, mean of {len(subs)} subjects", fontsize=8)
    plots.save(fig, "fig_7c_feature_weights_before_after")

    fig, ax = plt.subplots(figsize=(5, 3.2))
    sets = ["eeg_physio", "eeg_artifact_prone", "eeg_all"]
    for i, fs in enumerate(sets):
        ax.scatter(np.full(len(drops[fs]), i) + np.linspace(-0.12, 0.12, len(drops[fs])), drops[fs], color="k", s=12)
        ax.hlines(drops[fs].median(), i - 0.25, i + 0.25, color="C3")
    for s_ in drops["eeg_physio"].index:
        ax.plot([0, 1], [drops["eeg_physio"][s_], drops["eeg_artifact_prone"][s_]], color="0.8", lw=0.6, zorder=0)
    ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(3), sets, fontsize=8)
    ax.set_ylabel("AUC drop, none - ica"); ax.set_title(f"H3a: Wilcoxon p = {h3.loc[0, 'p']:.3g} (red = median)", fontsize=8)
    plots.save(fig, "fig_7d_delta_auc_by_feature_set")

    # ---------------- EXPLORATORY (not pre-registered): frontal EEG time-domain features
    ex_path = config.TABLES / "auc_exploratory_frontal_amp.csv"
    if ex_path.exists():
        ex = pd.read_csv(ex_path)
        ex_rows = []
        for sess in config.CFG["sessions"]:
            e = ex.query("session == @sess")
            eogs = st.pick(auc.query("task == 'left_vs_right' and classifier == 'logreg' and session == @sess"),
                           feature_set="eog_only", cleaning="none")
            n0 = st.pick(e, cleaning="none")
            for c in ("eog_reg", "ica"):
                cl = st.pick(e, cleaning=c)
                w = st.wilcoxon_greater(n0.to_numpy(), cl.to_numpy())
                sp_ = st.spearman(eogs.to_numpy(), (n0 - cl).to_numpy())
                ex_rows.append({"analysis": "EXPLORATORY frontal amplitude (Fz, FC1-4, 0-1 s)", "session": sess,
                                "comparison": f"none vs {c}", **w,
                                "median_auc_none": float(n0.median()), "median_auc_cleaned": float(cl.median()),
                                "n_above_p95_none": int((e.query("cleaning == 'none'").auc_mean > e.query("cleaning == 'none'").chance_p95).sum()),
                                "n_above_p95_cleaned": int((e.query("cleaning == @c").auc_mean > e.query("cleaning == @c").chance_p95).sum()),
                                "spearman_rho_eogAUC_drop": sp_["rho"], "spearman_p": sp_["p_two_sided"]})
        exd = pd.DataFrame(ex_rows)
        exd["p_holm"] = st.holm(exd["p"].tolist())
        exd.to_csv(config.TABLES / "stats_exploratory.csv", index=False)
        print(exd.round(4).T.to_string())

        e0 = ex.query("session == @SESS0")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for i, c in enumerate(config.CONDITIONS):
            v = st.pick(e0, cleaning=c)
            ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color="k", s=12)
            ax.hlines(v.median(), i - 0.25, i + 0.25, color="C3")
        for s_ in subs:
            ax.plot(range(3), [st.pick(e0, cleaning=c)[s_] for c in config.CONDITIONS], color="0.8", lw=0.6, zorder=0)
        ax.axhline(0.5, color="k", lw=0.5); ax.set_xticks(range(3), config.CONDITIONS)
        ax.set_ylabel("ROC AUC, left vs right")
        ax.set_title(f"EXPLORATORY: frontal EEG time-domain features (Fz, FC1-4), {SESS0}", fontsize=8)
        plots.save(fig, "fig_7e_exploratory_frontal_amplitude")
