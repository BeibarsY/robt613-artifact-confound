"""Stages 5-6 - features, classifiers, honest evaluation over the full condition grid.

task x cleaning x feature_set x classifier, leave-one-run-out CV within session, permutation
chance for every row, cross-session transfer, nested-C check, expert-artifact sensitivity,
4-class confusion matrices. Writes auc_*.csv / decode_*.csv and fig_6a/6b.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

import config
from src import decode, features, plots
from src.timing import timed

print(f"MODE={config.MODE} subjects={config.CFG['subjects']} perms={config.CFG['n_permutations']}/"
      f"{config.CFG['n_permutations_other']} n_jobs={config.N_JOBS}")


def grid() -> list[tuple[str, str]]:
    """(cleaning, feature_set) pairs. EOG channels are never cleaned -> eog_only only with none."""
    return [("none", "eog_only")] + [(c, f) for f in config.FEATURE_SETS[1:] for c in config.CONDITIONS]


def n_perm(task: str, fs: str) -> int:
    """Permutations: full count for the rows the H1 decision rule uses, reduced elsewhere."""
    h1_row = task == "left_vs_right" and fs == "eog_only"
    return config.CFG["n_permutations"] if h1_row else config.CFG["n_permutations_other"]


rows, xs_rows, nested_rows, sens_rows, expl_rows = [], [], [], [], []
cm_sum: dict[tuple, np.ndarray] = {}
with timed("05_decode_all_conditions"):
    feats = {(s, sess): features.ensure_recording(s, sess)
             for s in config.CFG["subjects"] for sess in config.CFG["sessions"]}
    for (s, sess), r in feats.items():
        for task in config.TASKS:
            idx, y, g = decode.task_data(r["label"], r["run"], task)
            for cleaning, fs in grid():
                X, _ = features.select(r[cleaning], fs)
                for clf in config.TASK_CLASSIFIERS[task]:
                    o = decode.evaluate(X[idx], y, g, task, clf, n_perm=n_perm(task, fs), n_jobs=config.N_JOBS)
                    rows.append({"subject": s, "session": sess, "task": task, "cleaning": cleaning,
                                 "feature_set": fs, "classifier": clf, "n_features": X.shape[1],
                                 "n_permutations": n_perm(task, fs), **o,
                                 "chance_p95": o["auc_chance_p95"],
                                 "binomial_acc_p95": decode.binomial_chance(len(y), len(np.unique(y)))})
                if task == "four_class" and sess == config.CFG["sessions"][0]:
                    pred = cross_val_predict(decode.make_pipeline("logreg"), X[idx], y, groups=g, cv=decode.cv_splitter())
                    cm_sum[(cleaning, fs)] = cm_sum.get((cleaning, fs), 0) + confusion_matrix(y, pred, labels=range(4))
                if task == "left_vs_right" and sess == config.CFG["sessions"][0] and fs == "eeg_all":
                    nested_rows.append({"subject": s, "session": sess, "cleaning": cleaning, "feature_set": fs,
                                        **decode.nested_C(X[idx], y, g, task)})
                if task == "left_vs_right" and sess == config.CFG["sessions"][0]:
                    # sensitivity: drop expert-flagged artifact trials (Phase 3 note)
                    idx2, y2, g2 = decode.task_data(r["label"], r["run"], task, mask=r["expert_artifact"] == 0)
                    o2 = decode.evaluate(X[idx2], y2, g2, task, "logreg", n_perm=n_perm(task, fs), n_jobs=config.N_JOBS)
                    sens_rows.append({"subject": s, "session": sess, "task": task, "cleaning": cleaning,
                                      "feature_set": fs, "classifier": "logreg", **o2,
                                      "chance_p95": o2["auc_chance_p95"]})
        # EXPLORATORY (not pre-registered): time-domain features on frontal EEG, left vs right
        idx, y, g = decode.task_data(r["label"], r["run"], "left_vs_right")
        for cleaning in config.CONDITIONS:
            X, _ = features.select(r[cleaning], "eeg_frontal_amp")
            o = decode.evaluate(X[idx], y, g, "left_vs_right", "logreg",
                                n_perm=config.CFG["n_permutations_other"], n_jobs=config.N_JOBS)
            expl_rows.append({"subject": s, "session": sess, "task": "left_vs_right", "cleaning": cleaning,
                              "feature_set": "eeg_frontal_amp", "classifier": "logreg", "n_features": X.shape[1], **o,
                              "chance_p95": o["auc_chance_p95"]})
        print(f"S{s} {sess} done", flush=True)

    # cross-session: train 0train -> test 1test
    if {"0train", "1test"} <= set(config.CFG["sessions"]):
        for s in config.CFG["subjects"]:
            a, b = feats[(s, "0train")], feats[(s, "1test")]
            for task in config.TASKS:
                ia, ya, _ = decode.task_data(a["label"], a["run"], task)
                ib, yb, _ = decode.task_data(b["label"], b["run"], task)
                for cleaning, fs in grid():
                    Xa, _ = features.select(a[cleaning], fs)
                    Xb, _ = features.select(b[cleaning], fs)
                    for clf in config.TASK_CLASSIFIERS[task]:
                        o = decode.cross_session(Xa[ia], ya, Xb[ib], yb, task, clf, n_perm=n_perm(task, fs))
                        xs_rows.append({"subject": s, "task": task, "cleaning": cleaning, "feature_set": fs,
                                        "classifier": clf, **o})

    df = pd.DataFrame(rows)
    df.to_csv(config.TABLES / "auc_per_subject.csv", index=False)
    pd.DataFrame(xs_rows).to_csv(config.TABLES / "auc_cross_session.csv", index=False)
    pd.DataFrame(nested_rows).to_csv(config.TABLES / "decode_nested_C.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(config.TABLES / "auc_expert_artifacts_excluded.csv", index=False)
    pd.DataFrame(expl_rows).to_csv(config.TABLES / "auc_exploratory_frontal_amp.csv", index=False)
    cm_rows = [{"cleaning": c, "feature_set": f, "true": config.CLASSES_4[i], **{f"pred_{config.CLASSES_4[j]}": int(m[i, j]) for j in range(4)}}
               for (c, f), m in cm_sum.items() for i in range(4)]
    pd.DataFrame(cm_rows).to_csv(config.TABLES / "decode_confusion_4class.csv", index=False)

    summ = (df.query("task == 'left_vs_right' and classifier == 'logreg'")
            .assign(above=lambda d: d.auc_mean > d.chance_p95)
            .groupby(["session", "feature_set", "cleaning"])
            .agg(auc_median=("auc_mean", "median"), n_above_p95=("above", "sum"), n=("above", "size")))
    print(summ.round(3).to_string())
    nC = pd.DataFrame(nested_rows)
    if len(nC):
        print("nested-C vs fixed C=1 (left vs right, logreg):")
        m = nC.merge(df.query("task=='left_vs_right' and classifier=='logreg'")[["subject", "session", "cleaning", "feature_set", "auc_mean"]],
                     on=["subject", "session", "cleaning", "feature_set"])
        print("median |nested - fixed| AUC:", round(float((m.auc_nested_mean - m.auc_mean).abs().median()), 3),
              "| C chosen counts:", pd.Series(";".join(nC.C_chosen).split(";")).value_counts().to_dict())

    # fig_6a: per-subject AUC by cleaning, one panel per feature set (left vs right, logreg)
    sess0 = config.CFG["sessions"][0]
    d0 = df.query("task == 'left_vs_right' and classifier == 'logreg' and session == @sess0")
    subs = config.CFG["subjects"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    cols = {"none": "k", "eog_reg": "C1", "ica": "C2"}
    for ax, fs in zip(axes, config.FEATURE_SETS):
        conds = ["none"] if fs == "eog_only" else config.CONDITIONS
        w = 0.8 / len(conds)
        for j, c in enumerate(conds):
            g = d0.query("feature_set == @fs and cleaning == @c").set_index("subject").reindex(subs)
            x = np.arange(len(subs)) + (j - (len(conds) - 1) / 2) * w
            ax.bar(x, g.auc_mean - 0.5, w, bottom=0.5, color=cols[c], label=c, yerr=g.auc_sd, error_kw={"lw": 0.5})
            ax.scatter(x, g.chance_p95, marker="_", color="C3", s=40, zorder=3, label="chance p95" if j == 0 else None)
        ax.axhline(0.5, color="k", lw=0.5); ax.set_title(fs, fontsize=9)
        ax.set_xticks(range(len(subs)), [f"S{s}" for s in subs], fontsize=7)
        ax.legend(fontsize=6, frameon=False)
    axes[0].set_ylabel("ROC AUC, leave-one-run-out\nmean ± SD over folds")
    fig.suptitle(f"Left vs right hand, logistic regression, session {sess0}", fontsize=9)
    plots.save(fig, "fig_6a_auc_per_subject_by_condition")

    # fig_6b: 4-class confusion matrices summed over subjects (row-normalised)
    keys = [("none", "eog_only"), ("none", "eeg_all"), ("ica", "eeg_all"), ("ica", "eeg_physio")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    for ax, k in zip(axes, keys):
        m = cm_sum[k] / cm_sum[k].sum(1, keepdims=True)
        ax.imshow(m, vmin=0, vmax=1, cmap=plots.CMAP_SEQ)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if m[i, j] < 0.5 else "k")
        ax.set_xticks(range(4), [c.split("_")[0] for c in config.CLASSES_4], fontsize=7)
        ax.set_yticks(range(4), [c.split("_")[0] for c in config.CLASSES_4], fontsize=7)
        ax.set_title(f"{k[1]} / {k[0]}", fontsize=8); ax.set_xlabel("predicted", fontsize=7)
    axes[0].set_ylabel("true", fontsize=7)
    fig.suptitle(f"4-class, logistic regression, leave-one-run-out, session {sess0}, summed over {len(subs)} subject(s)", fontsize=8)
    plots.save(fig, "fig_6b_confusion_4class")
