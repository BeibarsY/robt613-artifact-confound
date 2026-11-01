"""Classifier pipelines and grouped evaluation (leave-one-run-out), permutation chance.

Everything learned from data (StandardScaler, classifier) lives in one sklearn Pipeline that
is cloned and fitted on training folds only. Features are per-trial and stateless
(src/features.py), so they are computed once per recording.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config

# adapted from ROBT613-BCI-Workshop notebook 02 and 06 (StandardScaler + LDA / logistic
# regression pipelines, GroupKFold by run, permutation chance level)

LOGREG_C: float = 1.0  # fixed; justified by the nested-CV check in scripts/05 (decode_nested_C.csv)
C_GRID: list[float] = [0.01, 0.1, 1.0, 10.0]


def make_pipeline(clf: str, C: float = LOGREG_C) -> Pipeline:
    """StandardScaler -> classifier. ``clf`` in {'logreg', 'slda'}."""
    if clf == "logreg":
        est = LogisticRegression(C=C, l1_ratio=0.0, max_iter=2000, class_weight="balanced")
    elif clf == "slda":
        est = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    else:
        raise ValueError(clf)
    return Pipeline([("scale", StandardScaler()), ("clf", est)])


def cv_splitter() -> GroupKFold:
    """Leave-one-run-out: 6 runs per session -> 6 folds, run is the group."""
    return GroupKFold(n_splits=config.N_RUNS_PER_SESSION)


def scorers(task: str) -> dict[str, str]:
    """Scoring names per task (binary: ROC AUC; 4-class: balanced accuracy + macro OVR AUC)."""
    if task == "left_vs_right":
        return {"auc": "roc_auc"}
    return {"bacc": "balanced_accuracy", "auc": "roc_auc_ovr"}


def task_data(y_all: np.ndarray, groups_all: np.ndarray, task: str,
              mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index, encoded labels and groups of the trials used by ``task``.

    Binary: right_hand = 1, left_hand = 0. 4-class: index into config.CLASSES_4.
    """
    classes = config.CLASSES_LR if task == "left_vs_right" else config.CLASSES_4
    keep = np.isin(y_all, classes) if mask is None else (np.isin(y_all, classes) & mask)
    idx = np.where(keep)[0]
    y = np.array([classes.index(v) for v in y_all[idx]])
    return idx, y, groups_all[idx]


def permute_within_groups(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle labels within each run (keeps per-run class counts, as permutation_test_score)."""
    yp = y.copy()
    for g in np.unique(groups):
        i = np.where(groups == g)[0]
        yp[i] = y[rng.permutation(i)]
    return yp


def cv_scores(X: np.ndarray, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]],
              task: str, clf: str) -> dict[str, np.ndarray]:
    """Per-fold scores: a fresh pipeline is fitted on each training fold, scored on its test run.

    Same computation as sklearn ``cross_validate`` with ``scorers(task)``, without its
    per-call overhead (matters inside permutation loops; checked in tests/test_smoke.py).
    """
    out: dict[str, list[float]] = {m: [] for m in scorers(task)}
    for tr, te in folds:
        model = make_pipeline(clf).fit(X[tr], y[tr])
        if task == "left_vs_right":
            out["auc"].append(roc_auc_score(y[te], model.decision_function(X[te])))
        else:
            p = model.predict_proba(X[te])
            out["auc"].append(roc_auc_score(y[te], p, multi_class="ovr", average="macro", labels=model.classes_))
            out["bacc"].append(balanced_accuracy_score(y[te], model.classes_[p.argmax(1)]))
    return {m: np.asarray(v) for m, v in out.items()}


def evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, task: str, clf: str,
             n_perm: int = 0, seed: int = config.SEED, n_jobs: int = 1) -> dict:
    """Leave-one-run-out CV (+ optional within-run label permutations).

    Returns fold mean / SD of every metric, and for each metric the 95th percentile of the
    permutation null and the permutation p-value ((#null >= observed) + 1) / (n_perm + 1).
    """
    sc = scorers(task)
    folds = list(cv_splitter().split(X, y, groups))
    res = cv_scores(X, y, folds, task, clf)
    out = {"n_trials": len(y)}
    for m in sc:
        v = res[m]
        out[f"{m}_mean"], out[f"{m}_sd"] = float(v.mean()), float(v.std(ddof=1))
    if n_perm:
        # one child seed per permutation -> identical null for any n_jobs
        seeds = np.random.SeedSequence(seed).spawn(n_perm)

        def one(ss: np.random.SeedSequence) -> dict:
            yp = permute_within_groups(y, groups, np.random.default_rng(ss))
            return {m: v.mean() for m, v in cv_scores(X, yp, folds, task, clf).items()}

        res_p = Parallel(n_jobs=n_jobs)(delayed(one)(ss) for ss in seeds)
        null = {m: [d[m] for d in res_p] for m in sc}
        for m in sc:
            nm = np.asarray(null[m])
            out[f"{m}_chance_p95"] = float(np.percentile(nm, 95))
            out[f"{m}_null_mean"] = float(nm.mean())
            out[f"{m}_perm_p"] = float(((nm >= out[f"{m}_mean"]).sum() + 1) / (n_perm + 1))
    return out


def cross_session(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray, task: str,
                  clf: str, n_perm: int = 0, seed: int = config.SEED) -> dict:
    """Train on session 0train, test on 1test. Null: refit with permuted training labels."""
    def score(model: Pipeline) -> dict:
        if task == "left_vs_right":
            return {"auc": roc_auc_score(yte, model.decision_function(Xte))}
        p = model.predict_proba(Xte)
        return {"auc": roc_auc_score(yte, p, multi_class="ovr", average="macro"),
                "bacc": balanced_accuracy_score(yte, p.argmax(1))}

    out = {f"{k}": float(v) for k, v in score(clone(make_pipeline(clf)).fit(Xtr, ytr)).items()}
    out["n_train"], out["n_test"] = len(ytr), len(yte)
    if n_perm:
        rng = np.random.default_rng(seed)
        null = [score(clone(make_pipeline(clf)).fit(Xtr, rng.permutation(ytr))) for _ in range(n_perm)]
        for m in out.copy():
            if m.startswith("n_"):
                continue
            nm = np.array([d[m] for d in null])
            out[f"{m}_chance_p95"] = float(np.percentile(nm, 95))
            out[f"{m}_perm_p"] = float(((nm >= out[m]).sum() + 1) / (n_perm + 1))
    return out


def nested_C(X: np.ndarray, y: np.ndarray, groups: np.ndarray, task: str) -> dict:
    """Logistic regression with C tuned by an inner leave-one-run-out GridSearch.

    Returns the outer-fold mean AUC and the C chosen in each outer fold.
    """
    sc = "roc_auc" if task == "left_vs_right" else "roc_auc_ovr"
    aucs, chosen = [], []
    for tr, te in cv_splitter().split(X, y, groups):
        gs = GridSearchCV(make_pipeline("logreg"), {"clf__C": C_GRID}, scoring=sc,
                          cv=GroupKFold(n_splits=len(np.unique(groups[tr]))))
        gs.fit(X[tr], y[tr], groups=groups[tr])
        chosen.append(gs.best_params_["clf__C"])
        if task == "left_vs_right":
            aucs.append(roc_auc_score(y[te], gs.decision_function(X[te])))
        else:
            aucs.append(roc_auc_score(y[te], gs.predict_proba(X[te]), multi_class="ovr", average="macro"))
    return {"auc_nested_mean": float(np.mean(aucs)), "C_chosen": ";".join(f"{c:g}" for c in chosen)}


def binomial_chance(n: int, n_classes: int, alpha: float = 0.05) -> float:
    """Upper (1 - alpha) binomial bound of accuracy for a classifier guessing at 1/n_classes."""
    from scipy.stats import binom
    return float(binom.ppf(1 - alpha, n, 1 / n_classes) / n)
