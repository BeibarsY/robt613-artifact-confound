"""SMOKE checks. Phase 1: audit facts. Later phases add leakage and epoch-set tests."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import config
from src import data


def test_audit_subject1_session0() -> None:
    """Trial table for S1/0train matches the dataset description."""
    runs = data.load_subject(1)["0train"]
    tt = data.trial_table(1, "0train", runs)  # raises if annotations are not at cue onset
    assert len(runs) == config.N_RUNS_PER_SESSION
    assert len(tt) == config.N_RUNS_PER_SESSION * config.N_TRIALS_PER_RUN
    assert (tt["label"].value_counts() == 72).all()
    raw = runs["0"]
    assert raw.info["sfreq"] == config.SFREQ
    assert raw.get_channel_types().count("eeg") == config.N_EEG
    assert 1e-7 < np.std(raw.get_data(picks="eeg")) < 1e-3  # Volts, not µV


def test_filter_design() -> None:
    """Analysis filter: flat passband, 50 Hz in the stop band (justifies no separate notch)."""
    from src import preprocess
    d = preprocess.filter_description()
    assert abs(d["gain_dB_1Hz"]) < 0.1 and abs(d["gain_dB_45Hz"]) < 0.1
    assert d["gain_dB_50Hz"] < -40


def test_same_epoch_set_across_conditions() -> None:
    """All cleaning conditions yield the same trials, in the same order, with the same labels."""
    from src import preprocess
    runs = data.load_subject(1)["0train"]
    tt = data.trial_table(1, "0train", runs)
    conds, info = preprocess.build_conditions(1, "0train", runs)
    eps = {c: preprocess.make_epochs(r, tt) for c, r in conds.items()}
    ref = eps["none"]
    for c, ep in eps.items():
        assert len(ep) == len(tt)
        assert np.array_equal(ep.events, ref.events), c
        assert ep.metadata.equals(ref.metadata), c
    # cleaning changes EEG only: EOG channels identical, EEG not
    assert np.allclose(eps["ica"].get_data(picks="eog"), ref.get_data(picks="eog"))
    assert np.allclose(eps["eog_reg"].get_data(picks="eog"), ref.get_data(picks="eog"))
    assert not np.allclose(eps["ica"].get_data(picks="eeg"), ref.get_data(picks="eeg"))
    assert info["ica_excluded"], "ICA excluded no component on S1"


def test_cleaning_fit_excludes_mi_trials() -> None:
    """With CLEAN_FIT_ON='calibration' the cleaning models never see an MI sample."""
    import config
    from src import preprocess
    runs = data.load_subject(1)["0train"]
    _, info = preprocess.build_conditions(1, "0train", runs)
    assert config.CLEAN_FIT_ON == "calibration"
    fit_raw = info["fit_raw"]
    assert not set(fit_raw.annotations.description) & set(config.EVENT_ID)  # no MI cues in fit data


class _FitRecorder:
    """Transformer that records which samples it was fitted on (leakage probe)."""

    seen: list = []

    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        _FitRecorder.seen.append(set(np.round(X[:, -1]).astype(int)))
        return self

    def transform(self, X):
        return X[:, :-1]

    def get_params(self, deep=True):
        return {}

    def set_params(self, **p):
        return self


def test_no_leakage_in_cv() -> None:
    """Every learned step is fitted on training-run trials only, never on the test run."""
    from sklearn.model_selection import cross_validate
    from sklearn.pipeline import Pipeline
    from src import decode, features
    r = features.ensure_recording(1, "0train")
    idx, y, g = decode.task_data(r["label"], r["run"], "left_vs_right")
    X, _ = features.select(r["none"], "eeg_all")
    X = np.hstack([X[idx], np.arange(len(idx))[:, None]])  # last column = trial id
    pipe = decode.make_pipeline("logreg")
    probe = Pipeline([("probe", _FitRecorder())] + pipe.steps)
    _FitRecorder.seen = []
    folds = list(decode.cv_splitter().split(X, y, g))
    cross_validate(probe, X, y, groups=g, cv=decode.cv_splitter(), scoring="roc_auc")
    assert len(_FitRecorder.seen) == len(folds)
    for seen, (tr, te) in zip(_FitRecorder.seen, folds):
        assert seen == set(tr) and not seen & set(te)
        assert len(np.unique(g[te])) == 1  # one whole run per test fold


def test_features_are_per_trial() -> None:
    """Band power of a trial does not change when other trials change (stateless extraction)."""
    from src import features
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 3, 500)) * 1e-5
    a = features.log_bandpower(x, 250.0)
    x2 = x.copy(); x2[1:] *= 10
    assert np.allclose(features.log_bandpower(x2, 250.0)[0], a[0])


def test_permutation_keeps_run_class_counts() -> None:
    from src import decode
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], 12)[None].repeat(6, 0).ravel()
    g = np.repeat(np.arange(6), 24)
    yp = decode.permute_within_groups(y, g, rng)
    for k in range(6):
        assert yp[g == k].sum() == y[g == k].sum()
    assert not np.array_equal(yp, y)


def test_lean_cv_matches_sklearn() -> None:
    """decode.cv_scores gives the same fold scores as sklearn cross_validate."""
    from sklearn.model_selection import cross_validate
    from src import decode, features
    r = features.ensure_recording(1, "0train")
    for task in ("left_vs_right", "four_class"):
        idx, y, g = decode.task_data(r["label"], r["run"], task)
        X, _ = features.select(r["none"], "eeg_physio")
        a = decode.cv_scores(X[idx], y, list(decode.cv_splitter().split(X[idx], y, g)), task, "logreg")
        b = cross_validate(decode.make_pipeline("logreg"), X[idx], y, groups=g, cv=decode.cv_splitter(),
                           scoring=decode.scorers(task))
        for m in a:
            assert np.allclose(a[m], b[f"test_{m}"])
