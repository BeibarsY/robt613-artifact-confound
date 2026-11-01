"""Per-trial feature extraction and feature-set selection.

All extractors are stateless and per trial (no parameter is learned from other trials), so
computing them once per recording before cross-validation cannot leak information between
folds. Everything learned (StandardScaler, classifier) is fitted inside the CV pipeline.
"""
from __future__ import annotations

import mne
import numpy as np
from scipy.signal import welch

import config

# adapted from ROBT613-BCI-Workshop notebook 06 (Welch log band power) and
# notebook 02 (mean amplitude in time windows)


def log_bandpower(x: np.ndarray, sfreq: float, bands: dict[str, tuple[float, float]] = config.BANDS
                  ) -> np.ndarray:
    """log10 mean Welch PSD (µV²/Hz) per band. x: (n_trials, n_ch, n_times) in V.

    Returns (n_trials, n_ch, n_bands). Band edges are [lo, hi) on 1 Hz bins.
    """
    nper = min(config.WELCH_NPERSEG, x.shape[-1])
    f, p = welch(x * 1e6, fs=sfreq, nperseg=nper, noverlap=nper // 2, axis=-1)
    out = [p[..., (f >= lo) & (f < hi)].mean(-1) for lo, hi in bands.values()]
    return np.log10(np.stack(out, axis=-1))


def window_means(x: np.ndarray, times: np.ndarray, t0: float, t1: float, width: float,
                 baseline: tuple[float, float] = config.BASELINE) -> np.ndarray:
    """Baseline-corrected mean amplitude (µV) in consecutive bins of ``width`` s from t0 to t1.

    x: (n_trials, n_ch, n_times) in V. Returns (n_trials, n_ch, n_bins).
    """
    base = x[..., (times >= baseline[0]) & (times < baseline[1])].mean(-1, keepdims=True)
    y = (x - base) * 1e6
    edges = np.round(np.arange(t0, t1 + 1e-9, width), 6)
    return np.stack([y[..., (times >= a) & (times < b)].mean(-1) for a, b in zip(edges[:-1], edges[1:])], -1)


def epoch_features(epochs: mne.Epochs) -> dict[str, np.ndarray]:
    """All per-trial features of one recording x condition (EEG + EOG channels).

    bp      : log band power in the MI window        (n, n_ch, n_bands)
    bp_base : log band power in the fixation baseline (n, n_ch, n_bands)  [for ERD, Phase 4]
    amp     : mean amplitude, 100 ms bins 0-1 s after cue (n, n_ch, n_bins)
    """
    sf = epochs.info["sfreq"]
    t = epochs.times
    x = epochs.get_data(picks=["eeg", "eog"])
    mi = x[..., (t >= config.MI_WINDOW[0]) & (t < config.MI_WINDOW[1])]
    base = x[..., (t >= config.BASELINE[0]) & (t < config.BASELINE[1])]
    return {"bp": log_bandpower(mi, sf), "bp_base": log_bandpower(base, sf),
            "amp": window_means(x, t, *config.AMP_WINDOW, config.AMP_BIN),
            "ch_names": np.array(epochs.copy().pick(["eeg", "eog"]).ch_names)}


def select(feats: dict[str, np.ndarray], feature_set: str) -> tuple[np.ndarray, list[str]]:
    """Flatten the features of one feature set (PLAN.md Phase 5 table) -> (X, names)."""
    ch = list(feats["ch_names"])
    bands = list(config.BANDS)
    eeg = [c for c in ch if not c.startswith("EOG")]

    def bp(chans: list[str], bnames: list[str]) -> tuple[np.ndarray, list[str]]:
        ci = [ch.index(c) for c in chans]
        bi = [bands.index(b) for b in bnames]
        X = feats["bp"][:, ci][:, :, bi].reshape(len(feats["bp"]), -1)
        return X, [f"{c}:{b}" for c in chans for b in bnames]

    if feature_set == "eog_only":
        ci = [ch.index(c) for c in config.EOG_CHANNELS]
        A = feats["amp"][:, ci].reshape(len(feats["amp"]), -1)
        nb = feats["amp"].shape[-1]
        an = [f"{c}:amp{k}" for c in config.EOG_CHANNELS for k in range(nb)]
        B, bn = bp(config.EOG_CHANNELS, ["delta", "theta"])
        return np.hstack([A, B]), an + bn
    if feature_set == "eeg_all":
        return bp(eeg, bands)
    if feature_set == "eeg_frontal_amp":  # EXPLORATORY, not pre-registered (NOTES.md, Phase 7)
        ci = [ch.index(c) for c in config.FRONTAL_AMP_CHANNELS]
        A = feats["amp"][:, ci].reshape(len(feats["amp"]), -1)
        nb = feats["amp"].shape[-1]
        return A, [f"{c}:amp{k}" for c in config.FRONTAL_AMP_CHANNELS for k in range(nb)]
    if feature_set == "eeg_physio":
        return bp(config.PHYSIO_CHANNELS, ["mu", "beta"])
    if feature_set == "eeg_artifact_prone":
        return bp(config.ARTIFACT_PRONE_CHANNELS, ["delta", "high"])
    raise ValueError(feature_set)


def cache_path(subject: int, session: str) -> "Path":
    """Feature cache file of one recording (all conditions)."""
    p = config.CACHE / "features" / f"S{subject:02d}_{session}_{config.CLEAN_FIT_ON}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_recording(subject: int, session: str, epochs: dict[str, mne.Epochs]) -> None:
    """Compute and cache features for every condition of one recording."""
    meta = epochs["none"].metadata
    out = {"label": meta["label"].to_numpy().astype(str), "run": meta["run"].to_numpy(),
           "expert_artifact": meta["expert_artifact"].to_numpy()}
    for c, ep in epochs.items():
        f = epoch_features(ep)
        out["ch_names"] = f.pop("ch_names").astype(str)
        for k, v in f.items():
            out[f"{c}__{k}"] = v.astype(np.float32)
    np.savez_compressed(cache_path(subject, session), **out)


def load_recording(subject: int, session: str) -> dict:
    """Load cached features -> {'label','run','expert_artifact', cond: {bp, bp_base, amp, ch_names}}."""
    z = np.load(cache_path(subject, session), allow_pickle=False)
    out = {k: z[k] for k in ("label", "run", "expert_artifact")}
    for c in config.CONDITIONS:
        out[c] = {k: z[f"{c}__{k}"].astype(float) for k in ("bp", "bp_base", "amp")}
        out[c]["ch_names"] = z["ch_names"]
    return out


def ensure_recording(subject: int, session: str) -> dict:
    """Load cached features, building conditions/epochs first if the cache is missing."""
    if not cache_path(subject, session).exists():
        from src import data, preprocess
        runs = data.load_subject(subject)[session]
        tt = data.trial_table(subject, session, runs)
        conds, _ = preprocess.build_conditions(subject, session, runs)
        save_recording(subject, session, {c: preprocess.make_epochs(r, tt) for c, r in conds.items()})
    return load_recording(subject, session)
