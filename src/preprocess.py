"""Preprocessing: session concatenation, band-pass filtering, spectral QC.

Phase 3 adds CAR, EOG regression, ICA, epoching and rejection to this module.
"""
from __future__ import annotations

import mne
import numpy as np
import pandas as pd

import config

# adapted from ROBT613-BCI-Workshop notebook 03 (filtering)

# Analysis band. h_trans_bandwidth is set explicitly: MNE's default for a 45 Hz
# edge is 11.25 Hz (-6 dB at 50.6 Hz, only -4.5 dB at 50 Hz). 5 Hz puts 50 Hz in
# the stop band (-50 dB) at the same filter length. See fig_2b.
FILTER_KW: dict = {"l_freq": 1.0, "h_freq": 45.0, "l_trans_bandwidth": "auto",
                   "h_trans_bandwidth": 5.0, "method": "fir", "fir_design": "firwin",
                   "phase": "zero", "picks": ["eeg", "eog"]}
WELCH_KW: dict = {"method": "welch", "n_fft": 500, "n_overlap": 250, "fmin": 0.0, "fmax": 125.0}


def concat_session(runs: dict[str, mne.io.BaseRaw]) -> mne.io.BaseRaw:
    """Concatenate the 6 MI runs of a session in run order.

    MNE inserts BAD boundary / EDGE annotations at the joins, so filters are applied
    per run segment and no epoch can straddle two runs. The run index of each trial
    is recovered from ``data.trial_table`` (cue samples are per run).
    """
    ordered = [runs[k].copy() for k in sorted(runs, key=int)]
    return mne.concatenate_raws(ordered, verbose=False)


def bandpass(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Return a copy band-passed with ``FILTER_KW`` (EEG and EOG channels)."""
    return raw.copy().filter(**FILTER_KW, verbose=False)


def filter_description(sfreq: float = config.SFREQ) -> dict:
    """Design the analysis filter once and return its parameters and key gains (dB)."""
    kw = {k: v for k, v in FILTER_KW.items() if k not in ("picks", "method")}
    h = mne.filter.create_filter(None, sfreq, verbose=False, **kw)
    freqs = np.fft.rfftfreq(16384, 1 / sfreq)
    gain = 20 * np.log10(np.abs(np.fft.rfft(h, 16384)) + 1e-12)
    out = {"n_taps": len(h), "length_s": len(h) / sfreq,
           "l_freq": FILTER_KW["l_freq"], "h_freq": FILTER_KW["h_freq"],
           "h_trans_bandwidth": FILTER_KW["h_trans_bandwidth"]}
    for f in (0.5, 1.0, 45.0, 47.5, 50.0):
        out[f"gain_dB_{f:g}Hz"] = float(gain[np.argmin(np.abs(freqs - f))])
    # same band with MNE's default upper transition, for comparison (why h_trans_bandwidth is set)
    hd = mne.filter.create_filter(None, sfreq, verbose=False, **{**kw, "h_trans_bandwidth": "auto"})
    gd = 20 * np.log10(np.abs(np.fft.rfft(hd, 16384)) + 1e-12)
    out["gain_dB_50Hz_mne_default"] = float(gd[np.argmin(np.abs(freqs - 50.0))])
    return out


def psd(raw: mne.io.BaseRaw, picks: str | list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD (V^2/Hz), 0.5 Hz resolution, skipping BAD/EDGE segments."""
    sp = raw.compute_psd(picks=picks, reject_by_annotation=True, verbose=False, **WELCH_KW)
    return sp.get_data(picks="all"), sp.freqs


def spectral_qc(raw: mne.io.BaseRaw) -> dict:
    """Line-noise and drift metrics from the unfiltered PSD, EEG and EOG separately.

    line_50Hz_dB : 50 Hz bin over the median of 40-48 and 52-60 Hz bins (mean over channels).
    drift_dB     : median power below 1 Hz over median power at 1-4 Hz.
    """
    out = {}
    for kind in ("eeg", "eog"):
        p, f = psd(raw, kind)
        p = 10 * np.log10(p)
        flank = ((f >= 40) & (f <= 48)) | ((f >= 52) & (f <= 60))
        line = p[:, np.argmin(np.abs(f - 50.0))] - np.median(p[:, flank], axis=1)
        drift = np.median(p[:, (f > 0) & (f < 1)], axis=1) - np.median(p[:, (f >= 1) & (f <= 4)], axis=1)
        out[f"{kind}_line_50Hz_dB_mean"] = float(line.mean())
        out[f"{kind}_line_50Hz_dB_max"] = float(line.max())
        out[f"{kind}_drift_dB_median"] = float(np.median(drift))
        if kind == "eeg":
            out.update(high_band_peak(p, f, raw.copy().pick("eeg").ch_names))
    return out


def high_band_peak(p_db: np.ndarray, f: np.ndarray, ch_names: list[str],
                   fit: tuple[float, float] = (26.0, 44.0)) -> dict:
    """Largest narrowband excess in the 'high' feature band, per channel.

    A line is fitted to log10(f) vs dB over ``fit`` (the 1/f trend) for each channel;
    the peak is the largest residual inside config.BANDS['high'] (30-45 Hz).
    """
    m = (f >= fit[0]) & (f <= fit[1])
    x = np.log10(f[m])
    lo, hi = config.BANDS["high"]
    inband = (f[m] >= lo) & (f[m] <= hi)
    best = (np.nan, "", -np.inf)
    for ch, row in zip(ch_names, p_db):
        resid = row[m] - np.polyval(np.polyfit(x, row[m], 1), x)
        i = int(np.argmax(np.where(inband, resid, -np.inf)))
        if resid[i] > best[2]:
            best = (float(f[m][i]), ch, float(resid[i]))
    return {"eeg_high_peak_Hz": best[0], "eeg_high_peak_ch": best[1], "eeg_high_peak_dB": best[2]}


# ====================================================================== Phase 3
# adapted from ROBT613-BCI-Workshop notebook 04 (CAR, regression, ICA fitted on calibration)
# and notebook 05 (cue-locked epochs, peak-to-peak rejection)


def car(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Common average reference over the 22 EEG channels (in place, EOG untouched)."""
    raw.set_eeg_reference("average", ch_type="eeg", projection=False, verbose=False)
    return raw


def prepare(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """The `none` pipeline: 1-45 Hz band-pass then CAR. Returns a new object."""
    return car(bandpass(raw))


def cleaning_fit_raw(subject: int, session: str, mi_raw: mne.io.BaseRaw,
                     ref_run: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Prepared data the cleaning models are fitted on (see config.CLEAN_FIT_ON)."""
    from src import data
    if config.CLEAN_FIT_ON == "session":
        return mi_raw
    if config.CLEAN_FIT_ON != "calibration":
        raise ValueError(config.CLEAN_FIT_ON)
    calib = [prepare(r) for r in data.load_calibration_raws(subject, session, ref_run)]
    return mne.concatenate_raws(calib, verbose=False) if len(calib) > 1 else calib[0]


def fit_eog_regression(fit_raw: mne.io.BaseRaw) -> mne.preprocessing.EOGRegression:
    """Least-squares regression of the 3 EOG channels out of every EEG channel."""
    return mne.preprocessing.EOGRegression(picks="eeg", picks_artifact="eog").fit(fit_raw)


def fit_ica(fit_raw: mne.io.BaseRaw) -> tuple[mne.preprocessing.ICA, list[int], np.ndarray]:
    """FastICA on EEG; components with |r| >= threshold to any EOG channel are excluded.

    Returns the ICA (with ``exclude`` set), the excluded indices and the |score| matrix
    (n_eog_channels x n_components) used for the decision.
    """
    ica = mne.preprocessing.ICA(**config.ICA_KW)
    ica.fit(fit_raw, picks="eeg", decim=config.ICA_DECIM, verbose=False)
    bads, scores = ica.find_bads_eog(fit_raw, ch_name=config.EOG_CHANNELS,
                                     measure=config.ICA_EOG_MEASURE,
                                     threshold=config.ICA_EOG_THRESHOLD, verbose=False)
    ica.exclude = sorted(int(b) for b in bads)
    return ica, ica.exclude, np.abs(np.atleast_2d(scores))


def build_conditions(subject: int, session: str, runs: dict[str, mne.io.BaseRaw],
                     use_cache: bool = True) -> tuple[dict[str, mne.io.BaseRaw], dict]:
    """Return {'none', 'eog_reg', 'ica'} prepared session Raws and a fit-info dict.

    The ICA solution is cached to results/cache/ica/ so later scripts apply the same
    decomposition. EOG regression is closed-form and refitted (deterministic).
    """
    none = prepare(concat_session(runs))
    fit_raw = cleaning_fit_raw(subject, session, none, runs["0"])
    reg = fit_eog_regression(fit_raw)
    eog_reg = reg.apply(none.copy())

    path = config.CACHE / "ica" / f"S{subject:02d}_{session}_{config.CLEAN_FIT_ON}-ica.fif"
    path.parent.mkdir(parents=True, exist_ok=True)
    spath = path.with_suffix(".scores.npy")
    if use_cache and path.exists() and spath.exists():
        ica = mne.preprocessing.read_ica(path, verbose=False)
        scores = np.load(spath)
    else:
        ica, _, scores = fit_ica(fit_raw)
        ica.save(path, overwrite=True, verbose=False)
        np.save(spath, scores)
    ica_raw = ica.apply(none.copy(), verbose=False)

    info = {"fit_on": config.CLEAN_FIT_ON, "fit_duration_s": fit_raw.n_times / fit_raw.info["sfreq"],
            "fit_samples_per_comp2": fit_raw.n_times / config.ICA_DECIM / max(ica.n_components_, 1) ** 2,
            "n_ica_components": int(ica.n_components_), "ica_excluded": [int(k) for k in ica.exclude],
            "ica_eog_scores": scores, "eog_reg_coef_max": float(np.abs(reg.coef_).max()),
            "ica": ica, "fit_raw": fit_raw}
    return {"none": none, "eog_reg": eog_reg, "ica": ica_raw}, info


def make_epochs(raw: mne.io.BaseRaw, trials: pd.DataFrame) -> mne.Epochs:
    """Cue-locked epochs (config.EPOCH_TMIN..TMAX), no baseline, trial table as metadata.

    Annotations sit at cue onset (verified in Phase 1). Raises if any trial is lost at
    a run boundary, so every condition has exactly the trials in ``trials``.
    """
    events, _ = mne.events_from_annotations(raw, event_id=config.EVENT_ID, verbose=False)
    ep = mne.Epochs(raw, events, config.EVENT_ID, config.EPOCH_TMIN, config.EPOCH_TMAX,
                    baseline=None, preload=True, reject_by_annotation=True,
                    picks=["eeg", "eog"], verbose=False)
    if len(ep) != len(trials):
        raise ValueError(f"{len(ep)} epochs vs {len(trials)} trials (lost at a boundary?)")
    inv = {v: k for k, v in config.EVENT_ID.items()}
    if [inv[c] for c in ep.events[:, 2]] != list(trials["label"]):
        raise ValueError("epoch order differs from trial table")
    ep.metadata = trials.reset_index(drop=True)
    return ep


def ptp_uv(epochs: mne.Epochs, picks: str = "eeg") -> np.ndarray:
    """Per-epoch peak-to-peak (µV), max over channels of ``picks``, whole epoch window."""
    x = epochs.get_data(picks=picks)
    return np.ptp(x, axis=-1).max(axis=1) * 1e6


def eog_corr(raw: mne.io.BaseRaw, ch: str = "Fz") -> dict:
    """|Pearson r| between one EEG channel and each EOG channel over the session."""
    x = raw.get_data(picks=[ch] + config.EOG_CHANNELS, reject_by_annotation="omit")
    return {f"absr_{ch}_{e}": float(abs(np.corrcoef(x[0], x[i + 1])[0, 1]))
            for i, e in enumerate(config.EOG_CHANNELS)}
