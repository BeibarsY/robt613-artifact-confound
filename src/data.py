"""Data loading and audit for BNCI2014-001 (BCI Competition IV 2a).

MOABB gives the 6 MI runs per session as MNE Raw objects. It drops the EOG
calibration runs and (by default) the expert artifact flags, so both are read
back from the source .mat files that MOABB caches.
"""
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy.io as sio
from moabb.datasets import BNCI2014_001

import config

# adapted from ROBT613-BCI-Workshop notebook 00 (MOABB loading)


def get_dataset() -> BNCI2014_001:
    """Return the MOABB dataset object (artifact flags handled separately)."""
    return BNCI2014_001()


def load_subject(subject: int) -> dict[str, dict[str, mne.io.BaseRaw]]:
    """Load one subject as ``session -> run -> Raw`` (MOABB nested dict)."""
    return get_dataset().get_data(subjects=[subject])[subject]


def mat_path(subject: int, session: str) -> Path:
    """Path of the cached source .mat file for a subject and session."""
    suffix = config.SESSION_MAT_SUFFIX[session]
    for p in get_dataset().data_path(subject):
        if Path(p).name == f"A{subject:02d}{suffix}.mat":
            return Path(p)
    raise FileNotFoundError(f"A{subject:02d}{suffix}.mat not in MOABB cache")


def load_mat_runs(subject: int, session: str) -> list:
    """Read every run struct (calibration + MI) from the source .mat file."""
    m = sio.loadmat(mat_path(subject, session), squeeze_me=True, struct_as_record=False)
    runs = m["data"]
    return list(runs) if isinstance(runs, np.ndarray) else [runs]


def split_mat_runs(runs: list) -> tuple[list, list]:
    """Split .mat runs into (calibration runs without trials, MI runs with trials)."""
    calib = [r for r in runs if np.size(r.trial) == 0]
    mi = [r for r in runs if np.size(r.trial) > 0]
    return calib, mi


def trial_table(subject: int, session: str, runs: dict[str, mne.io.BaseRaw]) -> pd.DataFrame:
    """One row per trial: run, cue sample, label and the expert artifact flag.

    MOABB run ``k`` is matched to the ``k``-th .mat run that contains trials.
    The match is checked sample by sample: each annotation must sit exactly
    ``T_TRIAL_START_TO_CUE`` seconds after the .mat trial-start sample.
    """
    _, mi_runs = split_mat_runs(load_mat_runs(subject, session))
    if len(mi_runs) != len(runs):
        raise ValueError(f"S{subject} {session}: {len(mi_runs)} .mat MI runs vs {len(runs)} MOABB runs")
    shift = int(round(config.T_TRIAL_START_TO_CUE * config.SFREQ))
    rows = []
    for run_idx, (run_key, raw) in enumerate(sorted(runs.items(), key=lambda kv: int(kv[0]))):
        mr = mi_runs[run_idx]
        events, event_id = mne.events_from_annotations(raw, event_id=config.EVENT_ID, verbose=False)
        trial_start0 = np.asarray(mr.trial).ravel() - 1  # .mat samples are 1-based
        if not np.array_equal(events[:, 0] - shift, trial_start0):
            raise ValueError(f"S{subject} {session} run {run_key}: annotations are not at trial start + 2 s")
        mat_classes = [str(c).replace(" ", "_") for c in mr.classes]
        mat_labels = [mat_classes[int(y) - 1] for y in np.asarray(mr.y).ravel()]
        inv = {v: k for k, v in event_id.items()}
        moabb_labels = [inv[c] for c in events[:, 2]]
        if mat_labels != moabb_labels:
            raise ValueError(f"S{subject} {session} run {run_key}: labels differ between .mat and MOABB")
        artifacts = np.asarray(mr.artifacts).ravel().astype(int)
        for i, (samp, lab, art) in enumerate(zip(events[:, 0], moabb_labels, artifacts)):
            rows.append({"subject": subject, "session": session, "run": int(run_key),
                         "trial_in_run": i, "cue_sample": int(samp), "cue_time_s": samp / config.SFREQ,
                         "label": lab, "expert_artifact": int(art)})
    return pd.DataFrame(rows)


def calibration_summary(subject: int, session: str) -> list[dict]:
    """Describe the EOG calibration runs kept in the .mat file (MOABB drops them)."""
    calib, _ = split_mat_runs(load_mat_runs(subject, session))
    return [{"subject": subject, "session": session, "calib_run": i,
             "n_samples": int(r.X.shape[0]), "duration_s": r.X.shape[0] / float(r.fs)}
            for i, r in enumerate(calib)]


def load_calibration_raws(subject: int, session: str,
                          ref: mne.io.BaseRaw | None = None) -> list[mne.io.RawArray]:
    """Return the EOG calibration runs as Raw objects (Volts, MOABB channel names).

    ``ref`` is any MOABB MI run of the subject; it supplies channel names, types and
    montage (loaded if not given). The .mat stores µV, MOABB rescales to V the same way.
    """
    calib, _ = split_mat_runs(load_mat_runs(subject, session))
    if ref is None:
        ref = load_subject(subject)[session]["0"]
    names = [c for c in ref.ch_names if c != "STI"]
    types = [t for c, t in zip(ref.ch_names, ref.get_channel_types()) if c != "STI"]
    out = []
    for r in calib:
        info = mne.create_info(names, float(r.fs), types)
        raw = mne.io.RawArray(np.asarray(r.X, float).T * 1e-6, info, verbose=False)
        raw.set_montage(ref.get_montage())
        out.append(raw)
    return out


def recording_summary(subject: int, session: str, runs: dict[str, mne.io.BaseRaw],
                      trials: pd.DataFrame) -> dict:
    """One audit row per subject x session, every field read from the data."""
    raw0 = runs["0"]
    types = raw0.get_channel_types()
    eeg_std = float(np.median([r.get_data(picks="eeg").std() for r in runs.values()]))
    eog_std = float(np.median([r.get_data(picks="eog").std() for r in runs.values()]))
    calib = calibration_summary(subject, session)
    return {
        "subject": subject, "session": session, "n_runs": len(runs),
        "n_eeg": types.count("eeg"), "n_eog": types.count("eog"), "n_stim": types.count("stim"),
        "sfreq": raw0.info["sfreq"],
        "duration_s": float(sum(r.n_times / r.info["sfreq"] for r in runs.values())),
        "n_trials": len(trials),
        "n_expert_artifact": int(trials["expert_artifact"].sum()),
        "eeg_std_V": eeg_std, "eog_std_V": eog_std,
        "units_ok": bool(1e-7 < eeg_std < 1e-3),
        "montage_set": raw0.get_montage() is not None,
        "n_calib_runs": len(calib),
        "calib_durations_s": ";".join(f"{c['duration_s']:.1f}" for c in calib),
        "annot_at_cue": True,  # trial_table raises otherwise
        "reference": f"{config.REFERENCE} (source: Brunner2008)",
        "acq_filter": f"{config.ACQ_BANDPASS_HZ[0]}-{config.ACQ_BANDPASS_HZ[1]} Hz + {config.ACQ_NOTCH_HZ} Hz notch (source: Brunner2008)",
    }
