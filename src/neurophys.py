"""Neurophysiological checks before decoding: ERD over C3/C4 and cue-locked EOG by class."""
from __future__ import annotations

import numpy as np

import config

# adapted from ROBT613-BCI-Workshop notebook 06 (ERD as log band-power change vs baseline)
# and notebook 01 (class-wise averages of epochs)


def eog_derivations(x: np.ndarray, ch_names: list[str]) -> dict[str, np.ndarray]:
    """HEOG and VEOG (same unit as x) from monopolar EOG1-3. x: (..., n_ch, n_times)."""
    i = {c: ch_names.index(c) for c in config.EOG_CHANNELS}
    a, b = config.EOG_HEOG
    v, (l, r) = config.EOG_VEOG
    return {"HEOG": x[..., i[a], :] - x[..., i[b], :],
            "VEOG": x[..., i[v], :] - 0.5 * (x[..., i[l], :] + x[..., i[r], :])}


def erd_db(bp: np.ndarray, bp_base: np.ndarray) -> np.ndarray:
    """Per-trial ERD/ERS in dB: 10*(log10 P_MI - log10 P_baseline). Negative = desynchronisation."""
    return 10.0 * (bp - bp_base)


def lateralization(erd_mean: dict[str, np.ndarray], ch_names: list[str], band: int) -> float:
    """[ERD_C3 - ERD_C4](right hand) - [ERD_C3 - ERD_C4](left hand), dB. Expected < 0
    (contralateral desynchronisation). erd_mean[class]: (n_ch, n_bands) trial-mean ERD."""
    c3, c4 = ch_names.index("C3"), ch_names.index("C4")
    d = {k: erd_mean[k][c3, band] - erd_mean[k][c4, band] for k in config.CLASSES_LR}
    return float(d["right_hand"] - d["left_hand"])
