"""Write report/numbers.tex: every number quoted in the report, read from results/tables/.

The report text uses only these macros, so no number is typed by hand (PLAN.md 0.4).
Run after the notebook / scripts:  python slide_pdf/report/make_numbers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # project root (report now lives in slide_pdf/report)
sys.path.insert(0, str(ROOT))
T = ROOT / "results" / "tables"
macros: dict[str, str] = {}


def put(name: str, value, fmt: str = "{:.2f}") -> None:
    """Register \\name -> formatted value (name must be letters only)."""
    assert name.isalpha(), name
    macros[name] = value if isinstance(value, str) else fmt.format(value)


def p_fmt(p: float) -> str:
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


# ---- stage 1
a = pd.read_csv(T / "audit_summary.csv")
put("NRecordings", len(a), "{:d}")
put("NTrialsSession", int(a.n_trials.iloc[0]), "{:d}")
put("NTrialsTotal", int(a.n_trials.sum()), "{:d}")
put("ExpertMin", int(a.n_expert_artifact.min()), "{:d}")
put("ExpertMax", int(a.n_expert_artifact.max()), "{:d}")
put("ExpertTotal", int(a.n_expert_artifact.sum()), "{:d}")
put("ExpertPct", 100 * a.n_expert_artifact.sum() / a.n_trials.sum(), "{:.1f}")
cal = pd.read_csv(T / "audit_calibration_runs.csv")
put("CalibAFourT", cal.query("subject == 4 and session == '0train'").duration_s.sum(), "{:.0f}")

# ---- stage 2
f = pd.read_csv(T / "signal_filter_params.csv").iloc[0]
put("FilterTaps", int(f.n_taps), "{:d}")
put("FilterLenS", f.length_s, "{:.1f}")
put("GainFiftyDefault", f["gain_dB_50Hz_mne_default"], "{:.1f}")
put("GainFifty", f["gain_dB_50Hz"], "{:.0f}")
q = pd.read_csv(T / "signal_psd_qc.csv")
put("LineMax", q.eeg_line_50Hz_dB_mean.max(), "{:.0f}")
i = q.eeg_line_50Hz_dB_mean.idxmax()
put("LineMaxRec", f"S{q.subject[i]} {q.session[i]}")

# ---- stage 3
cf = pd.read_csv(T / "qc_cleaning_fit.csv")
put("IcaExclTwo", int((cf.n_ica_excluded == 2).sum()), "{:d}")
put("IcaExclMax", int(cf.n_ica_excluded.max()), "{:d}")
put("IcaMinScore", cf.ica_max_eog_score_excluded.min(), "{:.2f}")
ec = pd.read_csv(T / "qc_eog_correlation.csv")
for c, n in (("none", "None"), ("eog_reg", "Reg"), ("ica", "Ica")):
    put(f"FzEogR{n}", ec.query("condition == @c").absr_Fz_EOG2.median())
pt = pd.read_csv(T / "qc_ptp_per_epoch.csv")
put("PtpMax", pt.ptp_none_uV.max(), "{:.0f}")
put("PtpPctNineNineFive", np.percentile(pt.ptp_none_uV, 99.5), "{:.0f}")
inj = pt.ptp_eog_reg_uV > pt.ptp_none_uV.max()
put("RegInjected", int(inj.sum()), "{:d}")
put("RegInjectedExpert", int((inj & (pt.expert_artifact == 1)).sum()), "{:d}")
put("RegInjectedMax", pt.ptp_eog_reg_uV.max(), "{:.0f}")

# ---- stage 4
e = pd.read_csv(T / "neuro_eog_evoked_stats.csv").query("band == '1-45' and derivation == 'HEOG'")
put("HeogPos", int((e.left_minus_right_uV > 0).sum()), "{:d}")
put("HeogSig", int((e.p_two_sided < 0.001).sum()), "{:d}")
put("HeogDMax", e.cohen_d_LR.max())
lat = pd.read_csv(T / "neuro_eog_latency.csv").query("band == 'wide'")
put("HeogOnsetMin", lat.onset_s.min() * 1000, "{:.0f}")
put("HeogOnsetMax", lat.onset_s.max() * 1000, "{:.0f}")
put("HeogPeak", lat.peak_s.mean() * 1000, "{:.0f}")
el = pd.read_csv(T / "neuro_erd_lateralization.csv").drop_duplicates(["subject", "session", "condition"])
weak = el.query("condition == 'none' and erd_weak_flag")
la = pd.read_csv(T / "neuro_erd_lateralization.csv").query("condition == 'ica'")
put("LatNegAny", int(la.groupby(["subject", "session"]).lateralization_dB.min().lt(0).sum()), "{:d}")
put("LatNegBeta", int(la.query("band == 'beta'").lateralization_dB.lt(0).sum()), "{:d}")
put("WeakErd", ", ".join(f"S{r.subject} {r.session}" for r in weak.itertuples()))
hb = pd.read_csv(T / "neuro_high_band_by_class.csv")
put("SNineHighP", p_fmt(hb.query("subject == 9").welch_p_LR.max()))

# ---- stage 6
u = pd.read_csv(T / "auc_per_subject.csv")
lr = u.query("task == 'left_vs_right' and classifier == 'logreg'")
sets = {"eog_only": "Eog", "eeg_all": "All", "eeg_physio": "Physio", "eeg_artifact_prone": "Prone"}
for sess, sn in (("0train", "T"), ("1test", "E")):
    for fs, fn in sets.items():
        for c, cn in (("none", "None"), ("eog_reg", "Reg"), ("ica", "Ica")):
            g = lr.query("session == @sess and feature_set == @fs and cleaning == @c")
            if len(g):
                put(f"Auc{fn}{cn}{sn}", g.auc_mean.median())
                put(f"Above{fn}{cn}{sn}", int((g.auc_mean > g.chance_p95).sum()), "{:d}")
fc = u.query("task == 'four_class' and classifier == 'logreg' and session == '0train'")
for fs, fn in sets.items():
    for c, cn in (("none", "None"), ("ica", "Ica")):
        g = fc.query("feature_set == @fs and cleaning == @c")
        if len(g):
            put(f"Bacc{fn}{cn}", g.bacc_mean.median())
put("BaccChance", fc.bacc_chance_p95.median())
nc = pd.read_csv(T / "decode_nested_C.csv").merge(
    lr[["subject", "session", "cleaning", "feature_set", "auc_mean"]], on=["subject", "session", "cleaning", "feature_set"])
put("NestedDiff", (nc.auc_nested_mean - nc.auc_mean).abs().median(), "{:.3f}")
xs = pd.read_csv(T / "auc_cross_session.csv").query("task == 'left_vs_right' and classifier == 'logreg'")
put("XsEog", xs.query("feature_set == 'eog_only'").auc.median())
put("XsAllNone", xs.query("feature_set == 'eeg_all' and cleaning == 'none'").auc.median())
put("XsAllIca", xs.query("feature_set == 'eeg_all' and cleaning == 'ica'").auc.median())

# ---- stage 7
h1 = pd.read_csv(T / "stats_h1.csv").iloc[0]
put("HoneP", p_fmt(h1.p)); put("HoneW", h1.W, "{:.0f}"); put("HoneDelta", h1.delta_median)
put("HoneIqrLo", h1.delta_iqr_lo); put("HoneIqrHi", h1.delta_iqr_hi)
put("HoneAbove", int(h1.n_above_own_p95), "{:d}"); put("HoneN", int(h1.delta_n), "{:d}")
put("HoneVerdict", "supported" if h1.supported else "not supported")
h2 = pd.read_csv(T / "stats_h2.csv")
for r, n in zip(h2.itertuples(), ("Reg", "Ica")):
    put(f"HtwoP{n}", p_fmt(r.p)); put(f"HtwoPholm{n}", p_fmt(r.p_holm)); put(f"HtwoDelta{n}", r.delta_median, "{:.3f}")
    put(f"HtwoRho{n}", r.spearman_rho_eogAUC_drop); put(f"HtwoRhoP{n}", p_fmt(r.spearman_p))
put("HtwoVerdict", "supported" if h2.supported.any() else "not supported")
h3 = pd.read_csv(T / "stats_h3.csv")
put("HthreeP", p_fmt(h3.p[0])); put("HthreeDropPhysio", h3.median_drop_physio[0], "{:.3f}")
put("HthreeDropProne", h3.median_drop_artifact_prone[0], "{:.3f}")
put("HthreeShareNone", 100 * h3.median_share_none[1], "{:.1f}"); put("HthreeShareIca", 100 * h3.median_share_ica[1], "{:.1f}")
put("HthreeShareP", p_fmt(h3.p[1]))
put("HthreeVerdict", "supported" if h3.H3_supported[0] else "not supported")
rob = pd.read_csv(T / "stats_robustness.csv")
put("RobHoneAll", "all" if rob.query("hypothesis == 'H1'").supported.all() else "not all")
put("RobHoneN", int((rob.hypothesis == "H1").sum()), "{:d}")
put("RobHtwoAny", int(rob.query("hypothesis == 'H2'").supported.sum()), "{:d}")
put("RobHtwoN", int((rob.hypothesis == "H2").sum()), "{:d}")
ex_p = T / "stats_exploratory.csv"
if ex_p.exists():
    ex = pd.read_csv(ex_p).query("session == '0train'")
    for r, n in zip(ex.itertuples(), ("Reg", "Ica")):
        put(f"ExP{n}", p_fmt(r.p)); put(f"ExPholm{n}", p_fmt(r.p_holm)); put(f"ExAuc{n}", r.median_auc_cleaned)
        put(f"ExDelta{n}", r.delta_median, "{:.3f}"); put(f"ExRho{n}", r.spearman_rho_eogAUC_drop)
        put(f"ExRhoP{n}", p_fmt(r.spearman_p))
    put("ExAucNone", ex.median_auc_none.iloc[0])
    put("ExAboveNone", int(ex.n_above_p95_none.iloc[0]), "{:d}")

# ---- compute budget
tm = pd.read_csv(ROOT / "results" / "timing.csv").query("mode == 'FULL'").groupby("stage").seconds.last()
put("TimeDecode", tm.get("05_decode_all_conditions", np.nan) / 60, "{:.1f}")
put("NPermH", 100, "{:d}")

out = Path(__file__).resolve().parent / "numbers.tex"
out.write_text("% generated by report/make_numbers.py from results/tables -- do not edit\n" +
               "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
print(f"wrote {len(macros)} macros to {out}")
