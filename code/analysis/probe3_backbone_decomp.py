#!/usr/bin/env python
"""Probe 3: backbone dependency real mechanism vs threshold/calibration artifact.
Decompose cross-hospital Utility into ranking (AUROC) vs threshold-transfer components.
Read-only on logits; writes JSON to results/diag_improve/."""
import os, sys, json, time, numpy as np
P = "."
sys.path.insert(0, f"{P}/code/utils")
from evaluate_sepsis_score import compute_prediction_utility
from util_boot import per_patient_components, _norm, paired_bootstrap_util_diff
from sklearn.metrics import roc_auc_score

DT_EARLY, DT_OPTIMAL, DT_LATE = -12, -6, 3.0
MAX_U_TP, MIN_U_FN, U_FP, U_TN = 1, -2, -0.05, 0
LOG = f"{P}/results/logits"
GRID = np.linspace(0.01, 0.6, 60)   # 方案 best_threshold grid


def _cpu(y, pred):
    return compute_prediction_utility(y, pred, DT_EARLY, DT_OPTIMAL, DT_LATE,
                                      MAX_U_TP, MIN_U_FN, U_FP, U_TN, check_errors=False)


def relist(fg, arr):
    order = []; seen = set()
    for g in fg:
        if g not in seen:
            seen.add(g); order.append(g)
    idx = {g: np.where(fg == g)[0] for g in order}
    return [arr[idx[g]] for g in order]


def best_inact(y_list):
    """threshold-independent components, computed once."""
    n = len(y_list); ub = np.empty(n); ui = np.empty(n)
    for k in range(n):
        y = np.asarray(y_list[k]).astype(int); m = len(y)
        best = np.zeros(m); inact = np.zeros(m)
        if np.any(y):
            t = int(np.argmax(y)) - DT_OPTIMAL
            best[max(0, t + DT_EARLY): min(t + int(DT_LATE) + 1, m)] = 1
        ub[k] = _cpu(y, best); ui[k] = _cpu(y, inact)
    return ub, ui


def uobs(y_list, p_list, thr):
    n = len(y_list); uo = np.empty(n)
    for k in range(n):
        y = np.asarray(y_list[k]).astype(int)
        pred = (np.asarray(p_list[k]) >= thr).astype(int)
        uo[k] = _cpu(y, pred)
    return uo


def util_at(uo, ub, ui):
    return _norm(uo, ub, ui)


def oracle(y_list, p_list, ub, ui):
    best_u = -1e9; best_t = None
    for t in GRID:
        u = _norm(uobs(y_list, p_list, t), ub, ui)
        if u > best_u:
            best_u = u; best_t = float(t)
    return best_t, float(best_u)


def prior_match_thr(tgt_p_list, target_posfrac):
    """label-free threshold transport: pick thr so target predicted-positive fraction == target_posfrac."""
    allp = np.concatenate([np.asarray(x) for x in tgt_p_list])
    # fraction of rows >= thr as function of thr; find thr matching target_posfrac
    q = 1.0 - target_posfrac
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(allp, q))


METHODS = ["value_only", "value_mask", "R4_ipw", "OPS_full", "mask_only"]
out = {"grid": [float(GRID[0]), float(GRID[-1]), len(GRID)], "gru": {}, "lgb": {}}

# ---------------- GRU (has src + tgt) ----------------
for DIR in ["A2B", "B2A"]:
    d = {m: dict(np.load(f"{LOG}/40d_{DIR}_{m}.npz")) for m in METHODS
         if os.path.exists(f"{LOG}/40d_{DIR}_{m}.npz")}
    ref = d["value_mask"]; fg = ref["tgt_fg"]; sfg = ref["src_fg"]
    yl = relist(fg, ref["tgt_fy"])
    syl = relist(sfg, ref["src_fy"])
    ub, ui = best_inact(yl)
    per = {}
    plists = {}; sposfrac = {}
    for m in d:
        assert np.array_equal(d[m]["tgt_fg"], fg)
        thr = float(d[m]["thr_ens"])
        pl = relist(fg, d[m]["tgt_fp_ens"]); plists[m] = pl
        spl = relist(sfg, d[m]["src_fp_ens"])
        uo_src = _norm  # placeholder
        # src util at thr_ens (within-domain, for reference)
        sub, sui = None, None
        tgt_auroc = float(roc_auc_score(d[m]["tgt_fy"], d[m]["tgt_fp_ens"]))
        src_auroc = float(roc_auc_score(d[m]["src_fy"], d[m]["src_fp_ens"]))
        u_src = util_at(uobs(yl, pl, thr), ub, ui)              # target util @ source thr (deployed)
        ot, uo_o = oracle(yl, pl, ub, ui)                       # target util @ oracle thr
        # label-free prior-match recalibration: match src predicted-positive frac at thr_ens
        src_pf = float((d[m]["src_fp_ens"] >= thr).mean())
        pmt = prior_match_thr(pl, src_pf)
        u_pm = util_at(uobs(yl, pl, pmt), ub, ui)
        per[m] = dict(thr_ens=round(thr, 4), src_auroc=round(src_auroc, 4),
                      tgt_auroc=round(tgt_auroc, 4), auroc_drop=round(src_auroc - tgt_auroc, 4),
                      mean_src_pred=round(float(d[m]["src_fp_ens"].mean()), 5),
                      mean_tgt_pred=round(float(d[m]["tgt_fp_ens"].mean()), 5),
                      util_srcthr=round(u_src, 4), oracle_thr=round(ot, 4),
                      util_oracle=round(uo_o, 4), thr_transfer_gap=round(uo_o - u_src, 4),
                      src_posfrac=round(src_pf, 5), priormatch_thr=round(pmt, 4),
                      util_priormatch=round(u_pm, 4))
    # paired R4 vs value_mask and value_mask vs value_only at src-thr and at oracle-thr
    pairs = {}
    def pair(mB, mA, thrB, thrA, tag):
        r = paired_bootstrap_util_diff(yl, plists[mA], plists[mB], thrA, thrB, B=2000, seed=0)
        pairs[tag] = dict(delta=round(r["point"], 4), lo=round(r["lo"], 4), hi=round(r["hi"], 4), p_gt0=round(r["p_gt0"], 3))
    if "R4_ipw" in d:
        pair("R4_ipw", "value_mask", float(d["R4_ipw"]["thr_ens"]), float(d["value_mask"]["thr_ens"]), "R4_vs_vmask@srcthr")
        pair("R4_ipw", "value_mask", per["R4_ipw"]["oracle_thr"], per["value_mask"]["oracle_thr"], "R4_vs_vmask@oracle")
    pair("value_mask", "value_only", float(d["value_mask"]["thr_ens"]), float(d["value_only"]["thr_ens"]), "vmask_vs_vonly@srcthr")
    pair("value_mask", "value_only", per["value_mask"]["oracle_thr"], per["value_only"]["oracle_thr"], "vmask_vs_vonly@oracle")
    out["gru"][DIR] = dict(n_pat=len(yl), per=per, paired=pairs)
    print(f"[GRU {DIR}] done n_pat={len(yl)}")

# ---------------- LGB (tgt only) ----------------
LGB_FOLDS = [f.replace("74_", "").replace("_value_mask.npz", "")
             for f in os.listdir(LOG) if f.startswith("74_") and f.endswith("_value_mask.npz")]
LGB_METHODS = ["value_only", "value_mask", "R4_ipw", "OPS_full"]
for fold in sorted(LGB_FOLDS):
    d = {m: dict(np.load(f"{LOG}/74_{fold}_{m}.npz")) for m in LGB_METHODS
         if os.path.exists(f"{LOG}/74_{fold}_{m}.npz")}
    if "value_mask" not in d:
        continue
    ref = d["value_mask"]; fg = ref["tgt_fg"]
    yl = relist(fg, ref["tgt_fy"])
    ub, ui = best_inact(yl)
    per = {}; plists = {}
    for m in d:
        assert np.array_equal(d[m]["tgt_fg"], fg)
        thr = float(d[m]["thr_ens"])
        pl = relist(fg, d[m]["tgt_fp_ens"]); plists[m] = pl
        tgt_auroc = float(roc_auc_score(d[m]["tgt_fy"], d[m]["tgt_fp_ens"]))
        u_src = util_at(uobs(yl, pl, thr), ub, ui)
        ot, uo_o = oracle(yl, pl, ub, ui)
        per[m] = dict(thr_ens=round(thr, 4), tgt_auroc=round(tgt_auroc, 4),
                      mean_tgt_pred=round(float(d[m]["tgt_fp_ens"].mean()), 5),
                      util_srcthr=round(u_src, 4), oracle_thr=round(ot, 4),
                      util_oracle=round(uo_o, 4), thr_transfer_gap=round(uo_o - u_src, 4))
    pairs = {}
    def pairL(mB, mA, thrB, thrA, tag):
        if mA not in d or mB not in d: return
        r = paired_bootstrap_util_diff(yl, plists[mA], plists[mB], thrA, thrB, B=2000, seed=0)
        pairs[tag] = dict(delta=round(r["point"], 4), lo=round(r["lo"], 4), hi=round(r["hi"], 4), p_gt0=round(r["p_gt0"], 3))
    pairL("R4_ipw", "value_mask", float(d.get("R4_ipw", {}).get("thr_ens", 0)), float(d["value_mask"]["thr_ens"]), "R4_vs_vmask@srcthr")
    if "R4_ipw" in d:
        pairL("R4_ipw", "value_mask", per["R4_ipw"]["oracle_thr"], per["value_mask"]["oracle_thr"], "R4_vs_vmask@oracle")
    pairL("value_mask", "value_only", float(d["value_mask"]["thr_ens"]), float(d["value_only"]["thr_ens"]), "vmask_vs_vonly@srcthr")
    pairL("value_mask", "value_only", per["value_mask"]["oracle_thr"], per["value_only"]["oracle_thr"], "vmask_vs_vonly@oracle")
    out["lgb"][fold] = dict(n_pat=len(yl), per=per, paired=pairs)
    print(f"[LGB {fold}] done n_pat={len(yl)}")

json.dump(out, open(f"{P}/results/diag_improve/probe3_backbone_decomp.json", "w"), indent=1, ensure_ascii=False)
print("saved -> results/diag_improve/probe3_backbone_decomp.json")
