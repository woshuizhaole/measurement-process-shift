#!/usr/bin/env python
"""离线: 从 40d --save-logits 落盘的病人级预测直接算配对 Utility/AUROC CI (util_boot O(n) 快),
不必等 40d 内嵌的慢 roc_auc_score bootstrap。用法: python offline_ci_from_logits.py A2B"""
import os, sys, json, numpy as np
P = "."
sys.path.insert(0, f"{P}/code/utils")
from util_boot import paired_bootstrap_util_diff, bootstrap_util_ci
from sklearn.metrics import roc_auc_score

LOG = f"{P}/results/logits"
DIR = sys.argv[1] if len(sys.argv) > 1 else "A2B"
METHODS = ["value_only", "value_mask", "mask_only", "R1_reanchor", "R2_msr",
           "R3_dann_msr", "R4_ipw", "OPS_full", "count_only", "value_count"]


def relist(fg, arr):
    order = []; seen = set()
    for g in fg:
        if g not in seen:
            seen.add(g); order.append(g)
    idx = {g: np.where(fg == g)[0] for g in order}
    return [arr[idx[g]] for g in order]


data = {}
for m in METHODS:
    f = f"{LOG}/40d_{DIR}_{m}.npz"
    if os.path.exists(f):
        data[m] = dict(np.load(f))
print("methods loaded:", list(data.keys()))

ref = data["value_mask"]; fg = ref["tgt_fg"]
for m in data:
    assert np.array_equal(data[m]["tgt_fg"], fg), f"{m} fg mismatch!"
yl = relist(fg, ref["tgt_fy"])
plists = {m: relist(fg, data[m]["tgt_fp_ens"]) for m in data}
thr = {m: float(data[m]["thr_ens"]) for m in data}

print(f"\n=== {DIR} point (util CI + auroc), n_pat={len(yl)} ===")
pt = {}
for m in METHODS:
    if m not in data: continue
    u = bootstrap_util_ci(yl, plists[m], thr[m], B=2000, seed=0)
    a = roc_auc_score(data[m]["tgt_fy"], data[m]["tgt_fp_ens"])
    pt[m] = dict(tgt_util=round(u["point"], 4), util_lo=round(u["lo"], 4), util_hi=round(u["hi"], 4),
                 tgt_auroc=round(float(a), 4), thr=round(thr[m], 4))
    print(f"  {m:12s} util={pt[m]['tgt_util']:+.4f} CI[{pt[m]['util_lo']:+.4f},{pt[m]['util_hi']:+.4f}] auroc={pt[m]['tgt_auroc']:.4f}")

print(f"\n=== {DIR} paired Utility CI (method - baseline), B=2000 ===")
res = {}
for m in METHODS:
    if m not in data or m == "value_only": continue
    for base in ["value_only", "value_mask"]:
        if m == base or base not in data: continue
        d = paired_bootstrap_util_diff(yl, plists[base], plists[m], thr[base], thr[m], B=2000, seed=0)
        res[f"{m}_vs_{base}"] = dict(delta=round(d["point"], 4), lo=round(d["lo"], 4),
                                     hi=round(d["hi"], 4), p_gt0=round(d["p_gt0"], 3))
        flag = "  <-- WIN(CI>0)" if d["lo"] > 0 else ("  <-- CI incl 0" if d["hi"] > 0 > d["lo"] else "")
        print(f"  {m:12s} vs {base:12s}: {d['point']:+.4f} CI[{d['lo']:+.4f},{d['hi']:+.4f}] p>0={d['p_gt0']}{flag}")

json.dump({"dir": DIR, "point": pt, "paired": res}, open(f"{P}/results/stats/offline_ci_{DIR}.json", "w"), indent=1, ensure_ascii=False)
print(f"\nsaved -> results/stats/offline_ci_{DIR}.json")
