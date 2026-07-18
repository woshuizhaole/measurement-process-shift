#!/usr/bin/env python
"""离线独立核验: 从 74_{fold}_{method}.npz 病人级 logits 重算 Utility 点估计 + 配对 CI,
与 74 脚本内落盘的 stage7_lodo_utility.json 交叉核对 (独立复算, 防止聚合 bug)。
用法: python offline_ci_74.py            # 全部折
      python offline_ci_74.py LODO_A-MICU
"""
import os, sys, json, glob, numpy as np
P = "."
sys.path.insert(0, f"{P}/code/utils")
from util_boot import paired_bootstrap_util_diff, bootstrap_util_ci
from sklearn.metrics import roc_auc_score

LOG = f"{P}/results/logits"
METHODS = ["value_only", "value_mask", "R4_ipw", "OPS_full"]


def relist(fg, arr):
    order = []; seen = set()
    for g in fg:
        if g not in seen:
            seen.add(g); order.append(g)
    idx = {g: np.where(fg == g)[0] for g in order}
    return [arr[idx[g]] for g in order]


def fold_names():
    fs = set()
    for f in glob.glob(f"{LOG}/74_*_value_only.npz"):
        fs.add(os.path.basename(f)[3:-len("_value_only.npz")])
    return sorted(fs)


def check_fold(fold, B=2000):
    data = {}
    for m in METHODS:
        f = f"{LOG}/74_{fold}_{m}.npz"
        if not os.path.exists(f):
            print(f"  [{fold}] MISSING {m}"); return None
        data[m] = dict(np.load(f))
    fg = data["value_only"]["tgt_fg"]
    for m in METHODS:
        assert np.array_equal(data[m]["tgt_fg"], fg), f"{fold} {m} fg mismatch"
    yl = relist(fg, data["value_only"]["tgt_fy"])
    pl = {m: relist(fg, data[m]["tgt_fp_ens"]) for m in METHODS}
    thr = {m: float(data[m]["thr_ens"]) for m in METHODS}
    out = dict(fold=fold, n_pat=len(yl))
    pt = {}
    for m in METHODS:
        u = bootstrap_util_ci(yl, pl[m], thr[m], B=B, seed=0)
        a = roc_auc_score(data[m]["tgt_fy"], data[m]["tgt_fp_ens"])
        pt[m] = dict(tgt_util=round(u["point"], 4), lo=round(u["lo"], 4), hi=round(u["hi"], 4),
                     tgt_auroc=round(float(a), 4), thr=round(thr[m], 4))
    vo, vm = pt["value_only"]["tgt_util"], pt["value_mask"]["tgt_util"]
    base_name = "value_mask" if vm >= vo else "value_only"
    base_max = max(vo, vm)
    out["point"] = pt; out["base_name"] = base_name; out["base_max"] = round(base_max, 4)
    for m in ("R4_ipw", "OPS_full"):
        d = paired_bootstrap_util_diff(yl, pl[base_name], pl[m], thr[base_name], thr[m], B=B, seed=0)
        out[f"{m}_gain_vs_max"] = round(d["point"], 4)
        out[f"{m}_paired_vs_max"] = [round(d["lo"], 4), round(d["hi"], 4)]
        out[f"{m}_p_gt0"] = round(d["p_gt0"], 3)
        out[f"{m}_repair_holds"] = bool(d["lo"] > 0)
    return out


if __name__ == "__main__":
    folds = [sys.argv[1]] if len(sys.argv) > 1 else fold_names()
    res = []
    for fold in folds:
        r = check_fold(fold)
        if r is None:
            continue
        res.append(r)
        print(f"\n=== {fold} (n_pat={r['n_pat']}) base={r['base_name']}({r['base_max']}) ===")
        for m in METHODS:
            p = r["point"][m]
            print(f"  {m:11s} util={p['tgt_util']:+.4f} CI[{p['lo']:+.4f},{p['hi']:+.4f}] auroc={p['tgt_auroc']:.4f} thr={p['thr']}")
        for m in ("R4_ipw", "OPS_full"):
            flag = " WIN(CI>0)" if r[f"{m}_repair_holds"] else ""
            print(f"  {m} gain_vs_max={r[f'{m}_gain_vs_max']:+.4f} paired_CI={r[f'{m}_paired_vs_max']} p>0={r[f'{m}_p_gt0']}{flag}")
    json.dump(res, open(f"{P}/results/stats/offline_ci_74_verify.json", "w"), indent=2, ensure_ascii=False)
    print(f"\nsaved -> results/stats/offline_ci_74_verify.json ({len(res)} folds)")
