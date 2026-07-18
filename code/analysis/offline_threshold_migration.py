#!/usr/bin/env python
"""offline_threshold_migration.py — B2A R4_ipw vs value_mask 的"阈值/校准迁移"分解 (护栏产物)。

诊断: B2A 上 R4_ipw 相对 value_mask 的 +0.109 Utility 增益中, 有多少来自"目标域上换了更优阈值"
(阈值/校准脆弱性), 而非真实的排序/预测改进。做法: 对 value_mask 与 R4_ipw 各自
  (a) 用源域选出的阈值 thr_ens (方案 铁律: 阈值只能在源域选) 在目标算 Utility;
  (b) 用目标域 oracle-optimal 阈值 (在官方网格上扫 argmax 目标 Utility; 实践不可得, 仅上界) 算 Utility。
gain_srcthr = R4(a)-vm(a) (应≈+0.109), gain_oraclethr = R4(b)-vm(b) (应≈+0.06),
migration_frac = 1 - gain_oraclethr/gain_srcthr = 源选阈增益中归因于阈值/校准迁移的比例。
输入: results/logits/40d_B2A_{value_mask,R4_ipw}.npz (40d --save-logits 落盘)。
输出: results/stats/b2a_threshold_migration.json。点估计, 用 util_boot 官方 Utility 分量, 与 gates.json 逐位对得上。
"""
import sys, json, numpy as np
P = "."
sys.path.insert(0, f"{P}/code/utils")
from util_boot import per_patient_components, _norm

LOG = f"{P}/results/logits"
GRID = np.linspace(0.01, 0.6, 60)  # 与 models.best_threshold 官方网格一致


def relist(fg, arr):
    order = []; seen = set()
    for g in fg:
        if g not in seen:
            seen.add(g); order.append(g)
    idx = {g: np.where(fg == g)[0] for g in order}
    return [arr[idx[g]] for g in order]


def util_at(yl, pl, thr):
    uo, ub, ui = per_patient_components(yl, pl, thr)
    return _norm(uo, ub, ui)


def oracle(yl, pl):
    best_u, best_t = -9.0, None
    for t in GRID:
        u = util_at(yl, pl, float(t))
        if u > best_u:
            best_u, best_t = u, float(t)
    return best_u, best_t


def main():
    vm = dict(np.load(f"{LOG}/40d_B2A_value_mask.npz"))
    r4 = dict(np.load(f"{LOG}/40d_B2A_R4_ipw.npz"))
    assert np.array_equal(vm["tgt_fg"], r4["tgt_fg"]), "fg mismatch"
    fg = vm["tgt_fg"]
    yl = relist(fg, vm["tgt_fy"])
    vm_p = relist(fg, vm["tgt_fp_ens"]); r4_p = relist(fg, r4["tgt_fp_ens"])
    vm_srcthr = float(vm["thr_ens"]); r4_srcthr = float(r4["thr_ens"])

    vm_src = util_at(yl, vm_p, vm_srcthr); r4_src = util_at(yl, r4_p, r4_srcthr)
    vm_orc, vm_ot = oracle(yl, vm_p); r4_orc, r4_ot = oracle(yl, r4_p)
    gain_src = r4_src - vm_src; gain_orc = r4_orc - vm_orc
    mig = 1.0 - gain_orc / gain_src

    out = {
        "direction": "B2A (eICU->MIMIC target)",
        "method_pair": "R4_ipw vs value_mask",
        "n_patients": int(len(yl)),
        "grid": "np.linspace(0.01,0.6,60) (== models.best_threshold official grid)",
        "value_mask_srcthr": round(vm_srcthr, 4), "R4_srcthr": round(r4_srcthr, 4),
        "value_mask_oraclethr": round(vm_ot, 4), "R4_oraclethr": round(r4_ot, 4),
        "value_mask_util_srcthr": round(vm_src, 4), "R4_util_srcthr": round(r4_src, 4),
        "value_mask_util_oraclethr": round(vm_orc, 4), "R4_util_oraclethr": round(r4_orc, 4),
        "gain_srcthr": round(gain_src, 4), "gain_oraclethr": round(gain_orc, 4),
        "migration_frac": round(mig, 4),
        "note": ("B2A R4_ipw-vs-value_mask Utility gain, point estimates from 40d ensemble patient-level logits. "
                 "gain_srcthr uses each method's source-selected threshold thr_ens (方案: threshold only chosen on source). "
                 "gain_oraclethr uses each method's target-oracle-optimal threshold (argmax target Utility over the official grid; "
                 "not achievable in practice, upper bound). migration_frac = 1 - gain_oraclethr/gain_srcthr = fraction of the "
                 "source-threshold gain attributable to threshold/calibration migration rather than genuine ranking improvement."),
    }
    json.dump(out, open(f"{P}/results/stats/b2a_threshold_migration.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    # sanity vs gates.json
    assert abs(vm_src - 0.1268) < 1e-3 and abs(r4_src - 0.236) < 1e-3 and abs(gain_src - 0.1092) < 1e-3, "sanity vs gates.json 失败"
    print("SANITY OK: vm_util_srcthr~0.1268, R4_util_srcthr~0.236, gain_srcthr~0.1092")


if __name__ == "__main__":
    main()
