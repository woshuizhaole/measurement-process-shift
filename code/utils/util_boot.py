#!/usr/bin/env python
"""code/utils/util_boot.py — 病人级 cluster bootstrap for 官方归一化 Utility (方案 §8.3)。

boot.py 只支持 auroc/auprc；本模块补齐官方归一化 Utility 的病人级 bootstrap，
用于主要终点(§9.1)的"方法−基线"配对 Utility 差值 CI。

正确性+速度关键：固定阈值下，每病人的 (u_obs, u_best, u_inact) 三分量可加，
  normalized_utility = (Σu_obs − Σu_inact) / (Σu_best − Σu_inact)。
因此先对每病人预算三分量一次，bootstrap 只做 O(n) 求和 —— 与官方 normalized_utility
逐位等价（见 __main__ 对拍），且比每次重算快数百倍。
u_best/u_inact 只依赖标签 y，与预测/阈值无关 → 配对时 A/B 共享，只有 u_obs 不同。
"""
import numpy as np
import sys
sys.path.insert(0, "./code/utils")
from evaluate_sepsis_score import compute_prediction_utility

# 官方常数 (与 models.normalized_utility 完全一致)
DT_EARLY, DT_OPTIMAL, DT_LATE = -12, -6, 3.0
MAX_U_TP, MIN_U_FN, U_FP, U_TN = 1, -2, -0.05, 0


def _cpu(y, pred):
    return compute_prediction_utility(y, pred, DT_EARLY, DT_OPTIMAL, DT_LATE,
                                      MAX_U_TP, MIN_U_FN, U_FP, U_TN, check_errors=False)


def per_patient_components(y_list, p_list, thr):
    """返回每病人的 (u_obs, u_best, u_inact) 三个 1D array。thr 固定。"""
    n = len(y_list)
    uo = np.empty(n); ub = np.empty(n); ui = np.empty(n)
    for k in range(n):
        y = np.asarray(y_list[k]).astype(int)
        pred = (np.asarray(p_list[k]) >= thr).astype(int)
        m = len(y)
        best = np.zeros(m); inact = np.zeros(m)
        if np.any(y):
            t_sepsis = int(np.argmax(y)) - DT_OPTIMAL
            best[max(0, t_sepsis + DT_EARLY): min(t_sepsis + int(DT_LATE) + 1, m)] = 1
        uo[k] = _cpu(y, pred); ub[k] = _cpu(y, best); ui[k] = _cpu(y, inact)
    return uo, ub, ui


def _norm(uo, ub, ui):
    denom = ub.sum() - ui.sum()
    if denom == 0:
        return 0.0
    return float((uo.sum() - ui.sum()) / denom)


def bootstrap_util_ci(y_list, p_list, thr, B=2000, seed=0, alpha=0.05):
    """单方法归一化 Utility 的病人级 bootstrap 点估计 + (1-alpha) CI。"""
    uo, ub, ui = per_patient_components(y_list, p_list, thr)
    n = len(uo)
    point = _norm(uo, ub, ui)
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        vals[b] = _norm(uo[idx], ub[idx], ui[idx])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(point=point, lo=float(lo), hi=float(hi), mean=float(vals.mean()),
                std=float(vals.std()), n_boot=int(B), n_patients=int(n))


def paired_bootstrap_util_diff(y_list, pA_list, pB_list, thrA, thrB, B=2000, seed=0, alpha=0.05):
    """配对: Util(B) − Util(A)，同一批重采样病人。u_best/u_inact 由 y 决定，A/B 共享。"""
    uoA, ub, ui = per_patient_components(y_list, pA_list, thrA)
    uoB, ub2, ui2 = per_patient_components(y_list, pB_list, thrB)
    # ub/ui 只依赖 y，两者必须相等（防御性断言）
    assert np.allclose(ub, ub2) and np.allclose(ui, ui2), "u_best/u_inact 与预测无关，应相等"
    n = len(ub)
    point = _norm(uoB, ub, ui) - _norm(uoA, ub, ui)
    rng = np.random.default_rng(seed)
    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        diffs[b] = _norm(uoB[idx], ub[idx], ui[idx]) - _norm(uoA[idx], ub[idx], ui[idx])
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(point=float(point), lo=float(lo), hi=float(hi), p_gt0=float((diffs > 0).mean()),
                mean=float(diffs.mean()), std=float(diffs.std()), n_boot=int(B), n_patients=int(n))


if __name__ == "__main__":
    import time
    sys.path.insert(0, "./code/utils")
    from models import normalized_utility          # 仅用于对拍
    rng = np.random.default_rng(0)
    yL, pAL, pBL = [], [], []
    for i in range(4000):                           # 模拟目标测试集规模
        m = int(rng.integers(20, 60))
        y = np.zeros(m, int)
        if rng.random() < 0.08:
            t = int(rng.integers(5, m)); y[t:] = 1
        yL.append(y); pAL.append(rng.random(m) * 0.5)
        pBL.append(np.clip(y * 0.4 + rng.random(m) * 0.4, 0, 1))
    thr = 0.3
    # 对拍: 我的点估计 == 官方 normalized_utility
    uo, ub, ui = per_patient_components(yL, pAL, thr)
    mine = _norm(uo, ub, ui); off = normalized_utility(yL, pAL, thr)
    print(f"point mine={mine:.10f} official={off:.10f} diff={abs(mine-off):.2e}")
    assert abs(mine - off) < 1e-9, "对拍失败!"
    t0 = time.time()
    ci = bootstrap_util_ci(yL, pAL, thr, B=2000)
    print("CI A:", {k: round(v, 4) for k, v in ci.items() if k in ('point', 'lo', 'hi')}, f"({time.time()-t0:.2f}s)")
    t0 = time.time()
    pd = paired_bootstrap_util_diff(yL, pAL, pBL, thr, thr, B=2000)
    print("paired B-A:", {k: round(v, 4) for k, v in pd.items() if k in ('point', 'lo', 'hi', 'p_gt0')}, f"({time.time()-t0:.2f}s)")
    print("OK: util_boot 对拍官方通过, 速度可用")
