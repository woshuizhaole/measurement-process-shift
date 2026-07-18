#!/usr/bin/env python
"""code/utils/boot.py — 病人级 cluster bootstrap + 指标 (Stage 1+ 全程使用)

所有 CI 必须按病人重采样（不是按行），因为小时行在病人内相关。
方案 §8.3。
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def patient_bootstrap_ci(y, p, groups, metric="auroc", B=2000, seed=0, alpha=0.05):
    """按病人重采样计算指标的点估计与 (1-alpha) CI。
    y,p,groups: 长度 = 行数（每行一个小时）。groups = patient_id。"""
    y = np.asarray(y); p = np.asarray(p); groups = np.asarray(groups)
    fn = {"auroc": roc_auc_score, "auprc": average_precision_score}[metric]
    point = fn(y, p)
    uniq = np.unique(groups)
    # 预建 patient -> row index
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_g[g] for g in samp])
        ys, ps = y[rows], p[rows]
        if ys.min() == ys.max():
            continue
        vals.append(fn(ys, ps))
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(point=float(point), lo=float(lo), hi=float(hi), n_boot=len(vals),
                mean=float(vals.mean()), std=float(vals.std()))


def _codes_and_offsets(groups):
    """把 groups 映射成连续 code, 并给出每个 code 的行索引数组(list)。一次性预建。"""
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]
    boundaries = np.searchsorted(sorted_inv, np.arange(len(uniq) + 1))
    rows_by_code = [order[boundaries[c]:boundaries[c + 1]] for c in range(len(uniq))]
    return uniq, rows_by_code


def paired_bootstrap_diff(y, pA, pB, groups, metric="auroc", B=2000, seed=0, alpha=0.05,
                          max_patients_per_boot=None):
    """配对 bootstrap: 指标(pB) - 指标(pA)，同一批重采样病人。已向量化预建索引以提速。"""
    y = np.asarray(y); pA = np.asarray(pA); pB = np.asarray(pB); groups = np.asarray(groups)
    fn = {"auroc": roc_auc_score, "auprc": average_precision_score}[metric]
    point = fn(y, pB) - fn(y, pA)
    uniq, rows_by_code = _codes_and_offsets(groups)
    nG = len(uniq)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        samp = rng.integers(0, nG, size=nG)                     # 采样 code, 免哈希
        rows = np.concatenate([rows_by_code[c] for c in samp])
        ys = y[rows]
        if ys.min() == ys.max():
            continue
        diffs.append(fn(ys, pB[rows]) - fn(ys, pA[rows]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(point=float(point), lo=float(lo), hi=float(hi),
                p_gt0=float((diffs > 0).mean()), n_boot=len(diffs))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(200), 10)
    y = (rng.random(2000) < 0.1).astype(int)
    p = y * 0.3 + rng.random(2000) * 0.7
    print("auroc CI:", patient_bootstrap_ci(y, p, g, B=500))
