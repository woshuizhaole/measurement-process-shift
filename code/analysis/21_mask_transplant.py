#!/usr/bin/env python
"""Stage 2 :: 掩码移植反事实归因 (方案 §Stage2.2) + 三通道 Shapley (§Stage2.3)

思路: 把源域病人放进目标域的下单习惯里(值/标签不变, 只稀疏化 mask)。掉的性能归因于观测过程。
  - 只删不加(无法凭空造未测值): 逐(变量,行)以 p_keep = min(1, e_tgt(C)/e_src(C)) 保留观测。
  - 对照 A->A': 用同等平均稀疏程度、但源域自身的模式(每变量恒定 keep_frac)稀疏化,
    分离 "单纯变稀疏" 与 "稀疏化的模式"。
  - 评估器: 在源域训练的 value+mask 行级 GBM(冻结)。评估反事实队列上的 AUROC 与官方 Utility。
  - 可行性覆盖率: e_tgt<=e_src 的格点占比(只能删的比例), 必须报告。

倾向 e_h(C): 逐变量逻辑回归 M_j ~ phi(C), phi 只含永远可得通道(build_phi)。cross-fit 于各自域。
"""
import os, sys, json, time, argparse
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

P = "."
sys.path.insert(0, f"{P}/code/utils")
import features as F
from models import normalized_utility, best_threshold

TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"


def flatten(rows, X, M, Y, L, cap, srcnorm, meta=None):
    """展平为行级: 归一化值(34), mask(34), y, pid, 以及每行的 phi。"""
    Z = F.ffill_impute(np.asarray(X[rows][:, :cap]), np.asarray(M[rows][:, :cap]), srcnorm, channels=F.TV)
    mrows = meta[list(rows)] if meta is not None else None
    phi, _ = F.build_phi(X[rows], M[rows], L[rows], mrows, src=None, _norm=srcnorm)
    vv, mm, yy, gg, pp = [], [], [], [], []
    for k, i in enumerate(rows):
        n = int(min(L[i], cap))
        vv.append(Z[k, :n]); mm.append(M[i, :n].astype(np.float32))
        yy.append(Y[i, :n]); gg.append(np.full(n, i)); pp.append(phi[k, :n])
    return (np.concatenate(vv), np.concatenate(mm), np.concatenate(yy).astype(int),
            np.concatenate(gg), np.concatenate(pp))


def fit_propensity(PH, MM):
    """逐变量 e_j(C)。返回 list of (kind, model_or_const)。"""
    models = []
    for j in range(34):
        col = MM[:, j].astype(int)
        r = col.mean()
        if r < 0.002 or r > 0.998:
            models.append(("const", float(np.clip(r, 0.005, 0.995))))
        else:
            lr = LogisticRegression(max_iter=500, C=1.0).fit(PH, col)
            models.append(("lr", lr))
    return models


def predict_e(models, PH):
    E = np.zeros((PH.shape[0], 34), dtype=np.float32)
    for j, (kind, mod) in enumerate(models):
        E[:, j] = mod if kind == "const" else mod.predict_proba(PH)[:, 1]
    return np.clip(E, 0.005, 0.995)


def main(args):
    t0 = time.time()
    X, M, D, Y, L = F.load_arrays()
    meta = F.load_meta()
    sp = F.load_splits()
    rng = np.random.default_rng(20260711)
    cap = args.tcap
    results = []

    for direction in ["A2B", "B2A"]:
        src, tgt = ("A", "B") if direction == "A2B" else ("B", "A")
        srcnorm = F.load_norm(src)
        src_tr = rng.choice(sp[f"{src}_train"], min(args.nfit, len(sp[f"{src}_train"])), replace=False)
        src_te = np.array(sp[f"{src}_test"])
        tgt_te = np.array(sp[f"{tgt}_test"])

        # 训练评估器 (value+mask GBM, 冻结)
        Vtr, Mtr, ytr, gtr, PHtr = flatten(np.array(src_tr), X, M, Y, L, cap, srcnorm, meta)
        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                 min_child_samples=100, n_jobs=16, verbose=-1, scale_pos_weight=15)
        clf.fit(np.hstack([Vtr, Mtr]), ytr)

        # 倾向: e_src 于源训练; e_tgt 于目标测试
        e_src_models = fit_propensity(PHtr, Mtr)
        Vt, Mt, yt, gt, PHt = flatten(tgt_te, X, M, Y, L, cap, F.load_norm(tgt), meta)
        e_tgt_models = fit_propensity(PHt, Mt)

        # 源测试组的行级特征 + phi + e_src/e_tgt(在源测试的 C 上预测)
        Vs, Ms, ys, gs, PHs = flatten(src_te, X, M, Y, L, cap, srcnorm, meta)
        e_src_on_s = predict_e(e_src_models, PHs)
        e_tgt_on_s = predict_e(e_tgt_models, PHs)
        pkeep = np.clip(e_tgt_on_s / e_src_on_s, 0, 1)          # 只删: e_tgt<e_src 处 pkeep<1
        coverage = float((e_tgt_on_s <= e_src_on_s).mean())     # 可稀疏化(只删)覆盖率

        def util_of(Vx, Mx, yy, gg):
            p = clf.predict_proba(np.hstack([Vx, Mx]))[:, 1]
            # 逐病人序列
            probs, ys_l = [], []
            for g in np.unique(gg):
                sel = gg == g; probs.append(p[sel]); ys_l.append(yy[sel])
            thr, _ = best_threshold(ys_l, probs)   # 阈值在该队列上选(报告用, 非严格源域)
            return roc_auc_score(yy, p), normalized_utility(ys_l, probs, thr)

        def transplant(mode):
            Vx = Vs.copy(); Mx = Ms.copy()
            if mode == "to_tgt":
                drop = (Ms > 0) & (rng.random(Ms.shape) > pkeep)
            elif mode == "to_src":       # 对照: 每变量恒定 keep_frac = mean(pkeep_j), 打乱模式
                kf = pkeep.mean(axis=0, keepdims=True)
                drop = (Ms > 0) & (rng.random(Ms.shape) > kf)
            else:
                drop = np.zeros_like(Ms, dtype=bool)
            Mx[drop] = 0; Vx[drop] = 0.0    # 移除观测: mask=0, 归一化值->0(=源域均值)
            return Vx, Mx

        auc_src, u_src = util_of(Vs, Ms, ys, gs)
        auc_tgt, u_tgt = util_of(Vt, Mt, yt, gt)
        V1, M1 = transplant("to_tgt"); auc_tr, u_tr = util_of(V1, M1, ys, gs)
        V2, M2 = transplant("to_src"); auc_ct, u_ct = util_of(V2, M2, ys, gs)

        rec = dict(direction=direction, coverage_sparsifiable=coverage,
                   auc_src=float(auc_src), auc_tgt=float(auc_tgt),
                   auc_transplant=float(auc_tr), auc_ctrl_selfsparsify=float(auc_ct),
                   util_src=float(u_src), util_tgt=float(u_tgt),
                   util_transplant=float(u_tr), util_ctrl=float(u_ct),
                   drop_total_auc=float(auc_src - auc_tgt),
                   drop_transplant_auc=float(auc_src - auc_tr),
                   pattern_attributable_auc=float((auc_src - auc_tr) - (auc_src - auc_ct)),
                   frac_of_total_gap_explained=float((auc_src - auc_tr) / (auc_src - auc_tgt + 1e-9)))
        results.append(rec)
        print(f"[{direction}] cov={coverage:.2f} src={auc_src:.3f} tgt={auc_tgt:.3f} "
              f"transplant={auc_tr:.3f} ctrl={auc_ct:.3f} "
              f"pattern_attr={rec['pattern_attributable_auc']:.3f} "
              f"gap_explained={rec['frac_of_total_gap_explained']:.2f} ({time.time()-t0:.0f}s)", flush=True)

    pl.DataFrame(results).write_csv(f"{TAB}/stage2_mask_transplant.csv")
    print("\n=== mask transplant ===")
    print(pl.DataFrame(results).to_pandas().to_string(index=False))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfit", type=int, default=10000)
    ap.add_argument("--tcap", type=int, default=200)
    main(ap.parse_args())
