#!/usr/bin/env python
"""Stage 2 :: 捷径账本 —— LightGBM 行级版 (方案 §7 要求多模型类)

pilot 用的是 logistic/HGBT on 2000-subsample, 非 GRU。此脚本用行级 LightGBM 在全量上复算,
定位"mask 跨院反噬"是否模型类/规模特异。特征集:
  value_only : ffill 归一化值(34)
  value_mask : 值(34) + mask(34) + log1p(delta)(34)
  mask_only  : mask(34) + log1p(delta)(34)
  count8     : 8h 回看观测计数(34)  —— 冠军方案的那个特征, 单独测
双向 A2B/B2A, 多种子(bootstrap 划分), 病人级评估。阈值在源域 val 选。
"""
import os, sys, json, time, argparse
import numpy as np, polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

P = "."
sys.path.insert(0, f"{P}/code/utils")
import features as F
from models import normalized_utility, best_threshold
from boot import patient_bootstrap_ci

TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"


def flatten_feats(rows, X, M, D, Y, L, srcnorm, cap):
    Z = F.ffill_impute(np.asarray(X[rows][:, :cap]), np.asarray(M[rows][:, :cap]), srcnorm, channels=F.TV)
    val, msk, dlt, cnt, yy, gg = [], [], [], [], [], []
    for k, i in enumerate(rows):
        n = int(min(L[i], cap))
        m = np.asarray(M[i, :n]).astype(np.float32)
        val.append(Z[k, :n]); msk.append(m); dlt.append(np.log1p(np.asarray(D[i, :n])))
        # 8h 回看观测计数 (因果, running)
        c = np.zeros((n, 34), dtype=np.float32)
        for t in range(n):
            lo = max(0, t - 7)
            c[t] = m[lo:t + 1].sum(axis=0)
        cnt.append(c)
        yy.append(np.asarray(Y[i, :n])); gg.append(np.full(n, i))
    return (np.concatenate(val), np.concatenate(msk), np.concatenate(dlt),
            np.concatenate(cnt), np.concatenate(yy).astype(int), np.concatenate(gg))


def featset(name, val, msk, dlt, cnt):
    if name == "value_only": return val
    if name == "value_mask": return np.hstack([val, msk, dlt])
    if name == "mask_only":  return np.hstack([msk, dlt])
    if name == "delta_only": return dlt          # 距上次观测时长(log1p), 补全 方案 §7.1
    if name == "count8":     return cnt
    if name == "value_count": return np.hstack([val, cnt])


def main(args):
    t0 = time.time()
    X, M, D, Y, L = F.load_arrays()
    sp = F.load_splits()
    rng = np.random.default_rng(20260711)
    cap = args.tcap
    feats = ["value_only", "value_mask", "mask_only", "delta_only", "count8", "value_count"]
    rows_out = []

    for direction in ["A2B", "B2A"]:
        src, tgt = ("A", "B") if direction == "A2B" else ("B", "A")
        srcnorm = F.load_norm(src)
        tr = np.array(sp[f"{src}_train"]); va = np.array(sp[f"{src}_val"])
        ste = np.array(sp[f"{src}_test"]); tte = np.array(sp[f"{tgt}_test"])
        # 抽样训练集控制规模
        if len(tr) > args.nfit:
            tr = rng.choice(tr, args.nfit, replace=False)
        print(f"[{direction}] flattening...", flush=True)
        Ftr = flatten_feats(tr, X, M, D, Y, L, srcnorm, cap)
        Fva = flatten_feats(va, X, M, D, Y, L, srcnorm, cap)
        Fst = flatten_feats(ste, X, M, D, Y, L, srcnorm, cap)
        Ftt = flatten_feats(tte, X, M, D, Y, L, srcnorm, cap)
        print(f"[{direction}] flattened ({time.time()-t0:.0f}s), train rows={len(Ftr[4])}", flush=True)

        for feat in feats:
            Xtr = featset(feat, Ftr[0], Ftr[1], Ftr[2], Ftr[3]); ytr = Ftr[4]
            for seed in range(args.seeds):
                clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                         min_child_samples=100, subsample=0.8, subsample_freq=1,
                                         colsample_bytree=0.8, n_jobs=16, verbose=-1,
                                         scale_pos_weight=15, random_state=seed)
                clf.fit(Xtr, ytr)

                def ev(Fx):
                    Xe = featset(feat, Fx[0], Fx[1], Fx[2], Fx[3])
                    p = clf.predict_proba(Xe)[:, 1]; g = Fx[5]; y = Fx[4]
                    probs, ys = [], []
                    for gg in np.unique(g):
                        s = g == gg; probs.append(p[s]); ys.append(y[s])
                    return roc_auc_score(y, p), probs, ys

                a_va, pv, yv = ev(Fva)
                thr, _ = best_threshold(yv, pv)
                a_st, pst, yst = ev(Fst)
                a_tt, ptt, ytt = ev(Ftt)
                rows_out.append(dict(direction=direction, feat=feat, seed=seed, thr=thr,
                                     src_auroc=a_st, tgt_auroc=a_tt,
                                     src_util=normalized_utility(yst, pst, thr),
                                     tgt_util=normalized_utility(ytt, ptt, thr)))
                print(f"  {direction} {feat:12s} s{seed} in={a_st:.3f} cross={a_tt:.3f} "
                      f"in_u={rows_out[-1]['src_util']:.3f} cross_u={rows_out[-1]['tgt_util']:.3f}", flush=True)

    df = pl.DataFrame(rows_out)
    df.write_csv(f"{TAB}/stage2_ledger_gbm.csv")
    agg = (df.group_by(["direction", "feat"])
           .agg([pl.col(c).mean().alias(c) for c in ["src_auroc", "tgt_auroc", "src_util", "tgt_util"]])
           .sort(["direction", "feat"]))
    agg.write_csv(f"{TAB}/stage2_ledger_gbm_agg.csv")
    print("\n=== GBM aggregated ==="); print(agg.to_pandas().to_string(index=False))

    # mask 净效应
    for direction in ["A2B", "B2A"]:
        dd = {r["feat"]: r for r in agg.filter(pl.col("direction") == direction).to_dicts()}
        vm, vo = dd["value_mask"], dd["value_only"]
        print(f"\n[{direction}] mask AUROC: in {vm['src_auroc']-vo['src_auroc']:+.4f}  "
              f"cross {vm['tgt_auroc']-vo['tgt_auroc']:+.4f}  | "
              f"Utility: in {vm['src_util']-vo['src_util']:+.4f}  cross {vm['tgt_util']-vo['tgt_util']:+.4f}")
        cnt = dd["count8"]
        print(f"  count8-only: in {cnt['src_auroc']:.3f} cross {cnt['tgt_auroc']:.3f} "
              f"(= value-only cross {vo['tgt_auroc']:.3f} 的 {cnt['tgt_auroc']/vo['tgt_auroc']*100:.0f}%)")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--nfit", type=int, default=13015)
    ap.add_argument("--tcap", type=int, default=200)
    main(ap.parse_args())
