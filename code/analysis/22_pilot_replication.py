#!/usr/bin/env python
"""Stage 2 :: 决定性排查 —— pilot 现象的规模/模型依赖

pilot 报 "value+mask 跨院 AUROC 0.706->0.590 崩溃(低于 value-only)"，用 2000/院 子样本 + logistic/HGBT。
全量 GRU/GBM 未复现此崩溃。此脚本系统扫描 (样本量 n_per_hosp) × (模型类) × (特征集)，
定位反噬现象是否是小样本/弱模型伪影。每个 (n, model) 用多个随机子样本重复以给出分布。

结论指标: cross_delta = cross_auroc(value_mask) - cross_auroc(value_only)。
  pilot 主张 cross_delta << 0 (mask 跨院反噬)。
"""
import os, sys, json, time
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

P = "."
sys.path.insert(0, f"{P}/code/utils")
import features as F

TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"


def flat(rows, X, M, D, Y, L, srcnorm, cap=200):
    Z = F.ffill_impute(np.asarray(X[rows][:, :cap]), np.asarray(M[rows][:, :cap]), srcnorm, channels=F.TV)
    val, msk, dlt, yy = [], [], [], []
    for k, i in enumerate(rows):
        n = int(min(L[i], cap))
        val.append(Z[k, :n]); msk.append(np.asarray(M[i, :n]).astype(np.float32))
        dlt.append(np.log1p(np.asarray(D[i, :n]))); yy.append(np.asarray(Y[i, :n]))
    return np.concatenate(val), np.concatenate(msk), np.concatenate(dlt), np.concatenate(yy).astype(int)


def make_model(kind, seed):
    # 偏离记录: 方案 提到 logistic/HGBT; 实测 sklearn HistGradientBoosting fit 78k 行需 528-834s
    # (单线程分箱), 而 LightGBM(同为梯度提升树, pilot 也用此族)全量 504k 行仅 2.1s。
    # 去掉 sklearn-HGB, 用 logistic + LightGBM 两族覆盖 pilot 的两类模型, 科学信息不减、快 400x。
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=1.0))
    if kind == "lgb":
        return lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                  min_child_samples=100, n_jobs=28, verbose=-1,
                                  scale_pos_weight=15, random_state=seed)


def main():
    t0 = time.time()
    X, M, D, Y, L = F.load_arrays()
    sp = F.load_splits()
    rng = np.random.default_rng(20260711)
    src, tgt = "A", "B"
    srcnorm = F.load_norm(src)
    src_all = np.array(sp[f"{src}_train"] + sp[f"{src}_val"])
    tgt_all = np.array(sp[f"{tgt}_test"])
    # 目标固定用一份(足够大)
    Ftt = flat(tgt_all, X, M, D, Y, L, srcnorm)

    def featset(name, v, m, d):
        return {"value_only": v, "value_mask": np.hstack([v, m, d])}[name]

    rows_out = []
    for n_per in [2000, 5000, 10000, 20336]:
        for kind in ["logistic", "lgb"]:
            nrep = 5 if n_per < 20336 else 3
            for rep in range(nrep):
                # 子样本源域
                if n_per >= len(src_all):
                    src_sub = src_all
                else:
                    src_sub = rng.choice(src_all, n_per, replace=False)
                # 源内留出 30% 作院内 test
                perm = rng.permutation(len(src_sub))
                tr = src_sub[perm[:int(0.7 * len(src_sub))]]
                ste = src_sub[perm[int(0.7 * len(src_sub)):]]
                Ftr = flat(tr, X, M, D, Y, L, srcnorm)
                Fst = flat(ste, X, M, D, Y, L, srcnorm)
                res = {}
                for feat in ["value_only", "value_mask"]:
                    Xtr = featset(feat, Ftr[0], Ftr[1], Ftr[2]); ytr = Ftr[3]
                    clf = make_model(kind, rep)
                    clf.fit(Xtr, ytr)
                    ain = roc_auc_score(Fst[3], clf.predict_proba(featset(feat, Fst[0], Fst[1], Fst[2]))[:, 1])
                    acr = roc_auc_score(Ftt[3], clf.predict_proba(featset(feat, Ftt[0], Ftt[1], Ftt[2]))[:, 1])
                    res[feat] = (ain, acr)
                rows_out.append(dict(n_per_hosp=n_per, model=kind, rep=rep,
                                     vo_in=res["value_only"][0], vo_cross=res["value_only"][1],
                                     vm_in=res["value_mask"][0], vm_cross=res["value_mask"][1],
                                     mask_in_delta=res["value_mask"][0] - res["value_only"][0],
                                     mask_cross_delta=res["value_mask"][1] - res["value_only"][1]))
            # 聚合该 (n,kind)
            sub = [r for r in rows_out if r["n_per_hosp"] == n_per and r["model"] == kind]
            cd = np.array([r["mask_cross_delta"] for r in sub])
            print(f"n={n_per:6d} {kind:9s} mask_cross_delta = {cd.mean():+.4f} +-{cd.std():.4f}  "
                  f"(vo_cross={np.mean([r['vo_cross'] for r in sub]):.3f} "
                  f"vm_cross={np.mean([r['vm_cross'] for r in sub]):.3f})  [{time.time()-t0:.0f}s]", flush=True)

    df = pl.DataFrame(rows_out)
    df.write_csv(f"{TAB}/stage2_pilot_scaling.csv")
    agg = (df.group_by(["n_per_hosp", "model"])
           .agg([pl.col("mask_in_delta").mean(), pl.col("mask_cross_delta").mean(),
                 pl.col("mask_cross_delta").std().alias("cross_delta_std"),
                 pl.col("vo_cross").mean(), pl.col("vm_cross").mean()])
           .sort(["model", "n_per_hosp"]))
    agg.write_csv(f"{TAB}/stage2_pilot_scaling_agg.csv")
    print("\n=== scaling aggregated ==="); print(agg.to_pandas().to_string(index=False))
    # 判定: 是否存在任何 (n,model) 使 mask 跨院明显反噬(<= -0.03)
    revert = agg.filter(pl.col("mask_cross_delta") <= -0.03)
    print("\n配置使 mask 跨院反噬<=-0.03:")
    print(revert.to_pandas().to_string(index=False) if revert.height else "  (无)")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
