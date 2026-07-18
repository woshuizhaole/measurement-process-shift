#!/usr/bin/env python
"""D2 诊断: 把 pilot 反噬排查从 AUROC 空间搬到 Utility + 校准空间。

原 22_pilot_replication.py 只输出 cross AUROC 的 mask_cross_delta, 据此判"反噬=小样本伪影"
(因为随 n 增大 delta 单调趋 0)。但方案立身之本: 崩溃只在官方 Utility 现身, AUROC 会掩盖。
本脚本复用 22 的 (n × model × featset) 网格, 但对每个配置额外输出:
  - cross Utility (value_only / value_mask), util_cross_delta = U(vm)-U(vo)
  - cross 校准斜率 (value_only / value_mask)
  - 同时保留 AUROC 以对拍
阈值: 在源域 in-domain 留出集上按 Utility 最大选 (方案: 阈值只能源域选)。
判定: util_cross_delta 是否在大 n 下仍 <0 (AUROC 下已消失), 即反噬是否活在 Utility 空间。
"""
import os, sys, json, time
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import statsmodels.api as sm

P = "."
sys.path.insert(0, f"{P}/code/utils")
import features as F
from models import normalized_utility, best_threshold

TAB = f"{P}/results/tables"


def calib_slope(y, p):
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    try:
        r = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
        return float(r.params[1])
    except Exception:
        return np.nan


def flat(rows, X, M, D, Y, L, srcnorm, cap=200):
    """返回 value/mask/delta/y 以及 per-patient group id 列表 (供 utility 逐病人)。"""
    Z = F.ffill_impute(np.asarray(X[rows][:, :cap]), np.asarray(M[rows][:, :cap]), srcnorm, channels=F.TV)
    val, msk, dlt, yy, gg = [], [], [], [], []
    for k, i in enumerate(rows):
        n = int(min(L[i], cap))
        val.append(Z[k, :n]); msk.append(np.asarray(M[i, :n]).astype(np.float32))
        dlt.append(np.log1p(np.asarray(D[i, :n]))); yy.append(np.asarray(Y[i, :n]))
        gg.append(np.full(n, i))
    return (np.concatenate(val), np.concatenate(msk), np.concatenate(dlt),
            np.concatenate(yy).astype(int), np.concatenate(gg))


def per_patient(y, p, g):
    ys, ps = [], []
    for gid in np.unique(g):
        s = g == gid
        ys.append(y[s]); ps.append(p[s])
    return ys, ps


def make_model(kind, seed):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=1.0))
    if kind == "lgb":
        return lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                  min_child_samples=100, n_jobs=28, verbose=-1,
                                  scale_pos_weight=15, random_state=seed)


def featset(name, v, m, d):
    return {"value_only": v, "value_mask": np.hstack([v, m, d])}[name]


def main():
    t0 = time.time()
    X, M, D, Y, L = F.load_arrays()
    sp = F.load_splits()
    rng = np.random.default_rng(20260711)
    src, tgt = "A", "B"
    srcnorm = F.load_norm(src)
    src_all = np.array(sp[f"{src}_train"] + sp[f"{src}_val"])
    tgt_all = np.array(sp[f"{tgt}_test"])
    # 目标固定 (per-patient 保留)
    Vtt, Mtt, Dtt, ytt, gtt = flat(tgt_all, X, M, D, Y, L, srcnorm)

    rows_out = []
    for n_per in [2000, 5000, 10000, 20336]:
        for kind in ["logistic"]:
            nrep = 5 if n_per < 20336 else 3
            for rep in range(nrep):
                if n_per >= len(src_all):
                    src_sub = src_all
                else:
                    src_sub = rng.choice(src_all, n_per, replace=False)
                perm = rng.permutation(len(src_sub))
                tr = src_sub[perm[:int(0.7 * len(src_sub))]]
                ste = src_sub[perm[int(0.7 * len(src_sub)):]]
                Vtr, Mtr, Dtr, ytr, _ = flat(tr, X, M, D, Y, L, srcnorm)
                Vst, Mst, Dst, yst, gst = flat(ste, X, M, D, Y, L, srcnorm)
                rec = dict(n_per_hosp=n_per, model=kind, rep=rep)
                for feat in ["value_only", "value_mask"]:
                    clf = make_model(kind, rep)
                    clf.fit(featset(feat, Vtr, Mtr, Dtr), ytr)
                    # in-domain (源留出) 预测 -> 选阈值 (utility 最大)
                    p_in = clf.predict_proba(featset(feat, Vst, Mst, Dst))[:, 1]
                    ys_in, ps_in = per_patient(yst, p_in, gst)
                    thr, u_in = best_threshold(ys_in, ps_in)
                    # cross (目标) 预测
                    p_cr = clf.predict_proba(featset(feat, Vtt, Mtt, Dtt))[:, 1]
                    ys_cr, ps_cr = per_patient(ytt, p_cr, gtt)
                    u_cr = normalized_utility(ys_cr, ps_cr, thr)
                    auc_in = roc_auc_score(yst, p_in)
                    auc_cr = roc_auc_score(ytt, p_cr)
                    cal_cr = calib_slope(ytt, p_cr)
                    tag = "vo" if feat == "value_only" else "vm"
                    rec.update({f"{tag}_thr": thr, f"{tag}_util_in": u_in, f"{tag}_util_cross": u_cr,
                                f"{tag}_auc_in": auc_in, f"{tag}_auc_cross": auc_cr,
                                f"{tag}_calib_cross": cal_cr})
                rec["util_cross_delta"] = rec["vm_util_cross"] - rec["vo_util_cross"]
                rec["auc_cross_delta"] = rec["vm_auc_cross"] - rec["vo_auc_cross"]
                rec["calib_cross_delta"] = rec["vm_calib_cross"] - rec["vo_calib_cross"]
                rows_out.append(rec)
            sub = [r for r in rows_out if r["n_per_hosp"] == n_per and r["model"] == kind]
            ud = np.array([r["util_cross_delta"] for r in sub])
            ad = np.array([r["auc_cross_delta"] for r in sub])
            print(f"n={n_per:6d} {kind:9s} util_delta={ud.mean():+.4f}+-{ud.std():.4f}  "
                  f"auc_delta={ad.mean():+.4f}  "
                  f"(vo_uc={np.mean([r['vo_util_cross'] for r in sub]):+.3f} "
                  f"vm_uc={np.mean([r['vm_util_cross'] for r in sub]):+.3f})  [{time.time()-t0:.0f}s]", flush=True)

    df = pl.DataFrame(rows_out)
    df.write_csv(f"{TAB}/d2_pilot_utility_logit.csv")
    agg = (df.group_by(["n_per_hosp", "model"])
           .agg([pl.col("util_cross_delta").mean(), pl.col("util_cross_delta").std().alias("util_delta_std"),
                 pl.col("auc_cross_delta").mean(), pl.col("calib_cross_delta").mean(),
                 pl.col("vo_util_cross").mean(), pl.col("vm_util_cross").mean(),
                 pl.col("vo_calib_cross").mean(), pl.col("vm_calib_cross").mean()])
           .sort(["model", "n_per_hosp"]))
    agg.write_csv(f"{TAB}/d2_pilot_utility_logit_agg.csv")
    print("\n=== D2 utility-space aggregated ===")
    print(agg.to_pandas().to_string(index=False))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
