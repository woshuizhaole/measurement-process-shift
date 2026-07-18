#!/usr/bin/env python
"""Stage 7 :: CinC2019 六域方差分解 (方案 §Stage7.3)

6 域 = {A,B} x {MICU,SICU,unknown}。把"测量过程指纹"的方差分解为:
  院间(A vs B) 与 院内单元间(MICU/SICU/unknown within hospital)。
方法: 病人级每变量观测率 -> 混合效应式方差分解(院 + 单元 + 残差); 以及六域两两 mask->域 AUC。
回答: 测量指纹主要是院间差异, 还是院内单元也贡献?
"""
import os, sys, json, time
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

P = "."
sys.path.insert(0, f"{P}/code/utils")
import features as F
TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"
SEED = 20260711


def main():
    t0 = time.time()
    X, M, D, Y, L = F.load_arrays(); meta = F.load_meta()
    N, T, _ = X.shape
    valid = np.arange(T)[None, :] < L[:, None]
    obs = (M.sum(1) / np.maximum(valid.sum(1, keepdims=True), 1)).astype(np.float32)  # (N,34) 病人级观测率
    hosp = meta["hospital"].to_numpy(); unit = meta["unit"].to_numpy(); dom = meta["domain"].to_numpy()

    # 六域两两 mask->域 AUC
    doms = sorted(set(dom))
    pair_auc = {}
    rng = np.random.default_rng(SEED)
    for i in range(len(doms)):
        for j in range(i + 1, len(doms)):
            m = np.isin(dom, [doms[i], doms[j]])
            y = (dom[m] == doms[j]).astype(int)
            Xd = obs[m]; g = np.arange(m.sum())
            gss = GroupShuffleSplit(1, test_size=0.4, random_state=SEED)
            tr, te = next(gss.split(Xd, y, g))
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            clf.fit(Xd[tr], y[tr])
            pair_auc[f"{doms[i]}_vs_{doms[j]}"] = float(roc_auc_score(y[te], clf.predict_proba(Xd[te])[:, 1]))

    # 方差分解: 每变量观测率的院间 vs 院内单元间方差
    # 组均值: hospital mean, hospital×unit mean
    decomp = []
    for j, v in enumerate(F.TV):
        col = obs[:, j]
        gm = col.mean()
        # 院间: hospital 组均值方差
        hosp_means = {h: col[hosp == h].mean() for h in set(hosp)}
        var_hosp = np.mean([(hosp_means[h] - gm) ** 2 * (hosp == h).mean() for h in set(hosp)]) * len(set(hosp))
        var_hosp = np.sum([(hosp == h).mean() * (hosp_means[h] - gm) ** 2 for h in set(hosp)])
        # 院内单元间: hospital×unit 组均值 相对 hospital 均值 的方差
        var_unit = 0.0
        for h in set(hosp):
            hm = hosp_means[h]
            for u in set(unit):
                sel = (hosp == h) & (unit == u)
                if sel.sum() < 30: continue
                var_unit += sel.mean() * (col[sel].mean() - hm) ** 2
        var_total = col.var()
        decomp.append(dict(variable=v, var_total=float(var_total),
                           var_between_hospital=float(var_hosp), var_within_hosp_between_unit=float(var_unit),
                           frac_hospital=float(var_hosp / (var_total + 1e-12)),
                           frac_unit=float(var_unit / (var_total + 1e-12))))
    dd = pl.DataFrame(decomp).sort("var_total", descending=True)
    dd.write_csv(f"{TAB}/stage7_sixdomain_variance.csv")

    res = dict(domain_counts={d: int((dom == d).sum()) for d in doms},
               pairwise_mask_domain_auc=pair_auc,
               mean_frac_hospital=float(np.mean([r["frac_hospital"] for r in decomp])),
               mean_frac_within_hosp_unit=float(np.mean([r["frac_unit"] for r in decomp])),
               note="测量指纹方差: 院间 vs 院内单元间。six-domain LODO 基础。")
    json.dump(res, open(f"{STT}/stage7_sixdomain.json", "w"), indent=2, ensure_ascii=False)
    print("=== 六域 mask->域 AUC ==="); print(json.dumps(pair_auc, indent=2))
    print("\n=== 方差分解均值 ===")
    print(f"院间占比 {res['mean_frac_hospital']:.3f}  院内单元间占比 {res['mean_frac_within_hosp_unit']:.3f}")
    print("\ntop var 变量:"); print(dd.head(8).select(["variable", "frac_hospital", "frac_unit"]).to_pandas().to_string(index=False))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
