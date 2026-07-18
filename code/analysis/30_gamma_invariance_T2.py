#!/usr/bin/env python
"""Stage 3 :: T2 gamma-不变性检验 (方案 §Stage3.3, 门 G4)

检验: 测量得分残差 rho_j 与结局的关系 theta_j 是否跨院不变。
  拟合 logit P(Y) = f(value) + sum_j theta_j^h * rho_j   逐院分别拟合 theta_j^h,
  逐变量检验 H0: theta_j^A == theta_j^B  (cluster-robust Wald + BH)。
  rho_j = (M_j - e_j^h(C)) / (e_j^h(C)(1-e_j^h(C)))  测量得分残差, e 用 build_phi。

若多数关键变量拒绝 H0 -> gamma 跨院不同 -> Prop 3 触发(不可迁移)。
这需要两院标签(科学分析, 非部署工具)。
"""
import os, sys, json, time
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from scipy import stats

P = "."
sys.path.insert(0, f"{P}/code/utils")
import features as F

TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"


def build_rows(rows, X, M, D, Y, L, srcnorm, cap=200):
    """展平 -> value(ffill归一化,34), phi, mask, y, pid."""
    Z = F.ffill_impute(np.asarray(X[rows][:, :cap]), np.asarray(M[rows][:, :cap]), srcnorm, channels=F.TV)
    phi, _ = F.build_phi(X[rows], M[rows], L[rows], None, src=None, _norm=srcnorm)
    val, ph, msk, yy, gg = [], [], [], [], []
    for k, i in enumerate(rows):
        n = int(min(L[i], cap))
        val.append(Z[k, :n]); ph.append(phi[k, :n]); msk.append(np.asarray(M[i, :n]).astype(np.float32))
        yy.append(np.asarray(Y[i, :n])); gg.append(np.full(n, i))
    return (np.concatenate(val), np.concatenate(ph), np.concatenate(msk),
            np.concatenate(yy).astype(int), np.concatenate(gg))


def msr(msk, e, eps=0.02, kappa=20.0):
    e = np.clip(e, eps, 1 - eps)
    return np.clip((msk - e) / (e * (1 - e)), -kappa, kappa)


def cluster_robust_theta(y, value_lp, rho_j, groups):
    """拟合 logit(y) = a + b*value_lp + theta*rho_j, 返回 theta 与 cluster-robust SE。
    value_lp: 一个固定的 value-only 线性预测(offset), 控制生理。"""
    Xd = np.column_stack([np.ones_like(rho_j), value_lp, rho_j])
    try:
        m = sm.GLM(y, Xd, family=sm.families.Binomial())
        r = m.fit(cov_type="cluster", cov_kwds={"groups": groups})
        return float(r.params[2]), float(r.bse[2])
    except Exception:
        return np.nan, np.nan


def main():
    t0 = time.time()
    X, M, D, Y, L = F.load_arrays()
    sp = F.load_splits()

    # 用两院 train+val 拟合(标签充足)。归一化统一用 A 的 norm(保证 value 在同坐标系, 使 vlp 可比)。
    norm = F.load_norm("A")
    data = {}
    from ops_repair import MeasurementPropensity
    raw = {}
    for h in ["A", "B"]:
        rows = np.array(sp[f"{h}_train"] + sp[f"{h}_val"])
        v, ph, m, y, g = build_rows(rows, X, M, D, Y, L, norm)
        # 逐变量 e_j(C) 逐院 5 折按病人交叉拟合(OOF), 与 THEORY.md §4 / Stage4 一致(避免 in-sample 过拟合)
        E = MeasurementPropensity(n_splits=5, C=1.0).fit_transform_oof(ph, m, g)
        raw[h] = dict(v=v, R=msr(m, E), y=y, g=g, mrate=m.mean(axis=0))
    # value-only offset: 用两院合并数据拟合"单一共享"value 模型 (方案 §Stage3.3 修正:
    # 共享 offset 使 theta_j^A vs theta_j^B 隔离 mask 通道的结局耦合, 不混入 value 模型差异)
    v_all = np.vstack([raw["A"]["v"], raw["B"]["v"]])
    y_all = np.concatenate([raw["A"]["y"], raw["B"]["y"]])
    vlr = LogisticRegression(max_iter=500, C=0.5, class_weight="balanced").fit(v_all, y_all)
    for h in ["A", "B"]:
        data[h] = dict(R=raw[h]["R"], y=raw[h]["y"], g=raw[h]["g"], mrate=raw[h]["mrate"],
                       vlp=vlr.decision_function(raw[h]["v"]))
        print(f"[{h}] rows={len(raw[h]['y'])} (shared value offset) ({time.time()-t0:.0f}s)", flush=True)

    # 逐变量 theta_j^A, theta_j^B + Wald 检验 theta 差异
    out = []
    for j in range(34):
        name = F.TV[j]
        thetas = {}; ses = {}
        ok = True
        for h in ["A", "B"]:
            d = data[h]
            if d["mrate"][j] < 0.005 or d["mrate"][j] > 0.995:
                ok = False; break
            # 子抽样以控制 GLM 规模(cluster-robust 慢)
            idx = np.arange(len(d["y"]))
            if len(idx) > 200000:
                rng = np.random.default_rng(0); idx = rng.choice(idx, 200000, replace=False)
            th, se = cluster_robust_theta(d["y"][idx], d["vlp"][idx], d["R"][idx, j], d["g"][idx])
            thetas[h] = th; ses[h] = se
        if not ok or any(np.isnan(list(thetas.values()))):
            continue
        z = (thetas["A"] - thetas["B"]) / np.sqrt(ses["A"] ** 2 + ses["B"] ** 2 + 1e-12)
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
        out.append(dict(variable=name, theta_A=thetas["A"], theta_B=thetas["B"],
                        se_A=ses["A"], se_B=ses["B"], z=float(z), p_raw=float(pval),
                        mrate_A=float(data["A"]["mrate"][j]), mrate_B=float(data["B"]["mrate"][j])))
        print(f"  {name:16s} thetaA={thetas['A']:+.3f} thetaB={thetas['B']:+.3f} z={z:+.2f} p={pval:.3g}", flush=True)

    df = pl.DataFrame(out)
    # BH 校正
    pv = df["p_raw"].to_numpy()
    order = np.argsort(pv); m_tests = len(pv)
    bh = np.empty(m_tests); bh[order] = np.minimum.accumulate((pv[order] * m_tests / (np.arange(m_tests) + 1))[::-1])[::-1]
    df = df.with_columns(pl.Series("p_bh", np.clip(bh, 0, 1)))
    df = df.with_columns((pl.col("p_bh") < 0.05).alias("reject_gamma_invariant"))
    df.sort("p_raw").write_csv(f"{TAB}/stage3_T2_gamma.csv")

    n_reject = int(df["reject_gamma_invariant"].sum())
    gates = dict(n_variables_tested=m_tests, n_reject_gamma_invariant=n_reject,
                 frac_reject=float(n_reject / m_tests),
                 G4_majority_reject=bool(n_reject > m_tests / 2),
                 G4_note="若多数拒绝 gamma 不变 -> Prop3 触发(测量通道不可迁移)")
    json.dump(gates, open(f"{STT}/stage3_T2_gates.json", "w"), indent=2)
    print("\n=== T2 / G4 ==="); print(json.dumps(gates, indent=2))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
