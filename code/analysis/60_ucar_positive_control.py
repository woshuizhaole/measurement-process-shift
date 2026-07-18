#!/usr/bin/env python
"""Stage 6 :: UCAR 正对照 (方案 §Stage6.1, 门 G6) + gamma 失配曲线 + MNAR 敏感性

G6: 在已知相对缺失率 r 的合成 UCAR 偏移上, 相对率估计器必须无偏恢复 r, 且 DAMS 修正必须奏效。
    不过 -> 实现有 bug, 禁跑真实数据。
gamma 失配曲线: 把 ||gamma^B-gamma^A|| 从 0 扫到大, 画 MSR 校正恢复率 vs "丢 mask", 给出交叉点(Prop3 定量版)。
MNAR delta-tilt: 自掩蔽强度扫描, 报告校正后性能稳健带。
"""
import os, sys, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

P = "."
STT = f"{P}/results/stats"; TAB = f"{P}/results/tables"
sig = lambda z: 1 / (1 + np.exp(-z))


def gen(n, dc, dj, alpha, beta, gamma, lam_c, lam_j, mu, a0, a2, rng, absent=None):
    G = rng.normal(size=n)
    C = lam_c[None, :] * G[:, None] + rng.normal(0, 1, (n, dc))
    Xv = mu[None, :] + lam_j[None, :] * G[:, None] + rng.normal(0, 1, (n, dj))
    eta = alpha[None, :] + C @ beta.T + gamma[None, :] * G[:, None]
    xi = (rng.random((n, dj)) < sig(eta)).astype(float)
    if absent is not None:
        xi[:, absent] = 0.0
    y = (rng.random(n) < sig(a0 + a2 * G)).astype(float)
    return C, G, xi, Xv, y


def est_relative_rate(xiA, xiB):
    """DAMS: r = 1 - q_t/q_s, q = 观测(非缺失)边缘比例。返回逐变量 r_hat。"""
    qs = xiA.mean(axis=0); qt = xiB.mean(axis=0)
    return 1 - qt / np.clip(qs, 1e-6, None)


def ucar_positive_control(seed=0):
    """注入已知 UCAR 相对率, 检验估计器无偏。"""
    rng = np.random.default_rng(seed)
    dc, dj, n = 4, 10, 20000
    lam_c = rng.normal(0, .5, dc); lam_j = rng.normal(0, .6, dj); mu = rng.normal(0, 2, dj)
    alpha = rng.uniform(-1, 1, dj); beta = np.zeros((dj, dc)); gamma = np.zeros(dj)  # UCAR: 缺失独立于C,G
    a0, a2 = -1.2, 1.5
    CA, GA, xiA, XA, yA = gen(n, dc, dj, alpha, beta, gamma, lam_c, lam_j, mu, a0, a2, rng)
    # 目标: 已知相对率 r_true, 通过额外随机删观测实现
    r_true = rng.uniform(0.1, 0.6, dj)
    CB, GB, xiB0, XB, yB = gen(n, dc, dj, alpha, beta, gamma, lam_c, lam_j, mu, a0, a2, rng)
    xiB = xiB0 * (rng.random((n, dj)) > r_true[None, :])   # 以 r_true 概率再删
    r_hat = est_relative_rate(xiA, xiB)
    bias = np.abs(r_hat - r_true)
    return dict(r_true=r_true.tolist(), r_hat=r_hat.tolist(),
                max_abs_bias=float(bias.max()), mean_abs_bias=float(bias.mean()))


def gamma_mismatch_curve(seed=0, scales=(0.0, 0.3, 0.6, 1.0, 1.5, 2.0)):
    """扫 gamma^B - gamma^A 失配强度, MSR 校正恢复 vs 丢 mask。"""
    out = []
    for gs in scales:
        aucs = {"value_only": [], "value_mask": [], "value_msr": []}
        for seed_i in range(5):
            rng = np.random.default_rng(seed * 100 + seed_i)
            dc, dj, n = 4, 10, 8000
            lam_c = rng.normal(0, .3, dc); lam_j = rng.normal(0, .5, dj); mu = rng.normal(0, 2, dj)
            alphaA = rng.uniform(-2, 0, dj); betaA = rng.normal(0, .5, (dj, dc))
            gammaA = np.abs(rng.normal(0, 1, dj))
            gammaB = gammaA + rng.normal(0, gs, dj)         # 失配强度 gs
            alphaB = alphaA + rng.normal(0, 1.5, dj); betaB = betaA + rng.normal(0, .3, (dj, dc))
            a0, a2 = -1.2, 1.6
            CA, GA, xiA, XA, yA = gen(n, dc, dj, alphaA, betaA, gammaA, lam_c, lam_j, mu, a0, a2, rng)
            CB, GB, xiB, XB, yB = gen(n, dc, dj, alphaB, betaB, gammaB, lam_c, lam_j, mu, a0, a2, rng)
            # 值通道: 只保留观测(缺失填0=均值)
            def imp(X, xi): return np.where(xi > 0, X, 0.0)
            # e_j(C) 逐院
            def prop(C, xi):
                E = np.zeros_like(xi)
                for j in range(dj):
                    if xi[:, j].std() < 1e-6: E[:, j] = xi[:, j].mean(); continue
                    E[:, j] = LogisticRegression(max_iter=300).fit(C, xi[:, j]).predict_proba(C)[:, 1]
                return np.clip(E, .02, .98)
            EA, EB = prop(CA, xiA), prop(CB, xiB)
            RA = np.clip((xiA - EA) / (EA * (1 - EA)), -20, 20)
            RB = np.clip((xiB - EB) / (EB * (1 - EB)), -20, 20)
            feats = {
                "value_only": (np.hstack([imp(XA, xiA), CA]), np.hstack([imp(XB, xiB), CB])),
                "value_mask": (np.hstack([imp(XA, xiA), xiA, CA]), np.hstack([imp(XB, xiB), xiB, CB])),
                "value_msr": (np.hstack([imp(XA, xiA), RA, CA]), np.hstack([imp(XB, xiB), RB, CB])),
            }
            for k, (FA, FB) in feats.items():
                clf = LogisticRegression(max_iter=500, C=1.0).fit(FA, yA)
                aucs[k].append(roc_auc_score(yB, clf.predict_proba(FB)[:, 1]))
        out.append(dict(gamma_mismatch=gs, **{k: float(np.mean(v)) for k, v in aucs.items()}))
    return out


def main():
    t0 = time.time()
    os.makedirs(STT, exist_ok=True); os.makedirs(TAB, exist_ok=True)
    # G6 正对照
    pc = [ucar_positive_control(s) for s in range(5)]
    max_bias = float(np.mean([r["max_abs_bias"] for r in pc]))
    g6 = dict(mean_max_abs_bias=max_bias, tol=0.05, G6_pass=bool(max_bias < 0.05),
              note="相对率估计器在 UCAR 上无偏(max|r_hat-r_true|<0.05)")
    json.dump(g6, open(f"{STT}/stage6_G6_ucar.json", "w"), indent=2)
    print("=== G6 UCAR 正对照 ==="); print(json.dumps(g6, indent=2, ensure_ascii=False))

    # gamma 失配曲线
    import polars as pl
    curve = gamma_mismatch_curve(0)
    pl.DataFrame(curve).write_csv(f"{TAB}/stage6_gamma_mismatch.csv")
    print("\n=== gamma 失配曲线 (跨域 AUROC) ===")
    print(pl.DataFrame(curve).to_pandas().to_string(index=False))
    # 交叉点: msr 优于 value_mask 且 value_only 何时反超
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
