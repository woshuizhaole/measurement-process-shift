#!/usr/bin/env python
"""Stage 6 :: MNAR delta-tilt 敏感性 (方案 §6.7) + gamma 失配曲线增强

MNAR 敏感性: (A3)无自掩蔽不可检验。注入 pattern-mixture delta-tilt:
  logit P(M_j=1) += delta_j * x_std_j (自掩蔽强度), 扫 delta, 报告 MSR 校正结论(R2 vs value_mask)的稳健性。
  在合成数据上做(真实数据无真值)。若结论(MSR 不胜)在 delta 带内稳定, 则自掩蔽非致命。

gamma 失配增强: 找到 MSR 优于 value_mask 的区域(若存在)——Prop3 的定量边界。
  设计: mask 的捷径成分(alpha/beta 协议偏移)必须足够大, MSR 剥离它才有价值。
"""
import os, sys, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

P = "."
STT = f"{P}/results/stats"; TAB = f"{P}/results/tables"
sig = lambda z: 1 / (1 + np.exp(-z))


def gen(n, dc, dj, alpha, beta, gamma, lam_c, lam_j, mu, a0, a2, rng, delta_mnar=None):
    G = rng.normal(size=n)
    C = lam_c[None, :] * G[:, None] + rng.normal(0, 1, (n, dc))
    Xv = mu[None, :] + lam_j[None, :] * G[:, None] + rng.normal(0, 1, (n, dj))
    eta = alpha[None, :] + C @ beta.T + gamma[None, :] * G[:, None]
    if delta_mnar is not None:
        Xstd = (Xv - Xv.mean(0)) / (Xv.std(0) + 1e-8)
        eta = eta + delta_mnar[None, :] * Xstd            # 自掩蔽: 下单依赖自身值
    xi = (rng.random((n, dj)) < sig(eta)).astype(float)
    y = (rng.random(n) < sig(a0 + a2 * G)).astype(float)
    return C, G, xi, Xv, y


def prop(C, xi, dj, eps=.02):
    E = np.zeros_like(xi)
    for j in range(dj):
        if xi[:, j].std() < 1e-6:
            E[:, j] = np.clip(xi[:, j].mean(), eps, 1 - eps); continue
        E[:, j] = LogisticRegression(max_iter=300).fit(C, xi[:, j]).predict_proba(C)[:, 1]
    return np.clip(E, eps, 1 - eps)


def eval_methods(seed, dc=4, dj=10, gamma_scale=1.0, alpha_shift=2.0, beta_shift=0.4,
                 gamma_mismatch=0.0, delta_mnar_scale=0.0, n=8000):
    rng = np.random.default_rng(seed)
    lam_c = rng.normal(0, .3, dc); lam_j = rng.normal(0, .5, dj); mu = rng.normal(0, 2, dj)
    alphaA = rng.uniform(-2, 0, dj); betaA = rng.normal(0, .5, (dj, dc))
    gammaA = np.abs(rng.normal(0, gamma_scale, dj))
    alphaB = alphaA + rng.normal(0, alpha_shift, dj); betaB = betaA + rng.normal(0, beta_shift, (dj, dc))
    gammaB = gammaA + rng.normal(0, gamma_mismatch, dj)
    dmnar = rng.normal(0, delta_mnar_scale, dj) if delta_mnar_scale > 0 else None
    a0, a2 = -1.2, 1.6
    CA, GA, xiA, XA, yA = gen(n, dc, dj, alphaA, betaA, gammaA, lam_c, lam_j, mu, a0, a2, rng, dmnar)
    CB, GB, xiB, XB, yB = gen(n, dc, dj, alphaB, betaB, gammaB, lam_c, lam_j, mu, a0, a2, rng, dmnar)
    EA, EB = prop(CA, xiA, dj), prop(CB, xiB, dj)
    RA = np.clip((xiA - EA) / (EA * (1 - EA)), -20, 20); RB = np.clip((xiB - EB) / (EB * (1 - EB)), -20, 20)
    imp = lambda X, xi: np.where(xi > 0, X, 0.0)
    feats = {"value_only": (np.hstack([imp(XA, xiA), CA]), np.hstack([imp(XB, xiB), CB])),
             "value_mask": (np.hstack([imp(XA, xiA), xiA, CA]), np.hstack([imp(XB, xiB), xiB, CB])),
             "value_msr": (np.hstack([imp(XA, xiA), RA, CA]), np.hstack([imp(XB, xiB), RB, CB]))}
    out = {}
    for k, (FA, FB) in feats.items():
        clf = LogisticRegression(max_iter=500).fit(FA, yA)
        out[k] = roc_auc_score(yB, clf.predict_proba(FB)[:, 1])
    return out


def main():
    t0 = time.time()
    os.makedirs(STT, exist_ok=True)
    import polars as pl

    # === MNAR delta-tilt 敏感性 ===
    mnar_rows = []
    for dm in [0.0, 0.25, 0.5, 1.0, 1.5]:
        acc = {"value_only": [], "value_mask": [], "value_msr": []}
        for s in range(5):
            r = eval_methods(s, delta_mnar_scale=dm)
            for k in acc: acc[k].append(r[k])
        rec = dict(delta_mnar=dm, **{k: float(np.mean(v)) for k, v in acc.items()})
        rec["msr_minus_mask"] = rec["value_msr"] - rec["value_mask"]
        mnar_rows.append(rec)
        print(f"delta_mnar={dm}: vo={rec['value_only']:.3f} vmask={rec['value_mask']:.3f} "
              f"vmsr={rec['value_msr']:.3f} (msr-mask={rec['msr_minus_mask']:+.3f})", flush=True)
    pl.DataFrame(mnar_rows).write_csv(f"{TAB}/stage6_mnar_sensitivity.csv")

    # === gamma 失配增强: 扫 gamma_mismatch, 看 MSR 何时优于 value_mask ===
    gm_rows = []
    for gm in [0.0, 0.5, 1.0, 2.0, 3.0]:
        acc = {"value_only": [], "value_mask": [], "value_msr": []}
        for s in range(5):
            r = eval_methods(s, gamma_mismatch=gm, alpha_shift=3.0)
            for k in acc: acc[k].append(r[k])
        rec = dict(gamma_mismatch=gm, **{k: float(np.mean(v)) for k, v in acc.items()})
        rec["msr_minus_mask"] = rec["value_msr"] - rec["value_mask"]
        gm_rows.append(rec)
        print(f"gamma_mismatch={gm}: vmask={rec['value_mask']:.3f} vmsr={rec['value_msr']:.3f} "
              f"(msr-mask={rec['msr_minus_mask']:+.3f})", flush=True)
    pl.DataFrame(gm_rows).write_csv(f"{TAB}/stage6_gamma_mismatch_enhanced.csv")

    # 结论
    mnar_stable = all(r["msr_minus_mask"] <= 0.01 for r in mnar_rows)
    summary = dict(
        mnar_msr_never_beats_mask=bool(all(r["msr_minus_mask"] <= 0.01 for r in mnar_rows)),
        mnar_conclusion="MSR 在自掩蔽 delta 带内均未超过 value_mask -> 负结论对 MNAR 稳健" if mnar_stable else "MNAR 下 MSR 有区域超 mask",
        gamma_msr_beats_mask_at=[r["gamma_mismatch"] for r in gm_rows if r["msr_minus_mask"] > 0.005],
        note="合成数据: MNAR 敏感性 + gamma 失配下 MSR vs value_mask 边界")
    json.dump(summary, open(f"{STT}/stage6_sensitivity.json", "w"), indent=2, ensure_ascii=False)
    print("\n=== 敏感性小结 ==="); print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
