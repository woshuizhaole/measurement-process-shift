#!/usr/bin/env python
"""Stage 2 :: 跨院缺口的通道分解 (早期方案) + 仅患病率再校准消融(§Stage2.4, 门 G3)

修正说明(相对初版): 初版把 V(值分布)/P(患病率) 用 sample_weight 重加权源域测试集来"模拟目标域",
但重加权不改变特征值, 无法真正把源值分布变成目标值分布 -> V/P 被结构性低估, M 虚高吸收全部,
且 total_drop(0.017) != 真实源->目标缺口(0.074)。此为实现缺陷, 已弃用。

改用"真实队列 + 掩码移植锚定 M 通道"的可辨识分解 (与 21_mask_transplant 一致):
  总缺口 G = AUC(源测试) - AUC(目标测试)                              [真实, 用目标域标签]
  M 通道  = AUC(源测试) - AUC(源病人移植到目标下单习惯)               [掩码移植, 反事实]
  V+P 残差 = G - M                                                      [值分布+患病率, 合并]
  再把 V+P 用"患病率匹配"进一步拆:
    P 通道 = AUC(目标测试) - AUC(目标测试 患病率匹配到源率)            [仅改患病率]
    V 通道 = (V+P残差) - P                                            [残余=值分布]
门 G3(仅患病率再校准): BBSE 估目标患病率 pi_t + King-Zeng 截距校正, 报告关闭的 Utility 缺口比例。
"""
import os, sys, json, time
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

P = "."
sys.path.insert(0, f"{P}/code/utils")
import features as F
from models import normalized_utility, best_threshold

TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"
SEED = 20260711


def flatten(rows, X, M, Y, L, cap, srcnorm, meta):
    Z = F.ffill_impute(np.asarray(X[rows][:, :cap]), np.asarray(M[rows][:, :cap]), srcnorm, channels=F.TV)
    phi, _ = F.build_phi(X[rows], M[rows], L[rows], meta[list(rows)], src=None, _norm=srcnorm)
    vv, mm, yy, gg, pp = [], [], [], [], []
    for k, i in enumerate(rows):
        n = int(min(L[i], cap))
        vv.append(Z[k, :n]); mm.append(M[i, :n].astype(np.float32))
        yy.append(Y[i, :n]); gg.append(np.full(n, i)); pp.append(phi[k, :n])
    return (np.concatenate(vv), np.concatenate(mm), np.concatenate(yy).astype(int),
            np.concatenate(gg), np.concatenate(pp))


def fit_prop(PH, MM):
    out = []
    for j in range(34):
        col = MM[:, j].astype(int); r = col.mean()
        out.append(("const", float(np.clip(r, 0.005, 0.995))) if (r < 0.002 or r > 0.998)
                   else ("lr", LogisticRegression(max_iter=500, C=1.0).fit(PH, col)))
    return out


def pred_e(models, PH):
    E = np.zeros((PH.shape[0], 34), np.float32)
    for j, (k, m) in enumerate(models):
        E[:, j] = m if k == "const" else m.predict_proba(PH)[:, 1]
    return np.clip(E, 0.005, 0.995)


def prevalence_matched_auc(y, p, target_prev, rng, reps=20):
    """把评估集重采样到 target_prev 的病人级患病率后算 AUROC (真正改变患病率, 非重加权)。"""
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos) < 5 or len(neg) < 5:
        return roc_auc_score(y, p)
    n_total = len(y); n_pos = int(round(target_prev * n_total))
    n_pos = max(5, min(n_pos, len(pos)))
    n_neg = n_total - n_pos
    aucs = []
    for _ in range(reps):
        pi = rng.choice(pos, n_pos, replace=len(pos) < n_pos)
        ni = rng.choice(neg, n_neg, replace=len(neg) < n_neg)
        idx = np.r_[pi, ni]
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.mean(aucs))


def main():
    t0 = time.time()
    X, M, D, Y, L = F.load_arrays()
    meta = F.load_meta()
    sp = F.load_splits()
    rng = np.random.default_rng(SEED)
    cap = 200
    out = []

    for direction in ["A2B", "B2A"]:
        src, tgt = ("A", "B") if direction == "A2B" else ("B", "A")
        srcnorm = F.load_norm(src)
        tr = rng.choice(sp[f"{src}_train"], min(13015, len(sp[f"{src}_train"])), replace=False)
        ste = np.array(sp[f"{src}_test"]); tte = np.array(sp[f"{tgt}_test"])

        Vtr, Mtr, ytr, _, PHtr = flatten(np.array(tr), X, M, Y, L, cap, srcnorm, meta)
        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                 min_child_samples=100, n_jobs=28, verbose=-1, scale_pos_weight=15)
        clf.fit(np.hstack([Vtr, Mtr]), ytr)
        e_src = fit_prop(PHtr, Mtr)

        Vt, Mt, yt, gt, PHt = flatten(tte, X, M, Y, L, cap, F.load_norm(tgt), meta)
        e_tgt = fit_prop(PHt, Mt)
        Vs, Ms, ys, gs, PHs = flatten(ste, X, M, Y, L, cap, srcnorm, meta)

        def auc_of(V_, M_, y_):
            return roc_auc_score(y_, clf.predict_proba(np.hstack([V_, M_]))[:, 1])

        # 真实队列
        auc_src = auc_of(Vs, Ms, ys)
        p_tgt = clf.predict_proba(np.hstack([Vt, Mt]))[:, 1]
        auc_tgt = roc_auc_score(yt, p_tgt)
        G = auc_src - auc_tgt                                   # 总缺口

        # M 通道: 掩码移植 (源病人 -> 目标下单习惯)
        e_s_on_s = pred_e(e_src, PHs); e_t_on_s = pred_e(e_tgt, PHs)
        pkeep = np.clip(e_t_on_s / e_s_on_s, 0, 1)
        Vx = Vs.copy(); Mx = Ms.copy()
        drop = (Ms > 0) & (rng.random(Ms.shape) > pkeep)
        Mx[drop] = 0; Vx[drop] = 0.0
        auc_transplant = auc_of(Vx, Mx, ys)
        chan_M = auc_src - auc_transplant                      # 测量过程通道

        # P 通道: 目标测试患病率匹配到源率 (真正改变患病率)
        pi_s = ys.mean(); pi_t = yt.mean()
        auc_tgt_prevmatched = prevalence_matched_auc(yt, p_tgt, pi_s, rng)
        chan_P = auc_tgt_prevmatched - auc_tgt                 # 患病率通道对缺口的贡献
        # V+P 残差, V = 残差 - P
        chan_VP = G - chan_M
        chan_V = chan_VP - chan_P

        rec = dict(direction=direction, auc_src=float(auc_src), auc_tgt=float(auc_tgt),
                   total_gap=float(G), chan_M_measurement=float(chan_M),
                   chan_P_prevalence=float(chan_P), chan_V_value=float(chan_V),
                   frac_M=float(chan_M / (G + 1e-9)), frac_P=float(chan_P / (G + 1e-9)),
                   frac_V=float(chan_V / (G + 1e-9)), pi_src=float(pi_s), pi_tgt=float(pi_t))
        out.append(rec)
        print(f"[{direction}] gap={G:+.3f} | M(测量)={chan_M:+.3f}({rec['frac_M']:.0%}) "
              f"P(患病率)={chan_P:+.3f}({rec['frac_P']:.0%}) V(值)={chan_V:+.3f}({rec['frac_V']:.0%}) "
              f"({time.time()-t0:.0f}s)", flush=True)

    df = pl.DataFrame(out)
    df.write_csv(f"{TAB}/stage2_channel_decomp.csv")
    print("\n=== 通道分解 (M 测量 / P 患病率 / V 值) ===")
    print(df.to_pandas().to_string(index=False))

    # 门 G3: 患病率通道占比 (仅患病率再校准能关闭多少缺口)
    fracs_P = [abs(r["chan_P_prevalence"]) / (abs(r["total_gap"]) + 1e-9) for r in out]
    gates = dict(frac_gap_from_prevalence=[float(f) for f in fracs_P],
                 max_frac_P=float(max(fracs_P)),
                 G3_prevalence_dominates=bool(max(fracs_P) >= 0.8),
                 G3_note="若患病率通道>=80% -> 缺口本质是先验偏移(转再校准论文)")
    json.dump(gates, open(f"{STT}/stage2_G3_prevalence.json", "w"), indent=2)
    print("\n=== G3 ==="); print(json.dumps(gates, indent=2))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
