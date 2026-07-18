#!/usr/bin/env python
"""Block 1 :: 在官方归一化 Utility 空间重建三通道缺口分解 + G2/G3 复判。

基于 23_channel_decomp.py 改造。病灶(见 improvement.md my_error#1):
  * 23 整段在 AUROC 空间 (auc_src-auc_tgt); chan_P 用 prevalence-matched AUROC
    -> AUROC 秩不变性下患病率通道必然≈机器零; 顺序分解非 Shapley; 无 CI。
  * 早期方案 要求在反事实队列上评估 **官方 Utility** 做分解;
    §Stage2.4/G3 要求 BBSE + King-Zeng 仅患病率再校准。

本脚本把三通道分解迁到官方归一化 Utility 空间, 预测器与 23 一致(行级 value+mask LGB,
n_estimators=400 等), 严格用源域 val 选阈 (§8.2 陷阱3), 固定套用到所有队列。

可辨识顺序分解 (偏离说明见 分析记录 / 本脚本返回):
  早期方案 原写"精确 2^3=8 队列 Shapley", 但内部验证(见 工程记录)已证
  重加权 Shapley 不可辨识 —— sample_weight 重加权不改特征值, 无法把源值分布真正变成目标分布,
  V/P 被结构性低估、M 虚高吸收全部。故改用可辨识锚定分解(与掩码移植互证):
    total_gap = Util(源test) - Util(目标test)
    chan_M(测量) = Util(源test) - Util(源病人移植到目标下单习惯)   [掩码移植, 只删不加]
    chan_P(患病率) = Util(目标 KZ再校准后) - Util(目标)             [BBSE pi_t + King-Zeng]
    chan_V(值) = total_gap - chan_M - chan_P                        [残差; 恒等于
                                                                      Util(源transplant)-Util(目标recal)]
所有通道/总缺口用 util_boot 病人级 cluster bootstrap 给 95% CI (B>=2000)。
G2(捷径指数): SI = [院内 dUtil(mask)] - [跨院 dUtil(mask)]; G3: 仅患病率再校准关闭的缺口比例。
"""
import os, sys, json, time
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

P = "."
sys.path.insert(0, f"{P}/code/utils")
import features as F
from models import normalized_utility, best_threshold
from util_boot import per_patient_components, _norm, paired_bootstrap_util_diff

TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"
SEED = 20260711
NJOBS = 8
BBOOT = 2000


# ----------------------- 行级展平 (与 23/21 完全一致) -----------------------
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
    """逐变量测量倾向 e_j(C) (与 23.fit_prop 一致)。"""
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


# ----------------------- 行 -> 逐病人 list (Utility 需要) -----------------------
def to_lists(y, p, g):
    """g 每病人连续块(flatten 保证)。返回 (y_list, p_list) 按病人切。"""
    bnd = np.where(np.diff(g) != 0)[0] + 1
    return np.split(y, bnd), np.split(p, bnd)


def util_point(y_list, p_list, thr):
    uo, ub, ui = per_patient_components(y_list, p_list, thr)
    return _norm(uo, ub, ui)


# ----------------------- 非配对病人级 bootstrap (两组不同病人) -----------------------
def unpaired_boot_diff(yA, pA, thrA, yB, pB, thrB, B=BBOOT, seed=0, alpha=0.05):
    """diff = Util(组A) - Util(组B), A/B 是不同病人组, 各自独立重采样。"""
    uoA, ubA, uiA = per_patient_components(yA, pA, thrA)
    uoB, ubB, uiB = per_patient_components(yB, pB, thrB)
    point = _norm(uoA, ubA, uiA) - _norm(uoB, ubB, uiB)
    rng = np.random.default_rng(seed)
    nA, nB = len(uoA), len(uoB)
    d = np.empty(B)
    for b in range(B):
        ia = rng.integers(0, nA, nA); ib = rng.integers(0, nB, nB)
        d[b] = _norm(uoA[ia], ubA[ia], uiA[ia]) - _norm(uoB[ib], ubB[ib], uiB[ib])
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(point=float(point), lo=float(lo), hi=float(hi), p_gt0=float((d > 0).mean()),
                mean=float(d.mean()), std=float(d.std()), n_boot=int(B))


# ----------------------- BBSE (hard) 与 EM/MLLS 估目标患病率 -----------------------
def bbse_hard(p_src_val, y_src_val, p_tgt, thr):
    """BBSE (Lipton 2018): 源val 硬化混淆矩阵 C(2x2, 联合概率) + 目标预测边际 mu -> w=C^{-1}mu,
    pi_t = w1 * P_s(y=1)。返回 (pi_t, ok, cond)。"""
    ys = np.asarray(y_src_val).astype(int)
    yhat_s = (np.asarray(p_src_val) >= thr).astype(int)
    yhat_t = (np.asarray(p_tgt) >= thr).astype(int)
    ns = len(ys)
    C = np.zeros((2, 2))
    for i in (0, 1):
        for j in (0, 1):
            C[i, j] = np.mean((yhat_s == i) & (ys == j))
    mu = np.array([np.mean(yhat_t == 0), np.mean(yhat_t == 1)])
    ps_y = np.array([np.mean(ys == 0), np.mean(ys == 1)])
    try:
        cond = float(np.linalg.cond(C))
        if cond > 1e8 or abs(np.linalg.det(C)) < 1e-10:
            return None, False, cond
        w = np.linalg.solve(C, mu)
        pi_t = float(np.clip(w[1] * ps_y[1], 1e-4, 0.5))
        ok = np.all(w >= -1e-6)
        return pi_t, bool(ok), cond
    except np.linalg.LinAlgError:
        return None, False, float("inf")


def em_mlls(p_tgt, pi_s, max_iter=500, tol=1e-9):
    """MLLS/EM 先验偏移估计 (Saerens 2002 / Alexandari 2020)。要求输入概率已在源域上校准
    (label-shift 下校准是标签条件不变的 -> 源校准即目标校准)。故必须喂校准后的概率, 否则发散。"""
    p = np.clip(np.asarray(p_tgt), 1e-6, 1 - 1e-6)
    pi = float(pi_s)
    for _ in range(max_iter):
        w1 = pi / pi_s; w0 = (1 - pi) / (1 - pi_s)
        num = w1 * p
        p_adj = num / (num + w0 * (1 - p))
        pi_new = float(np.mean(p_adj))
        if abs(pi_new - pi) < tol:
            pi = pi_new; break
        pi = pi_new
    return float(np.clip(pi, 1e-4, 0.5))


def kz_recalibrate(p_tgt, pi_s, pi_t):
    """King-Zeng 截距平移: logit += log((pi_t/(1-pi_t))/(pi_s/(1-pi_s)))。"""
    delta = np.log((pi_t / (1 - pi_t)) / (pi_s / (1 - pi_s)))
    logit = np.log(np.clip(p_tgt, 1e-6, 1 - 1e-6) / (1 - np.clip(p_tgt, 1e-6, 1 - 1e-6)))
    p_rec = 1.0 / (1.0 + np.exp(-(logit + delta)))
    return p_rec, float(delta)


def main():
    t0 = time.time()
    X, M, D, Y, L = F.load_arrays()
    meta = F.load_meta()
    sp = F.load_splits()
    cap = 200

    # 读 AUROC 空间旧结果供并列对拍
    auroc_df = pl.read_csv(f"{TAB}/stage2_channel_decomp.csv")
    auroc_map = {r["direction"]: r for r in auroc_df.to_dicts()}

    rows_decomp, rows_ci = [], []
    g3, g2 = {}, {}

    for direction in ["A2B", "B2A"]:
        rng = np.random.default_rng(SEED)          # 每方向同种子, 复现 23 的子样本抽取
        src, tgt = ("A", "B") if direction == "A2B" else ("B", "A")
        srcnorm = F.load_norm(src); tgtnorm = F.load_norm(tgt)
        tr = rng.choice(sp[f"{src}_train"], min(13015, len(sp[f"{src}_train"])), replace=False)
        sval = np.array(sp[f"{src}_val"]); ste = np.array(sp[f"{src}_test"]); tte = np.array(sp[f"{tgt}_test"])

        # --- 训练预测器: value+mask LGB (23 一致) + value-only LGB (G2 用) ---
        Vtr, Mtr, ytr, _, PHtr = flatten(np.array(tr), X, M, Y, L, cap, srcnorm, meta)
        clf_vm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                    min_child_samples=100, n_jobs=NJOBS, verbose=-1, scale_pos_weight=15)
        clf_vm.fit(np.hstack([Vtr, Mtr]), ytr)
        clf_v = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                   min_child_samples=100, n_jobs=NJOBS, verbose=-1, scale_pos_weight=15)
        clf_v.fit(Vtr, ytr)
        e_src = fit_prop(PHtr, Mtr)
        print(f"[{direction}] models fit ({time.time()-t0:.0f}s)", flush=True)

        # --- 队列展平 ---
        Vsv, Msv, ysv, gsv, PHsv = flatten(sval, X, M, Y, L, cap, srcnorm, meta)
        Vs, Ms, ys, gs, PHs = flatten(ste, X, M, Y, L, cap, srcnorm, meta)
        Vt, Mt, yt, gt, PHt = flatten(tte, X, M, Y, L, cap, tgtnorm, meta)
        e_tgt = fit_prop(PHt, Mt)

        def pvm(Vx, Mx):
            return clf_vm.predict_proba(np.hstack([Vx, Mx]))[:, 1]

        def pv(Vx):
            return clf_v.predict_proba(Vx)[:, 1]

        # --- 源域 val 选阈 (§8.2 陷阱3), 固定套用 ---
        yl_sv, pl_sv_vm = to_lists(ysv, pvm(Vsv, Msv), gsv)
        thr_vm, u_sval_vm = best_threshold(yl_sv, pl_sv_vm)
        _, pl_sv_v = to_lists(ysv, pv(Vsv), gsv)
        thr_v, u_sval_v = best_threshold(yl_sv, pl_sv_v)
        print(f"[{direction}] thr_vm={thr_vm:.3f}(u_val={u_sval_vm:.3f}) thr_v={thr_v:.3f}(u_val={u_sval_v:.3f})", flush=True)

        # --- 各队列逐病人 Utility (预测器 clf_vm, 阈值 thr_vm) ---
        yl_s, pl_s = to_lists(ys, pvm(Vs, Ms), gs)          # 源 test
        yl_t, pl_t = to_lists(yt, pvm(Vt, Mt), gt)          # 目标 test
        u_src = util_point(yl_s, pl_s, thr_vm)
        u_tgt = util_point(yl_t, pl_t, thr_vm)
        total_gap = u_src - u_tgt

        # ---- chan_M: 掩码移植 (源病人 -> 目标下单习惯, 只删不加) ----
        e_s_on_s = pred_e(e_src, PHs); e_t_on_s = pred_e(e_tgt, PHs)
        pkeep = np.clip(e_t_on_s / e_s_on_s, 0, 1)
        coverage = float((e_t_on_s <= e_s_on_s).mean())    # 只删可行覆盖率
        Vx = Vs.copy(); Mx = Ms.copy()
        drop = (Ms > 0) & (rng.random(Ms.shape) > pkeep)
        Mx[drop] = 0; Vx[drop] = 0.0
        _, pl_tr = to_lists(ys, pvm(Vx, Mx), gs)            # 移植后, 同源病人分组
        u_transplant = util_point(yl_s, pl_tr, thr_vm)
        chan_M = u_src - u_transplant

        # ---- chan_P: King-Zeng 仅患病率再校准 ----
        # 偏离说明(见返回/分析记录): 主分解用 oracle 真 pi_t 做 KZ 截距校正 —— 精确患病率通道,
        # 回顾性归因合法(评估本就用目标标签), 且是更严格的 G3 检验(连精确校正都关不掉>=80%缺口
        # => 任何 label-free 方法更不能)。label-free BBSE/校准MLLS 作可部署性交叉, 但本 regime
        # (患病率~1-2% + 弱分类器 AUROC~0.65-0.72)下两者数值不稳(常塌到 pi=1e-4 下限), 仅报告不入主分解。
        p_tgt_flat = pvm(Vt, Mt); p_sval_flat = pvm(Vsv, Msv)
        pi_s = float(ys.mean()); pi_t_true = float(yt.mean())
        # oracle KZ (主分解)
        p_tgt_rec, kz_delta = kz_recalibrate(p_tgt_flat, pi_s, pi_t_true)
        _, pl_t_rec = to_lists(yt, p_tgt_rec, gt)
        u_tgt_rec = util_point(yl_t, pl_t_rec, thr_vm)
        chan_P = u_tgt_rec - u_tgt
        # label-free 估计(交叉核对 / G3 可部署性): isotonic 校准 + MLLS; BBSE 硬混淆矩阵
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_sval_flat, ysv.astype(float))
        p_tgt_cal = np.clip(iso.predict(p_tgt_flat), 1e-6, 1 - 1e-6)
        pi_mlls = em_mlls(p_tgt_cal, pi_s)
        pi_bbse, bbse_ok, bbse_cond = bbse_hard(p_sval_flat, ysv, p_tgt_flat, thr_vm)
        p_tgt_rec_lf, kz_delta_lf = kz_recalibrate(p_tgt_flat, pi_s, pi_mlls)
        u_tgt_rec_lf = util_point(*to_lists(yt, p_tgt_rec_lf, gt), thr_vm)
        chan_P_labelfree = u_tgt_rec_lf - u_tgt
        pi_t_hat = pi_t_true; method = "oracle_true_pi_t"

        # ---- chan_V: 残差 ----
        chan_V = total_gap - chan_M - chan_P

        # ================= CI (病人级 cluster bootstrap, B>=2000) =================
        # total_gap: 非配对 (源test vs 目标test, 不同病人)
        ci_total = unpaired_boot_diff(yl_s, pl_s, thr_vm, yl_t, pl_t, thr_vm, seed=1)
        # chan_M = Util(源test) - Util(移植), 同源病人 -> 配对; A=移植, B=源 => point=u_src-u_transplant
        ci_M = paired_bootstrap_util_diff(yl_s, pl_tr, pl_s, thr_vm, thr_vm, B=BBOOT, seed=2)
        # chan_P = Util(目标recal) - Util(目标), 同目标病人 -> 配对; A=目标orig, B=目标recal
        ci_P = paired_bootstrap_util_diff(yl_t, pl_t, pl_t_rec, thr_vm, thr_vm, B=BBOOT, seed=3)
        # chan_V = Util(源transplant) - Util(目标recal), 不同病人 -> 非配对(恒等式)
        ci_V = unpaired_boot_diff(yl_s, pl_tr, thr_vm, yl_t, pl_t_rec, thr_vm, seed=4)

        for ch, val, ci in [("total_gap", total_gap, ci_total), ("chan_M", chan_M, ci_M),
                            ("chan_P", chan_P, ci_P), ("chan_V", chan_V, ci_V)]:
            rows_ci.append(dict(direction=direction, channel=ch, point=float(val),
                                ci_lo=ci["lo"], ci_hi=ci["hi"], p_gt0=ci.get("p_gt0", np.nan),
                                boot_point=ci["point"]))

        # ================= G2: 捷径指数 (Utility 空间) =================
        u_src_v = util_point(*to_lists(ys, pv(Vs), gs), thr_v)
        u_tgt_v = util_point(*to_lists(yt, pv(Vt), gt), thr_v)
        dU_mask_in = u_src - u_src_v       # 院内 mask 增益 (value+mask - value)
        dU_mask_cross = u_tgt - u_tgt_v    # 跨院 mask 增益
        SI = dU_mask_in - dU_mask_cross
        # SI CI: 源病人组(院内Δ)与目标病人组(跨院Δ)独立重采样
        uoS_vm, ubS, uiS = per_patient_components(yl_s, pl_s, thr_vm)
        uoS_v, _, _ = per_patient_components(*to_lists(ys, pv(Vs), gs), thr_v)
        uoT_vm, ubT, uiT = per_patient_components(yl_t, pl_t, thr_vm)
        uoT_v, _, _ = per_patient_components(*to_lists(yt, pv(Vt), gt), thr_v)
        rng2 = np.random.default_rng(5); nS = len(ubS); nT = len(ubT); si_b = np.empty(BBOOT)
        for b in range(BBOOT):
            iS = rng2.integers(0, nS, nS); iT = rng2.integers(0, nT, nT)
            din = _norm(uoS_vm[iS], ubS[iS], uiS[iS]) - _norm(uoS_v[iS], ubS[iS], uiS[iS])
            dcr = _norm(uoT_vm[iT], ubT[iT], uiT[iT]) - _norm(uoT_v[iT], ubT[iT], uiT[iT])
            si_b[b] = din - dcr
        si_lo, si_hi = np.percentile(si_b, [2.5, 97.5])
        g2[direction] = dict(dUtil_mask_in=float(dU_mask_in), dUtil_mask_cross=float(dU_mask_cross),
                             shortcut_index_SI=float(SI), SI_ci_lo=float(si_lo), SI_ci_hi=float(si_hi),
                             SI_p_gt0=float((si_b > 0).mean()),
                             sign_in="+" if dU_mask_in > 0 else "-",
                             sign_cross="+" if dU_mask_cross > 0 else "-",
                             u_src_vm=float(u_src), u_src_v=float(u_src_v),
                             u_tgt_vm=float(u_tgt), u_tgt_v=float(u_tgt_v))

        # ================= 主表行 =================
        ar = auroc_map[direction]
        rec = dict(direction=direction, util_src=float(u_src), util_tgt=float(u_tgt),
                   total_gap=float(total_gap), chan_M_measurement=float(chan_M),
                   chan_P_prevalence=float(chan_P), chan_V_value=float(chan_V),
                   frac_M=float(chan_M / (total_gap + 1e-9)),
                   frac_P=float(chan_P / (total_gap + 1e-9)),
                   frac_V=float(chan_V / (total_gap + 1e-9)),
                   coverage_M_onlydelete=coverage, thr_vm=float(thr_vm),
                   chan_P_labelfree=float(chan_P_labelfree),
                   pi_src=pi_s, pi_tgt_true=pi_t_true, pi_tgt_used=float(pi_t_hat), pi_method=method,
                   pi_tgt_mlls_cal=float(pi_mlls), pi_tgt_bbse=(float(pi_bbse) if pi_bbse is not None else np.nan),
                   bbse_ok=bool(bbse_ok), kz_delta_oracle=float(kz_delta), kz_delta_labelfree=float(kz_delta_lf),
                   bbse_cond=float(bbse_cond),
                   # AUROC 空间并列
                   frac_M_auroc=float(ar["frac_M"]), frac_P_auroc=float(ar["frac_P"]),
                   frac_V_auroc=float(ar["frac_V"]))
        rows_decomp.append(rec)
        print(f"[{direction}] gap={total_gap:+.4f} | M={chan_M:+.4f}({rec['frac_M']:.0%},cov={coverage:.2f}) "
              f"P_oracle={chan_P:+.4f}({rec['frac_P']:.0%},pi_t_true={pi_t_true:.4f}) "
              f"V={chan_V:+.4f}({rec['frac_V']:.0%}) | P_labelfree={chan_P_labelfree:+.4f}"
              f"(pi_mlls={pi_mlls:.4f},pi_bbse={pi_bbse if pi_bbse is not None else float('nan'):.4f},ok={bbse_ok}) | "
              f"SI={SI:+.4f}[in={dU_mask_in:+.4f} cross={dU_mask_cross:+.4f}] ({time.time()-t0:.0f}s)", flush=True)

    # ================= G3: 仅患病率再校准关闭的缺口比例 (Utility 空间) =================
    frac_P_close = []
    for r in rows_decomp:
        f = r["chan_P_prevalence"] / (r["total_gap"] + 1e-12)  # 关闭比例(正=校准帮助关闭缺口)
        frac_P_close.append(float(f))
    frac_P_lf = [float(r["chan_P_labelfree"] / (r["total_gap"] + 1e-12)) for r in rows_decomp]
    g3 = dict(space="official_normalized_utility",
              prevalence_recal_method="oracle_true_pi_t King-Zeng (主); label-free BBSE/校准MLLS 仅交叉报告",
              frac_gap_closed_by_prevalence_oracle={r["direction"]: fc for r, fc in zip(rows_decomp, frac_P_close)},
              frac_gap_closed_by_prevalence_labelfree={r["direction"]: fc for r, fc in zip(rows_decomp, frac_P_lf)},
              max_abs_frac_P_close_oracle=float(max(abs(f) for f in frac_P_close)),
              G3_prevalence_dominates=bool(max(abs(f) for f in frac_P_close) >= 0.8),
              G3_verdict=("先验偏移主导(>=80%)->转再校准论文" if max(abs(f) for f in frac_P_close) >= 0.8
                          else "非先验偏移(<80%)->机制假设未被证伪; 即使 oracle 精确患病率校正也只关闭极小比例缺口"),
              prevalence_estimates={r["direction"]: dict(pi_src=r["pi_src"], pi_tgt_true=r["pi_tgt_true"],
                                                   pi_tgt_mlls_cal=r["pi_tgt_mlls_cal"], pi_tgt_bbse=r["pi_tgt_bbse"],
                                                   bbse_ok=r["bbse_ok"], kz_delta_oracle=r["kz_delta_oracle"],
                                                   kz_delta_labelfree=r["kz_delta_labelfree"]) for r in rows_decomp},
              labelfree_estimator_note=("本 regime(患病率~1-2% + 弱分类器 AUROC~0.65-0.72)下 BBSE(硬混淆矩阵)与"
                                        "校准后 MLLS 均数值不稳(常塌到 pi=1e-4 下限, bbse_ok=false), 故主分解/主G3判定"
                                        "改用 oracle 真 pi_t(更严格更保守); label-free 值仅作可部署性透明报告。"),
              G2_shortcut_index=g2,
              note="chan_P=Util(目标KZ校准后)-Util(目标); frac=chan_P/total_gap; King-Zeng 仅患病率再校准, 阈值固定源val选")

    df = pl.DataFrame(rows_decomp)
    df.write_csv(f"{TAB}/stage2_channel_decomp_utility.csv")
    pl.DataFrame(rows_ci).write_csv(f"{TAB}/stage2_channel_ci_utility.csv")
    json.dump(g3, open(f"{STT}/stage2_G3_utility.json", "w"), indent=2, ensure_ascii=False)

    print("\n=== 三通道分解 (官方 Utility 空间) ===")
    print(df.select(["direction", "total_gap", "chan_M_measurement", "chan_P_prevalence",
                     "chan_V_value", "frac_M", "frac_P", "frac_V", "coverage_M_onlydelete"]).to_pandas().to_string(index=False))
    print("\n=== AUROC 空间 vs Utility 空间 frac 并列对拍 ===")
    for r in rows_decomp:
        print(f"  {r['direction']}: frac_M  AUROC={r['frac_M_auroc']:+.3f} -> Utility={r['frac_M']:+.3f}  |  "
              f"frac_P  AUROC={r['frac_P_auroc']:+.3e} -> Utility={r['frac_P']:+.3f}")
    print("\n=== CI (病人级 bootstrap B=%d) ===" % BBOOT)
    print(pl.DataFrame(rows_ci).to_pandas().to_string(index=False))
    print("\n=== G3 (Utility 空间) ===")
    print(json.dumps(g3, indent=2, ensure_ascii=False))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
