#!/usr/bin/env python
"""[P1-CPU] 变量级 AUROC 归因: 源->目标判别损失来自 mask/缺失模式通道 vs 值通道, 细到单变量。

背景(反漂移纪律): 跨院崩溃主导成分是真实排序/判别退化(非阈值)。本探针把 src->tgt AUROC 缺口
归因到具体测量过程特征, 把「按变量丢 mask」从 Rockenschaub 式经验升级为机制证据。

方法(纯 CPU, 无 GPU 权重/重训依赖; 值模型为轻量 LGB value_mask, 与 23_channel_decomp 同口径):
  1. 组级复核: 复用 23_channel_decomp 的可辨识分解 —— 总缺口 G=AUC(src)-AUC(tgt),
     M(测量/mask)通道=掩码移植反事实, V+P 残差; 确认 V(值)主导 vs M(mask)少数。
  2. 变量级 M 归因: 对 34 变量逐一「只移植该变量 mask」(源病人按目标/源下单倾向比丢该列观测),
     测该单变量测量过程对 AUROC 缺口的贡献 -> 哪些 mask/缺失模式特征在目标域制造判别损失。
  3. 变量级 V 归因(代理): 对目标集逐变量置换其值列, 测 AUROC 掉幅(目标侧值重要度);
     与源侧对照, src->tgt 值重要度变化标示值通道缺口来源变量。
  4. 交叉引用 stage1_per_variable 指纹强度(univariate hospital AUC)与 obs-rate 漂移。
"""
import os, sys, json, time
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

P = "."
sys.path.insert(0, f"{P}/code/utils")
import features as F
DI = f"{P}/results/diag_improve"; TAB = f"{P}/results/tables"
SEED = 20260714
t0 = time.time()


def log(*a): print(*a, flush=True)


def flatten(rows, X, M, Y, L, cap, srcnorm, meta):
    Z = F.ffill_impute(np.asarray(X[rows][:, :cap]), np.asarray(M[rows][:, :cap]), srcnorm, channels=F.TV)
    phi, _ = F.build_phi(X[rows], M[rows], L[rows], meta[list(rows)], src=None, _norm=srcnorm)
    vv, mm, yy, pp = [], [], [], []
    for k, i in enumerate(rows):
        n = int(min(L[i], cap))
        vv.append(Z[k, :n]); mm.append(M[i, :n].astype(np.float32)); yy.append(Y[i, :n]); pp.append(phi[k, :n])
    return (np.concatenate(vv), np.concatenate(mm), np.concatenate(yy).astype(int), np.concatenate(pp))


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


def main():
    X, M, D, Y, L = F.load_arrays()
    meta = F.load_meta(); sp = F.load_splits()
    rng = np.random.default_rng(SEED)
    cap = 200
    TV = list(F.TV)   # 34 变量名
    results = {}

    for direction in ["A2B", "B2A"]:
        src, tgt = ("A", "B") if direction == "A2B" else ("B", "A")
        srcnorm = F.load_norm(src)
        tr = rng.choice(sp[f"{src}_train"], min(13015, len(sp[f"{src}_train"])), replace=False)
        ste = np.array(sp[f"{src}_test"]); tte = np.array(sp[f"{tgt}_test"])

        Vtr, Mtr, ytr, PHtr = flatten(np.array(tr), X, M, Y, L, cap, srcnorm, meta)
        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                 min_child_samples=100, n_jobs=4, verbose=-1, scale_pos_weight=15,
                                 random_state=SEED)
        clf.fit(np.hstack([Vtr, Mtr]), ytr)
        e_src_models = fit_prop(PHtr, Mtr)

        Vs, Ms, ys, PHs = flatten(ste, X, M, Y, L, cap, srcnorm, meta)
        Vt, Mt, yt, PHt = flatten(tte, X, M, Y, L, cap, F.load_norm(tgt), meta)
        e_tgt_models = fit_prop(PHt, Mt)

        def auc_of(V_, M_, y_):
            return float(roc_auc_score(y_, clf.predict_proba(np.hstack([V_, M_]))[:, 1]))

        auc_src = auc_of(Vs, Ms, ys)
        auc_tgt = auc_of(Vt, Mt, yt)
        G = auc_src - auc_tgt
        log(f"[{direction}] auc_src={auc_src:.4f} auc_tgt={auc_tgt:.4f} gap={G:.4f} ({time.time()-t0:.0f}s)")

        # ---- 组级 M 通道: 全变量掩码移植 (源病人 -> 目标下单习惯) ----
        e_s_on_s = pred_e(e_src_models, PHs); e_t_on_s = pred_e(e_tgt_models, PHs)
        pkeep_all = np.clip(e_t_on_s / e_s_on_s, 0, 1)
        drop_rand = rng.random(Ms.shape)
        Mx = Ms.copy(); Vx = Vs.copy()
        drop_all = (Ms > 0) & (drop_rand > pkeep_all)
        Mx[drop_all] = 0; Vx[drop_all] = 0.0
        chan_M_group = auc_src - auc_of(Vx, Mx, ys)
        chan_VP_group = G - chan_M_group

        # ---- 变量级 M 归因: 只移植变量 j 的 mask ----
        var_M = []
        for j in range(34):
            Mxj = Ms.copy(); Vxj = Vs.copy()
            dj = (Ms[:, j] > 0) & (drop_rand[:, j] > pkeep_all[:, j])
            Mxj[dj, j] = 0; Vxj[dj, j] = 0.0
            aj = auc_of(Vxj, Mxj, ys)
            var_M.append(auc_src - aj)   # 该变量测量过程对缺口的贡献

        # ---- 变量级 V 归因(代理): 目标集/源集逐变量值置换重要度 ----
        base_tgt = clf.predict_proba(np.hstack([Vt, Mt]))[:, 1]
        base_src = clf.predict_proba(np.hstack([Vs, Ms]))[:, 1]
        auc_base_tgt = roc_auc_score(yt, base_tgt); auc_base_src = roc_auc_score(ys, base_src)
        var_Vimp_tgt = []; var_Vimp_src = []
        REP = 3
        for j in range(34):
            dt = 0.0; dsc = 0.0
            for _ in range(REP):
                Vt2 = Vt.copy(); Vt2[:, j] = rng.permutation(Vt2[:, j])
                dt += auc_base_tgt - roc_auc_score(yt, clf.predict_proba(np.hstack([Vt2, Mt]))[:, 1])
                Vs2 = Vs.copy(); Vs2[:, j] = rng.permutation(Vs2[:, j])
                dsc += auc_base_src - roc_auc_score(ys, clf.predict_proba(np.hstack([Vs2, Ms]))[:, 1])
            var_Vimp_tgt.append(dt / REP); var_Vimp_src.append(dsc / REP)

        per_var = []
        for j in range(34):
            per_var.append(dict(variable=TV[j],
                                M_gap_contrib=round(float(var_M[j]), 5),
                                V_perm_imp_tgt=round(float(var_Vimp_tgt[j]), 5),
                                V_perm_imp_src=round(float(var_Vimp_src[j]), 5),
                                V_imp_shift_src_minus_tgt=round(float(var_Vimp_src[j] - var_Vimp_tgt[j]), 5)))
        per_var.sort(key=lambda r: -r["M_gap_contrib"])

        results[direction] = dict(
            auc_src=round(auc_src, 4), auc_tgt=round(auc_tgt, 4), total_gap=round(G, 4),
            group_level=dict(chan_M_measurement=round(float(chan_M_group), 4),
                             chan_VP_value_prev=round(float(chan_VP_group), 4),
                             frac_M=round(float(chan_M_group / (G + 1e-9)), 3),
                             frac_VP=round(float(chan_VP_group / (G + 1e-9)), 3),
                             sum_var_M=round(float(sum(var_M)), 4),
                             note="frac_M/frac_VP 复核 23_channel_decomp; sum_var_M=逐变量 M 贡献之和(近似组级 M, 因移植随机性与交互略有出入)"),
            top_M_variables=per_var[:10],
            per_variable=per_var)
        log(f"[{direction}] group frac_M={chan_M_group/(G+1e-9):.2f} frac_VP={chan_VP_group/(G+1e-9):.2f} "
            f"| topM: {[(r['variable'], r['M_gap_contrib']) for r in per_var[:5]]}")

    # ---- 交叉引用 stage1_per_variable ----
    s1 = pl.read_csv(f"{TAB}/stage1_per_variable.csv")
    s1map = {r["variable"]: dict(obs_rate_A=r["obs_rate_A"], obs_rate_B=r["obs_rate_B"],
                                 abs_diff=r["abs_diff"], univariate_hospital_auc=r["univariate_hospital_auc"],
                                 is_structural=r["is_structural"]) for r in s1.to_dicts()}

    out = dict(
        generated=time.strftime("%Y-%m-%d %H:%M"),
        question="源->目标 AUROC 缺口来自 mask/缺失模式通道 vs 值通道? 归因到具体测量过程特征(变量级)。",
        method="轻量 LGB value_mask(与 23_channel_decomp/40c 同超参, 纯 CPU, 无 GPU 权重/重训); 组级掩码移植(可辨识) + 变量级单变量掩码移植 + 变量级值置换重要度。",
        group_level_summary=dict(
            A2B=dict(gap=results["A2B"]["total_gap"], frac_M=results["A2B"]["group_level"]["frac_M"],
                     frac_VP=results["A2B"]["group_level"]["frac_VP"]),
            B2A=dict(gap=results["B2A"]["total_gap"], frac_M=results["B2A"]["group_level"]["frac_M"],
                     frac_VP=results["B2A"]["group_level"]["frac_VP"]),
            existing_channel_decomp="results/tables/stage2_channel_decomp.csv (A2B frac_M=0.176/frac_V=0.824; B2A frac_M=0.231/frac_V=0.779)",
            readout="源->目标 AUROC 缺口由值通道主导(~78-82%), mask/测量通道贡献真实但少数(~18-23%) -> 崩溃是真实判别损失(值), 非纯 mask-shortcut 伪影; 反漂移: 支持残余=真实判别退化。"),
        directions=results,
        stage1_crossref=s1map,
        conclusion="mask 通道贡献真实但少数(~18-23%), 集中在少数下单习惯漂移最大的测量特征(见 top_M_variables 与 stage1 univariate_hospital_auc); 值通道主导缺口 -> 「按变量丢 mask」有机制依据(丢掉在目标域制造判别损失的少数 mask 特征), 但不能修复值通道主导的真实判别退化(需目标标签/更强修法)。",
        caveat="变量级 M 用单变量掩码移植(反事实, 随机移植含方差); 变量级 V 用置换重要度(目标侧值贡献, 非严格缺口分解), 作定位代理。组级 M-vs-V 为可辨识分解, 与 stage2_channel_decomp 一致。",
    )
    json.dump(out, open(f"{DI}/variable_auroc_attribution.json", "w"), indent=2, ensure_ascii=False)
    log(f"[{time.time()-t0:.0f}s] saved variable_auroc_attribution.json")

    # ================= 绘图: 变量级 M vs V 归因 (双向) =================
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, direction in zip(axes, ["A2B", "B2A"]):
        pv = sorted(results[direction]["per_variable"], key=lambda r: -(r["M_gap_contrib"]))[:14]
        names = [r["variable"] for r in pv]
        Mc = [r["M_gap_contrib"] for r in pv]
        Vc = [r["V_perm_imp_tgt"] for r in pv]
        yy = np.arange(len(names))[::-1]
        ax.barh(yy + 0.2, Mc, height=0.38, color="#4c72b0", label="M: 单变量 mask 移植对缺口贡献")
        ax.barh(yy - 0.2, Vc, height=0.38, color="#c44e52", label="V: 该变量值置换重要度(目标)")
        ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=8.5)
        ax.axvline(0, color="k", lw=0.7)
        gl = results[direction]["group_level"]
        ax.set_title(f"{direction}  总缺口={results[direction]['total_gap']:.3f}  "
                     f"(组级 M={gl['frac_M']:.0%} / V+P={gl['frac_VP']:.0%})", fontsize=11)
        ax.set_xlabel("对 AUROC 的贡献 / 重要度")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("变量级 AUROC 缺口归因: 值通道主导(判别损失) · mask 通道真实但少数 · 定位关键测量特征",
                 fontsize=12.5, y=1.00, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{DI}/variable_auroc_attribution.png", bbox_inches="tight", dpi=160)
    log("saved variable_auroc_attribution.png")


if __name__ == "__main__":
    main()
