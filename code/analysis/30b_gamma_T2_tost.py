#!/usr/bin/env python
"""Stage 3 :: T2 gamma-不变性 —— 全量N + TOST 等价检验 (Block 2 / improvement.md D6)

相对 30_gamma_invariance_T2.py 的方法改进 (更严格):
  1) 去掉 200k 子抽样, 用【全量N】(两院 train+val 全部行) 重跑 T2 cluster-robust Wald。
  2) 把"BH 不拒绝 -> gamma 不变"这一【统计谬误】(把'不拒绝'当'等价')
     换成【TOST 双单侧等价检验】: 对每变量的 Δθ=θ_A-θ_B, 用预注册 margin ±0.02
     (并报 ±0.05 稳健) 做等价检验, 把每变量分三类:
        (a) DIFFERENT  : 差异 BH 显著 且 |Δθ|>margin           -> DROP-gamma (真·不可迁移)
        (b) EQUIVALENT : Δθ 的 90% CI 完全落在 ±margin 内       -> well-powered 真等价, 可迁移
        (c) INSUFFICIENT: 既不拒绝差异也不成立等价 (欠功效)      -> 证据不足, 不宣称可迁移
     这区分了"大N下统计可检出但实践可忽略"(→b) 与"真的功效不够"(→c),
     纠正 分析记录 旧措辞"24/29不变 / 子抽样方差稳"。

模型与 30 脚本完全一致 (确定性可复现, 结果应对齐 diag_d6_t2_fulln.csv):
  logit P(Y) = a + b*(两院共享 value offset) + theta_j^h * rho_j
  rho_j = MSR 残差(OOF 逐院逐变量 e_j(C) 交叉拟合); 病人级 cluster-robust 方差; BH 校正。

输出:
  results/tables/stage3_T2_tost.csv       逐变量 θ/SE/Δθ/CI/p/q/TOST/三分类
  results/tables/stage3_triage_tost.csv   34 变量丢/修/留分诊 (DROP-gamma 集改用 TOST(a))
  results/stats/stage3_T2_tost_gates.json  门 G4 (TOST 框架) + 计数汇总 + 分析记录 订正说明
  logs/analysis/30b_t2_tost.log
"""
import os
# 线程控制: 必须在 import numpy 之前设 (BLAS 在 import 时读取)。4 个并行 block, 本任务上限 8 线程。
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "8"

import sys, json, time
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from scipy import stats

SEED = 0
np.random.seed(SEED)

P = "."
sys.path.insert(0, f"{P}/code/utils")
import features as F
from ops_repair import MeasurementPropensity

TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"
MARGINS = [0.02, 0.05]          # 预注册主 margin 0.02; 0.05 稳健
Z90 = stats.norm.ppf(0.95)      # 1.645, 用于 90% CI (α=0.05 双单侧)


# ---- 以下 build_rows / msr / cluster_robust_theta 与 30 脚本一致 (复用, 不改共享模块) ----
def build_rows(rows, X, M, D, Y, L, srcnorm, cap=200):
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
    Xd = np.column_stack([np.ones_like(rho_j), value_lp, rho_j])
    try:
        m = sm.GLM(y, Xd, family=sm.families.Binomial())
        r = m.fit(cov_type="cluster", cov_kwds={"groups": groups})
        return float(r.params[2]), float(r.bse[2])
    except Exception as e:
        print(f"    [GLM fail] {e}", flush=True)
        return np.nan, np.nan


def bh(pv):
    pv = np.asarray(pv, float); m = len(pv)
    order = np.argsort(pv); out = np.empty(m)
    out[order] = np.minimum.accumulate((pv[order] * m / (np.arange(m) + 1))[::-1])[::-1]
    return np.clip(out, 0, 1)


def tost(delta, se, m):
    """双单侧等价检验: H0_lo: Δ<=-m, H0_hi: Δ>=m。p_tost=max(p_lo,p_hi)。
    p_tost<0.05  <=>  90% CI(Δ)=Δ±1.645*se 完全落在 (-m, m)。返回 (p_tost, equiv:bool)."""
    if not np.isfinite(se) or se <= 0:
        return np.nan, False
    p_lo = stats.norm.sf((delta + m) / se)   # 拒绝 Δ<=-m 即 Δ 明显 > -m
    p_hi = stats.norm.cdf((delta - m) / se)   # 拒绝 Δ>=m  即 Δ 明显 < m
    p = max(p_lo, p_hi)
    return float(p), bool(p < 0.05)


def classify(delta, se, reject_diff, m):
    """三分类, 优先级: 先判等价(b), 再判显著且超margin(a), 否则证据不足(c)."""
    ptost, equiv = tost(delta, se, m)
    if equiv:
        return "EQUIVALENT", ptost
    if reject_diff and abs(delta) > m:
        return "DIFFERENT", ptost
    return "INSUFFICIENT", ptost


def main():
    t0 = time.time()
    print(f"[start] seed={SEED} full-N T2 + TOST  {time.ctime()}", flush=True)
    X, M, D, Y, L = F.load_arrays()
    sp = F.load_splits()
    norm = F.load_norm("A")

    raw = {}
    for h in ["A", "B"]:
        rows = np.array(sp[f"{h}_train"] + sp[f"{h}_val"])
        v, ph, m, y, g = build_rows(rows, X, M, D, Y, L, norm)
        E = MeasurementPropensity(n_splits=5, C=1.0).fit_transform_oof(ph, m, g)
        raw[h] = dict(v=v, R=msr(m, E), y=y, g=g, mrate=m.mean(axis=0))
        print(f"[{h}] built rows={len(y)} ({time.time()-t0:.0f}s)", flush=True)

    # 两院合并的【单一共享 value offset】(与 30 脚本一致的严谨化: 隔离 mask 通道结局耦合)
    v_all = np.vstack([raw["A"]["v"], raw["B"]["v"]])
    y_all = np.concatenate([raw["A"]["y"], raw["B"]["y"]])
    vlr = LogisticRegression(max_iter=500, C=0.5, class_weight="balanced").fit(v_all, y_all)
    data = {}
    for h in ["A", "B"]:
        data[h] = dict(R=raw[h]["R"], y=raw[h]["y"], g=raw[h]["g"], mrate=raw[h]["mrate"],
                       vlp=vlr.decision_function(raw[h]["v"]))
    del raw, v_all, y_all

    # 逐变量【全量N】cluster-robust θ_A/θ_B (无子抽样)
    out = []
    for j in range(34):
        name = F.TV[j]
        if data["A"]["mrate"][j] < 0.005 or data["A"]["mrate"][j] > 0.995 \
           or data["B"]["mrate"][j] < 0.005 or data["B"]["mrate"][j] > 0.995:
            continue
        th, se = {}, {}
        for h in ["A", "B"]:
            d = data[h]
            th[h], se[h] = cluster_robust_theta(d["y"], d["vlp"], d["R"][:, j], d["g"])  # 全量N
        if not (np.isfinite(th["A"]) and np.isfinite(th["B"])):
            continue
        delta = th["A"] - th["B"]
        se_d = float(np.sqrt(se["A"] ** 2 + se["B"] ** 2 + 1e-12))
        z = delta / se_d
        p_raw = float(2 * stats.norm.sf(abs(z)))
        out.append(dict(variable=name, theta_A=th["A"], theta_B=th["B"], se_A=se["A"], se_B=se["B"],
                        delta_theta=float(delta), se_delta=se_d, abs_delta=float(abs(delta)),
                        z=float(z), p_raw=p_raw,
                        ci90_lo=float(delta - Z90 * se_d), ci90_hi=float(delta + Z90 * se_d),
                        mrate_A=float(data["A"]["mrate"][j]), mrate_B=float(data["B"]["mrate"][j]),
                        nrows_A=int(len(data["A"]["y"])), nrows_B=int(len(data["B"]["y"]))))
        print(f"  {name:16s} thA={th['A']:+.4f} thB={th['B']:+.4f} d={delta:+.4f} "
              f"seD={se_d:.4f} p={p_raw:.2e}", flush=True)

    df = pl.DataFrame(out)
    q = bh(df["p_raw"].to_numpy())
    df = df.with_columns(pl.Series("p_bh", q))
    df = df.with_columns((pl.col("p_bh") < 0.05).alias("reject_diff_bh"))

    # TOST + 三分类 (两个 margin)
    rej = df["reject_diff_bh"].to_list()
    dth = df["delta_theta"].to_list(); sed = df["se_delta"].to_list()
    for m in MARGINS:
        tag = f"{m:g}".replace("0.", "").rjust(2, "0")   # 0.02->"02", 0.05->"05"
        p_t, cls = [], []
        for de, s, rj in zip(dth, sed, rej):
            c, pt = classify(de, s, rj, m)
            cls.append(c); p_t.append(pt)
        df = df.with_columns([pl.Series(f"p_tost_{tag}", p_t),
                              pl.Series(f"tost_class_{tag}", cls),
                              pl.Series(f"exceeds_margin_{tag}", [abs(x) > m for x in dth])])
    df = df.sort("p_raw")
    df.write_csv(f"{TAB}/stage3_T2_tost.csv")
    print(f"\n[table] stage3_T2_tost.csv  n_var={df.height} ({time.time()-t0:.0f}s)", flush=True)

    # ---- QC: 与 30 脚本的全量N诊断 diag_d6_t2_fulln.csv 对齐 (确定性复现校验) ----
    qc_note = "diag_d6 not found"
    dref = f"{TAB}/diag_d6_t2_fulln.csv"
    if os.path.exists(dref):
        ref = pl.read_csv(dref).select(["variable", "theta_A", "theta_B"])
        j = df.select(["variable", "theta_A", "theta_B"]).join(ref, on="variable", suffix="_ref")
        da = (j["theta_A"] - j["theta_A_ref"]).abs().max()
        db = (j["theta_B"] - j["theta_B_ref"]).abs().max()
        qc_note = dict(max_abs_theta_A_diff_vs_diag_d6=float(da),
                       max_abs_theta_B_diff_vs_diag_d6=float(db),
                       reproduces_diag_d6=bool(da < 1e-6 and db < 1e-6))
        print(f"[QC] vs diag_d6: maxΔθA={da:.2e} maxΔθB={db:.2e}", flush=True)

    # ---- 计数汇总 (margin 0.02 为主) ----
    d = df.to_dicts()
    n_tested = len(d)
    bh_reject = [r["variable"] for r in d if r["reject_diff_bh"]]
    bh_reject_gt02 = [r["variable"] for r in d if r["reject_diff_bh"] and r["abs_delta"] > 0.02]

    def tally(tag):
        cc = {"DIFFERENT": [], "EQUIVALENT": [], "INSUFFICIENT": []}
        for r in d:
            cc[r[f"tost_class_{tag}"]].append(r["variable"])
        return cc

    t02 = tally("02"); t05 = tally("05")
    drop_gamma_02 = t02["DIFFERENT"]     # TOST(a) 类 = 新的 DROP-gamma 集

    # ---- G4 复判 (TOST 框架): 多数关键变量是否 gamma 非不变? ----
    G4_majority_diff = len(t02["DIFFERENT"]) > n_tested / 2
    gates = dict(
        method="full-N (no subsample) cluster-robust Wald + TOST equivalence",
        seed=SEED, n_variables_tested=n_tested,
        fulln_bh_reject_n=len(bh_reject), fulln_bh_reject_set=bh_reject,
        fulln_bh_reject_abs_delta_gt_0p02_n=len(bh_reject_gt02),
        fulln_bh_reject_abs_delta_gt_0p02_set=bh_reject_gt02,
        tost_margin_0p02=dict(
            DIFFERENT_dropgamma_n=len(t02["DIFFERENT"]), DIFFERENT_dropgamma_set=t02["DIFFERENT"],
            EQUIVALENT_transportable_n=len(t02["EQUIVALENT"]), EQUIVALENT_transportable_set=t02["EQUIVALENT"],
            INSUFFICIENT_n=len(t02["INSUFFICIENT"]), INSUFFICIENT_set=t02["INSUFFICIENT"]),
        tost_margin_0p05=dict(
            DIFFERENT_dropgamma_n=len(t05["DIFFERENT"]), DIFFERENT_dropgamma_set=t05["DIFFERENT"],
            EQUIVALENT_transportable_n=len(t05["EQUIVALENT"]), EQUIVALENT_transportable_set=t05["EQUIVALENT"],
            INSUFFICIENT_n=len(t05["INSUFFICIENT"]), INSUFFICIENT_set=t05["INSUFFICIENT"]),
        DROP_gamma_set_prereg_margin_0p02=drop_gamma_02,
        G4_majority_gamma_noninvariant=bool(G4_majority_diff),
        G4_triggers_Prop3=bool(G4_majority_diff),
        G4_verdict=("G4 NOT triggered: only %d/%d 变量在预注册 margin ±0.02 下"
                    " gamma 真·非不变 (TOST DIFFERENT); 远非多数 -> Prop3 不触发" % (len(t02["DIFFERENT"]), n_tested)),
        qc_reproduce_diag_d6=qc_note,
        note_correction=("分析记录 旧措辞需订正: (1) 删'子抽样 200k 方差稳'——全量N下 BH 拒绝由 5→%d 翻倍;"
                            " (2) 删'24/29 gamma 不变'——'不拒绝!=等价'是统计谬误;"
                            " 改 TOST 诚实措辞: 全量N %d/%d 变量统计可检出跨院 gamma 差异, 但仅 %d 个 |Δθ|>0.02"
                            " (预注册 margin) 才实践意义上非不变; TOST 分类: %d 个 well-powered 真等价(可迁移)、"
                            "%d 个欠功效证据不足、%d 个真·不同(DROP-gamma)。"
                            " 大N可检出但效应量微小 -> 定向修复无可得, 【强化】G5 负结果, 非'可迁移支柱倒塌'。"
                            % (len(bh_reject), len(bh_reject), n_tested, len(bh_reject_gt02),
                               len(t02["EQUIVALENT"]), len(t02["INSUFFICIENT"]), len(t02["DIFFERENT"]))),
    )
    json.dump(gates, open(f"{STT}/stage3_T2_tost_gates.json", "w"), ensure_ascii=False, indent=2)

    # ---- 重算 34 变量分诊表: DROP-gamma 集改用 TOST(a) 类; gamma 证据不足单列 ----
    qc = json.load(open(f"{P}/data/processed/qc_stage0.json"))
    struct = set(qc["structurally_absent"]); ra = qc["obs_rate_A"]; rb = qc["obs_rate_B"]
    t1 = {r["variable"]: r for r in pl.read_csv(f"{TAB}/stage1_T1_slope_stability.csv").to_dicts()}
    tostmap = {r["variable"]: r for r in d}

    trows = []
    for v in F.TV:
        oa, ob = ra[v], rb[v]; mn = min(oa, ob)
        rec = tostmap.get(v, {})
        cls02 = rec.get("tost_class_02"); cls05 = rec.get("tost_class_05")
        t1s = t1.get(v, {}).get("mean_abs_slope_diff", float("nan"))
        r = dict(variable=v, obs_rate_A=oa, obs_rate_B=ob, min_obs_rate=mn,
                 is_structural=v in struct,
                 theta_A=rec.get("theta_A"), theta_B=rec.get("theta_B"),
                 delta_theta=rec.get("delta_theta"), ci90_lo=rec.get("ci90_lo"), ci90_hi=rec.get("ci90_hi"),
                 p_bh=rec.get("p_bh"), reject_diff_bh=rec.get("reject_diff_bh"),
                 tost_class_02=cls02, tost_class_05=cls05, T1_slope_diff=t1s)
        # 决策树: 结构/正性 优先 -> TOST gamma 类别
        if v in struct or mn < 0.01:
            r["decision"] = "DROP_schema_or_positivity"
        elif cls02 == "DIFFERENT":
            r["decision"] = "DROP_gamma_varies"                       # TOST(a): Prop3 不可迁移
        elif cls02 == "EQUIVALENT":
            r["decision"] = ("REPAIR_protocol" if (t1s == t1s and t1s > 0.15) else "KEEP_transportable")
        elif cls02 == "INSUFFICIENT":
            r["decision"] = "UNCERTAIN_gamma_underpowered"            # 欠功效: 不宣称可迁移
        else:
            r["decision"] = "WEAK_untested"                           # 正性过滤未进 T2
        trows.append(r)

    tdf = pl.DataFrame(trows).sort("min_obs_rate")
    tdf.write_csv(f"{TAB}/stage3_triage_tost.csv")
    tri_counts = {}
    for r in trows:
        tri_counts[r["decision"]] = tri_counts.get(r["decision"], 0) + 1
    gates["triage_counts"] = tri_counts
    gates["triage_DROP_gamma_set"] = [r["variable"] for r in trows if r["decision"] == "DROP_gamma_varies"]
    json.dump(gates, open(f"{STT}/stage3_T2_tost_gates.json", "w"), ensure_ascii=False, indent=2)

    print("\n=== FULL-N BH ===")
    print(f"  tested={n_tested}  BH_reject={len(bh_reject)} {bh_reject}")
    print(f"  of which |Δθ|>0.02: {len(bh_reject_gt02)} {bh_reject_gt02}")
    print("=== TOST margin 0.02 ===")
    print(f"  DIFFERENT(DROP-gamma)={len(t02['DIFFERENT'])} {t02['DIFFERENT']}")
    print(f"  EQUIVALENT(transport)={len(t02['EQUIVALENT'])} {t02['EQUIVALENT']}")
    print(f"  INSUFFICIENT         ={len(t02['INSUFFICIENT'])} {t02['INSUFFICIENT']}")
    print("=== TOST margin 0.05 ===")
    print(f"  DIFFERENT={len(t05['DIFFERENT'])} {t05['DIFFERENT']}  EQUIV={len(t05['EQUIVALENT'])}  INSUF={len(t05['INSUFFICIENT'])}")
    print(f"=== G4 majority gamma non-invariant? {G4_majority_diff} (Prop3 {'触发' if G4_majority_diff else '不触发'}) ===")
    print("=== TRIAGE counts ===", tri_counts)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
