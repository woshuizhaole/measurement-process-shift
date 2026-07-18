#!/usr/bin/env python
"""Block 3 :: 严谨主终点 (LightGBM backbone) — R4/OPS-Full 干净隔离 + count 崩溃 (D7)

基于 diag_40_r4.py, 修全部预注册缺陷 (方案 §9.1/§9.2/§8.2/§8.3):
  1. G5 门与 CI 的基线从 value_only 升级为 **max(value_only, value_mask)** (全量下 value_mask 更强,
     value_only 是"最弱基线", 方案 §9.1 要求超"最优笨基线")。
  2. 配对 CI 覆盖 **Utility** (util_boot.paired_bootstrap_util_diff, B>=2000, 病人级) 与 AUROC
     (boot.paired_bootstrap_diff), 且对 **value_only 与 value_mask 两个基线** 都算。
  3. R4_ipw / OPS_full 都进配对 Utility CI (原 diag_r4 R4_ipw 完全没算 CI)。
  4. **>=10 种子** headline; CI 用"种子集成"(目标域概率跨种子平均, 源 val 集成选阈)。
  5. 新增 count_only (8h 回看观测计数, 34) 与 value_count (值+count) 两个特征模式做 D7 count 崩溃。
  6. §9.2 附加院内不退化门: 方法 src_util >= value_mask src_util - 0.01。

阈值: 每方法源域 val 选 (best_threshold)。官方 Utility + AUROC + 校准斜率永远并列 (§8.1)。
所有倾向 e_j(C) 用 build_phi(仅永远可得通道) 交叉拟合(源) / 全量拟合(目标)。绝不改共享模块。
"""
import os, sys, json, time, argparse
import numpy as np, polars as pl
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

P = "."
sys.path.insert(0, f"{P}/code/utils")
import features as F
from ops_repair import MeasurementPropensity, msr, reanchor
from models import normalized_utility, best_threshold
from boot import paired_bootstrap_diff, patient_bootstrap_ci
from util_boot import paired_bootstrap_util_diff, bootstrap_util_ci

TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"

METHODS = ["value_only", "value_mask", "mask_only", "R1_reanchor", "R2_msr",
           "R4_ipw", "OPS_full", "count_only", "value_count"]
# 特征构造 (行级设计矩阵); 参数固定为 (V, Mk=raw mask, R, Dl=log1p(delta), C=8h count)
FEAT_FNS = {
    "value_only":  lambda V, Mk, R, Dl, C: V,
    "value_mask":  lambda V, Mk, R, Dl, C: np.hstack([V, Mk, Dl]),
    "mask_only":   lambda V, Mk, R, Dl, C: np.hstack([Mk, Dl]),
    "R1_reanchor": lambda V, Mk, R, Dl, C: np.hstack([V, R, Dl]),   # R = 协议重锚 mask
    "R2_msr":      lambda V, Mk, R, Dl, C: np.hstack([V, R, Dl]),   # R = MSR 残差
    "R4_ipw":      lambda V, Mk, R, Dl, C: np.hstack([V, Mk, Dl]),  # V 已被 IPW 条件插补替换
    "OPS_full":    lambda V, Mk, R, Dl, C: np.hstack([V, R, Dl]),   # V=IPW, R=MSR
    "count_only":  lambda V, Mk, R, Dl, C: C,                        # D7: 8h 观测计数单通道
    "value_count": lambda V, Mk, R, Dl, C: np.hstack([V, C]),       # D7: 值 + count
}


def calib_slope(y, p):
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    try:
        r = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
        return float(r.params[1])
    except Exception:
        return np.nan


def build(rows, X, M, D, Y, L, srcnorm, cap, meta):
    """返回 (val, msk, dlt, cnt, ph, yy, gg) 行级拼接。cnt=8h 回看观测计数 (因果, running)。"""
    Z = F.ffill_impute(np.asarray(X[rows][:, :cap]), np.asarray(M[rows][:, :cap]), srcnorm, channels=F.TV)
    phi, _ = F.build_phi(X[rows], M[rows], L[rows], meta[list(map(int, rows))], src=None, _norm=srcnorm)
    val, msk, dlt, cnt, ph, yy, gg = [], [], [], [], [], [], []
    for k, i in enumerate(rows):
        n = int(min(L[i], cap))
        m = np.asarray(M[i, :n]).astype(np.float32)
        val.append(Z[k, :n]); msk.append(m)
        dlt.append(np.log1p(np.asarray(D[i, :n]))); ph.append(phi[k, :n])
        c = np.zeros((n, 34), dtype=np.float32)          # 8h 回看观测计数 (口径同 20b/diag_d7)
        for t in range(n):
            c[t] = m[max(0, t - 7):t + 1].sum(axis=0)
        cnt.append(c)
        yy.append(np.asarray(Y[i, :n])); gg.append(np.full(n, i))
    return (np.concatenate(val), np.concatenate(msk), np.concatenate(dlt),
            np.concatenate(cnt), np.concatenate(ph), np.concatenate(yy).astype(int), np.concatenate(gg))


def ipw_impute_fit(val, msk, ph, e_src):
    models = []
    for j in range(34):
        obs = msk[:, j] > 0
        if obs.sum() < 100:
            models.append(None); continue
        w = 1.0 / np.clip(e_src[obs, j], 0.05, 1)
        m = Ridge(alpha=1.0).fit(ph[obs], val[obs, j], sample_weight=w)
        models.append(m)
    return models


def ipw_impute_apply(val, msk, ph, models):
    V = val.copy()
    for j in range(34):
        if models[j] is None:
            continue
        miss = msk[:, j] == 0
        V[miss, j] = models[j].predict(ph[miss])
    return V


def patient_index(g):
    """返回 (uniq, idx_list): idx_list[k] = 属于第 k 个病人(按 uniq 顺序) 的行索引数组。"""
    uniq, inv = np.unique(g, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    sinv = inv[order]
    bnd = np.searchsorted(sinv, np.arange(len(uniq) + 1))
    return uniq, [order[bnd[c]:bnd[c + 1]] for c in range(len(uniq))]


def to_lists(v, idx_list):
    return [v[ix] for ix in idx_list]


def main(args):
    t0 = time.time()
    X, M, D, Y, L = F.load_arrays()
    meta = F.load_meta(); sp = F.load_splits()
    cap = args.tcap
    rows_out = []          # 每方法×方向×种子
    ens_rows = []          # 每方法×方向 种子集成指标
    ci_rows = []           # 配对 CI (Utility+AUROC, vs value_only 与 value_mask)
    count_rows = []        # D7 count 崩溃

    for direction in ["A2B", "B2A"]:
        src, tgt = ("A", "B") if direction == "A2B" else ("B", "A")
        srcnorm = F.load_norm(src)
        tr = np.array(sp[f"{src}_train"]); va = np.array(sp[f"{src}_val"])
        ste = np.array(sp[f"{src}_test"]); tte = np.array(sp[f"{tgt}_test"])
        if args.nfit and len(tr) > args.nfit:          # 仅用于 smoke test; 默认 0=全量(=ledger 口径)
            tr = np.random.default_rng(20260712).choice(tr, args.nfit, replace=False)

        Vtr, Mtr, Dtr, Ctr, PHtr, ytr, gtr = build(tr, X, M, D, Y, L, srcnorm, cap, meta)
        Vva, Mva, Dva, Cva, PHva, yva, gva = build(va, X, M, D, Y, L, srcnorm, cap, meta)
        Vst, Mst, Dst, Cst, PHst, yst, gst = build(ste, X, M, D, Y, L, srcnorm, cap, meta)
        Vtt, Mtt, Dtt, Ctt, PHtt, ytt, gtt = build(tte, X, M, D, Y, L, srcnorm, cap, meta)
        print(f"[{direction}] built rows tr={len(ytr)} va={len(yva)} st={len(yst)} tt={len(ytt)} "
              f"({time.time()-t0:.0f}s)", flush=True)

        # 病人分组索引 (每集一次)
        _, ixva = patient_index(gva); yva_L = to_lists(yva, ixva)
        _, ixst = patient_index(gst); yst_L = to_lists(yst, ixst)
        _, ixtt = patient_index(gtt); ytt_L = to_lists(ytt, ixtt)

        # 倾向 e_j(C)
        mp_src = MeasurementPropensity()
        e_tr = mp_src.fit_transform_oof(PHtr, Mtr, gtr)
        mp_src.fit(PHtr, Mtr)
        e_va = mp_src.predict_e(PHva); e_st = mp_src.predict_e(PHst)
        mp_tgt = MeasurementPropensity().fit(PHtt, Mtt)
        e_tt_h = mp_tgt.predict_e(PHtt)
        e_tt_src = mp_src.predict_e(PHtt)
        # MSR 残差
        R_tr = msr(Mtr, e_tr); R_va = msr(Mva, e_va); R_st = msr(Mst, e_st); R_tt = msr(Mtt, e_tt_h)
        # 协议重锚
        RA_tr = reanchor(Mtr, e_tr, mp_src.predict_e(PHtr))
        RA_va = reanchor(Mva, e_va, e_va); RA_st = reanchor(Mst, e_st, e_st)
        RA_tt = reanchor(Mtt, e_tt_h, e_tt_src)
        # IPW 条件插补 (源拟合, 两域套用)
        ipw_models = ipw_impute_fit(Vtr, Mtr, PHtr, e_tr)
        Vtr_ipw = ipw_impute_apply(Vtr, Mtr, PHtr, ipw_models)
        Vva_ipw = ipw_impute_apply(Vva, Mva, PHva, ipw_models)
        Vst_ipw = ipw_impute_apply(Vst, Mst, PHst, ipw_models)
        Vtt_ipw = ipw_impute_apply(Vtt, Mtt, PHtt, ipw_models)
        print(f"[{direction}] propensity+IPW done ({time.time()-t0:.0f}s)", flush=True)

        def method_data(name):
            if name == "R1_reanchor":
                return dict(Vtr=Vtr, Rtr=RA_tr, Vva=Vva, Rva=RA_va, Vst=Vst, Rst=RA_st, Vtt=Vtt, Rtt=RA_tt)
            if name == "R2_msr":
                return dict(Vtr=Vtr, Rtr=R_tr, Vva=Vva, Rva=R_va, Vst=Vst, Rst=R_st, Vtt=Vtt, Rtt=R_tt)
            if name == "R4_ipw":
                return dict(Vtr=Vtr_ipw, Rtr=Mtr, Vva=Vva_ipw, Rva=Mva, Vst=Vst_ipw, Rst=Mst, Vtt=Vtt_ipw, Rtt=Mtt)
            if name == "OPS_full":
                return dict(Vtr=Vtr_ipw, Rtr=R_tr, Vva=Vva_ipw, Rva=R_va, Vst=Vst_ipw, Rst=R_st, Vtt=Vtt_ipw, Rtt=R_tt)
            return dict(Vtr=Vtr, Rtr=Mtr, Vva=Vva, Rva=Mva, Vst=Vst, Rst=Mst, Vtt=Vtt, Rtt=Mtt)

        ens_tt = {}          # method -> 集成目标域行级概率
        ens_thr = {}         # method -> 集成阈值 (源 val 选)
        ens_metrics = {}     # method -> ensemble 指标 dict

        for name in METHODS:
            fn = FEAT_FNS[name]; md = method_data(name)
            Xtr = fn(md["Vtr"], Mtr, md["Rtr"], Dtr, Ctr)
            Xva = fn(md["Vva"], Mva, md["Rva"], Dva, Cva)
            Xst = fn(md["Vst"], Mst, md["Rst"], Dst, Cst)
            Xtt = fn(md["Vtt"], Mtt, md["Rtt"], Dtt, Ctt)
            sum_va = np.zeros(len(yva)); sum_st = np.zeros(len(yst)); sum_tt = np.zeros(len(ytt))
            for seed in range(args.seeds):
                clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                         min_child_samples=100, subsample=0.8, subsample_freq=1,
                                         colsample_bytree=0.8, n_jobs=args.jobs, verbose=-1,
                                         scale_pos_weight=15, random_state=seed)
                clf.fit(Xtr, ytr)
                pva = clf.predict_proba(Xva)[:, 1]
                pst = clf.predict_proba(Xst)[:, 1]
                ptt = clf.predict_proba(Xtt)[:, 1]
                sum_va += pva; sum_st += pst; sum_tt += ptt
                thr_s, _ = best_threshold(yva_L, to_lists(pva, ixva))
                rec = dict(direction=direction, method=name, seed=seed, thr=round(thr_s, 4),
                           src_auroc=roc_auc_score(yst, pst), tgt_auroc=roc_auc_score(ytt, ptt),
                           src_util=normalized_utility(yst_L, to_lists(pst, ixst), thr_s),
                           tgt_util=normalized_utility(ytt_L, to_lists(ptt, ixtt), thr_s),
                           tgt_calib_slope=calib_slope(ytt, ptt))
                rows_out.append(rec)
                print(f"  {direction} {name:12s} s{seed} src_u={rec['src_util']:.3f} "
                      f"tgt_u={rec['tgt_util']:.3f} tgt_auroc={rec['tgt_auroc']:.3f} "
                      f"calib={rec['tgt_calib_slope']:.2f} ({time.time()-t0:.0f}s)", flush=True)
            eva = sum_va / args.seeds; est = sum_st / args.seeds; ett = sum_tt / args.seeds
            thr_e, _ = best_threshold(yva_L, to_lists(eva, ixva))
            ens_tt[name] = ett; ens_thr[name] = thr_e
            emet = dict(direction=direction, method=name, ens_thr=round(thr_e, 4),
                        ens_src_auroc=roc_auc_score(yst, est), ens_tgt_auroc=roc_auc_score(ytt, ett),
                        ens_src_util=normalized_utility(yst_L, to_lists(est, ixst), thr_e),
                        ens_tgt_util=normalized_utility(ytt_L, to_lists(ett, ixtt), thr_e),
                        ens_tgt_calib=calib_slope(ytt, ett))
            ens_metrics[name] = emet; ens_rows.append(emet)
            print(f"  [{direction}] ENSEMBLE {name:12s} src_u={emet['ens_src_util']:.3f} "
                  f"tgt_u={emet['ens_tgt_util']:.3f} tgt_auroc={emet['ens_tgt_auroc']:.3f}", flush=True)

        # ---- 配对 CI: 每方法 vs value_only 与 value_mask (Utility + AUROC), 用种子集成预测 ----
        ptt_tt_lists = {m: to_lists(ens_tt[m], ixtt) for m in METHODS}
        for name in METHODS:
            if name == "value_only":
                continue
            for base in ["value_only", "value_mask"]:
                if name == base:
                    continue
                # Utility 配对 (method - base); paired_util_diff 算 Util(B)-Util(A), 传 A=base,B=method
                u = paired_bootstrap_util_diff(ytt_L, ptt_tt_lists[base], ptt_tt_lists[name],
                                               ens_thr[base], ens_thr[name], B=args.B, seed=0)
                ci_rows.append(dict(direction=direction, method=name, vs=base, metric="utility",
                                    delta=round(u["point"], 4), lo=round(u["lo"], 4), hi=round(u["hi"], 4),
                                    p_gt0=round(u["p_gt0"], 3), n_boot=u["n_boot"], n_pat=u["n_patients"]))
                # AUROC 配对 (method - base)
                a = paired_bootstrap_diff(ytt, ens_tt[base], ens_tt[name], gtt, "auroc", B=args.B, seed=0)
                ci_rows.append(dict(direction=direction, method=name, vs=base, metric="auroc",
                                    delta=round(a["point"], 4), lo=round(a["lo"], 4), hi=round(a["hi"], 4),
                                    p_gt0=round(a["p_gt0"], 3), n_boot=a["n_boot"], n_pat=len(ixtt)))
            print(f"  [{direction}] CI done: {name}", flush=True)

        # ---- D7 count 崩溃: count_only/value_count/mask_only/value_mask 绝对 CI ----
        for feat in ["count_only", "value_count", "mask_only", "value_mask", "value_only"]:
            aci = patient_bootstrap_ci(ytt, ens_tt[feat], gtt, "auroc", B=args.B, seed=0)
            uci = bootstrap_util_ci(ytt_L, ptt_tt_lists[feat], ens_thr[feat], B=args.B, seed=0)
            em = ens_metrics[feat]
            count_rows.append(dict(direction=direction, feat=feat,
                                   ens_src_auroc=round(em["ens_src_auroc"], 4),
                                   ens_tgt_auroc=round(em["ens_tgt_auroc"], 4),
                                   tgt_auroc_lo=round(aci["lo"], 4), tgt_auroc_hi=round(aci["hi"], 4),
                                   ens_src_util=round(em["ens_src_util"], 4),
                                   ens_tgt_util=round(em["ens_tgt_util"], 4),
                                   tgt_util_lo=round(uci["lo"], 4), tgt_util_hi=round(uci["hi"], 4)))
        print(f"[{direction}] count collapse CI done ({time.time()-t0:.0f}s)", flush=True)

    # ================= 落盘 =================
    df = pl.DataFrame(rows_out); df.write_csv(f"{TAB}/stage4_ops_lgb_main.csv")
    ci_df = pl.DataFrame(ci_rows); ci_df.write_csv(f"{TAB}/stage4_ops_lgb_ci.csv")
    ens_df = pl.DataFrame(ens_rows)

    # 每方法×方向: 种子的 mean/median/std
    stat_cols = ["src_auroc", "tgt_auroc", "src_util", "tgt_util", "tgt_calib_slope"]
    agg = (df.group_by(["direction", "method"])
           .agg([pl.col(c).mean().alias(f"{c}_mean") for c in stat_cols]
                + [pl.col(c).median().alias(f"{c}_median") for c in stat_cols]
                + [pl.col(c).std().alias(f"{c}_std") for c in stat_cols]))
    agg = agg.join(ens_df, on=["direction", "method"], how="left").sort(["direction", "method"])
    agg.write_csv(f"{TAB}/stage4_ops_lgb_agg.csv")

    # D7 count 崩溃表: 合并 per-seed mean/std + 绝对 CI
    cdf = pl.DataFrame(count_rows)
    seed_stat = (df.filter(pl.col("method").is_in(["count_only", "value_count", "mask_only", "value_mask", "value_only"]))
                 .group_by(["direction", "method"])
                 .agg([pl.col("tgt_auroc").mean().alias("tgt_auroc_seedmean"),
                       pl.col("tgt_auroc").std().alias("tgt_auroc_seedstd"),
                       pl.col("tgt_util").mean().alias("tgt_util_seedmean"),
                       pl.col("tgt_util").std().alias("tgt_util_seedstd")])
                 .rename({"method": "feat"}))
    cdf = cdf.join(seed_stat, on=["direction", "feat"], how="left").sort(["direction", "feat"])
    cdf.write_csv(f"{TAB}/stage4_count_collapse.csv")

    # ================= 门 G5 (正确判定) =================
    ci_lookup = {(r["direction"], r["method"], r["vs"], r["metric"]): r for r in ci_rows}
    gates = {"spec": {
        "G5_primary": "OPS_full 与 R4_ipw 双向 tgt_util(集成) 超 max(value_only,value_mask) >= +0.02 "
                      "且 配对 Utility CI 下界(vs value_only 且 vs value_mask) > 0 (方案 §9.1)",
        "G5_2_indomain": "方法 src_util(集成) >= value_mask src_util(集成) - 0.01 (方案 §9.2 院内不退化)",
        "baseline": "最优笨基线 = max(value_only, value_mask) 的 tgt_util(集成)",
        "backbone": "LightGBM 行级", "seeds": args.seeds, "B": args.B}}
    per = {}
    focus = ["value_mask", "R1_reanchor", "R2_msr", "R4_ipw", "OPS_full"]
    emap = {(r["direction"], r["method"]): r for r in ens_rows}
    for direction in ["A2B", "B2A"]:
        vo = emap[(direction, "value_only")]; vm = emap[(direction, "value_mask")]
        best_base = max(vo["ens_tgt_util"], vm["ens_tgt_util"])
        best_base_name = "value_mask" if vm["ens_tgt_util"] >= vo["ens_tgt_util"] else "value_only"
        for meth in focus:
            e = emap[(direction, meth)]
            gain_vs_max = e["ens_tgt_util"] - best_base
            u_vo = ci_lookup.get((direction, meth, "value_only", "utility"))
            u_vm = ci_lookup.get((direction, meth, "value_mask", "utility"))
            a_vo = ci_lookup.get((direction, meth, "value_only", "auroc"))
            a_vm = ci_lookup.get((direction, meth, "value_mask", "auroc"))
            indomain_ok = e["ens_src_util"] >= vm["ens_src_util"] - 0.01
            ci_vo_lo = u_vo["lo"] if u_vo else None
            ci_vm_lo = u_vm["lo"] if u_vm else (0.0 if meth == "value_mask" else None)
            # G5: 超 max 基线 >= +0.02 且 vs 两基线 Utility CI 下界 > 0
            ci_both_pos = (ci_vo_lo is not None and ci_vo_lo > 0) and \
                          (meth == "value_mask" or (ci_vm_lo is not None and ci_vm_lo > 0))
            per[f"{direction}_{meth}"] = dict(
                ens_tgt_util=round(e["ens_tgt_util"], 4), best_base=round(best_base, 4),
                best_base_name=best_base_name, gain_vs_max_base=round(gain_vs_max, 4),
                util_ci_vs_value_only=[u_vo["lo"], u_vo["hi"]] if u_vo else None,
                util_ci_vs_value_mask=[u_vm["lo"], u_vm["hi"]] if u_vm else None,
                auroc_ci_vs_value_only=[a_vo["lo"], a_vo["hi"]] if a_vo else None,
                auroc_ci_vs_value_mask=[a_vm["lo"], a_vm["hi"]] if a_vm else None,
                ens_src_util=round(e["ens_src_util"], 4), value_mask_src_util=round(vm["ens_src_util"], 4),
                indomain_ok=bool(indomain_ok),
                margin_pass=bool(gain_vs_max >= 0.02),
                ci_both_lo_gt0=bool(ci_both_pos),
                G5_pass_this=bool(gain_vs_max >= 0.02 and ci_both_pos))
    gates["per_direction_method"] = per
    # 汇总: OPS_full / R4_ipw 是否双向通过
    for key in ["OPS_full", "R4_ipw"]:
        both = all(per[f"{d}_{key}"]["G5_pass_this"] for d in ["A2B", "B2A"])
        indom = all(per[f"{d}_{key}"]["indomain_ok"] for d in ["A2B", "B2A"])
        gates[f"G5_{key}_bidirectional_pass"] = bool(both)
        gates[f"G5_{key}_indomain_ok_both"] = bool(indom)
    gates["G5_overall_pass"] = bool(gates["G5_OPS_full_bidirectional_pass"] or
                                    gates["G5_R4_ipw_bidirectional_pass"])
    gates["candidate_C_supported"] = gates["G5_overall_pass"]
    json.dump(gates, open(f"{STT}/stage4_ops_lgb_gates.json", "w"), indent=2, ensure_ascii=False)

    print("\n=== ENSEMBLE agg (tgt_util / tgt_auroc) ===")
    print(ens_df.sort(["direction", "method"]).to_pandas().to_string(index=False))
    print("\n=== G5 GATES ===")
    print(json.dumps({k: v for k, v in gates.items() if k != "per_direction_method"}, indent=2, ensure_ascii=False))
    for k, v in per.items():
        print(f"  {k}: tgt_u={v['ens_tgt_util']} base={v['best_base']}({v['best_base_name']}) "
              f"gain={v['gain_vs_max_base']} util_CI_vs_vm={v['util_ci_vs_value_mask']} "
              f"indom={v['indomain_ok']} PASS={v['G5_pass_this']}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--tcap", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--nfit", type=int, default=0)     # 0=全量源训练(=diag_r4/ledger口径); >0 仅 smoke
    main(ap.parse_args())
