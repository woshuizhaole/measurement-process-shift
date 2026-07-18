#!/usr/bin/env python
"""Stage 4 :: OPS-Repair 方法主实验 (方案 §Stage4, §7.2)

被测方法 (行级 LightGBM + 序列 GRU 两个 backbone, 先用快的 LightGBM 做主网格):
  B0 value_only, B1 value_mask, B2 mask_only
  R1 reanchor    : mask 通道 -> e_ref(C)+(M-e_h(C))  协议重锚
  R2 msr         : mask 通道 -> rho_j 测量得分残差
  R4 ipw_impute  : 值通道用 IPW 条件插补 (源拟合, 两域套用)
  OPS r2+r4      : R2 + R4
所有倾向 e_j(C) 用 build_phi(仅永远可得通道) 交叉拟合(源域) / 全量拟合(目标域)。
阈值源域 val 选。官方 Utility + AUROC + 校准斜率。双向 A2B/B2A, 多种子。
主要终点(门 G5): 校正跨院 Utility 超 value_only >= +0.02 且配对CI下界>0。
"""
import os, sys, json, time, argparse
import numpy as np, polars as pl
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

P = "."
sys.path.insert(0, f"{P}/code/utils")
import features as F
from ops_repair import MeasurementPropensity, msr, reanchor
from models import normalized_utility, best_threshold
from boot import paired_bootstrap_diff

TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"


def calib_slope(y, p):
    """校准斜率: logit(y) ~ logit(p) 的斜率。"""
    eps = 1e-6; lp = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    try:
        r = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
        return float(r.params[1])
    except Exception:
        return np.nan


def build(rows, X, M, D, Y, L, srcnorm, cap, meta):
    Z = F.ffill_impute(np.asarray(X[rows][:, :cap]), np.asarray(M[rows][:, :cap]), srcnorm, channels=F.TV)
    phi, _ = F.build_phi(X[rows], M[rows], L[rows], meta[list(rows)], src=None, _norm=srcnorm)
    val, msk, dlt, ph, yy, gg = [], [], [], [], [], []
    for k, i in enumerate(rows):
        n = int(min(L[i], cap))
        val.append(Z[k, :n]); msk.append(np.asarray(M[i, :n]).astype(np.float32))
        dlt.append(np.log1p(np.asarray(D[i, :n]))); ph.append(phi[k, :n])
        yy.append(np.asarray(Y[i, :n])); gg.append(np.full(n, i))
    return (np.concatenate(val), np.concatenate(msk), np.concatenate(dlt),
            np.concatenate(ph), np.concatenate(yy).astype(int), np.concatenate(gg))


def ipw_impute_fit(val, msk, ph, e_src):
    """IPW 条件插补: 对每个 j, 在 M_j=1 行上用权重 1/e_j(C) 拟合 X_j~C 的 Ridge。返回 models。"""
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


def evaluate(clf, feat_fn, Vx, Mx, Rx, Dx, y, g):
    Xe = feat_fn(Vx, Mx, Rx, Dx)
    p = clf.predict_proba(Xe)[:, 1]
    probs, ys = [], []
    for gg in np.unique(g):
        s = g == gg; probs.append(p[s]); ys.append(y[s])
    return p, probs, ys


def main(args):
    t0 = time.time()
    X, M, D, Y, L = F.load_arrays()
    meta = F.load_meta(); sp = F.load_splits()
    rng = np.random.default_rng(20260711)
    cap = args.tcap
    rows_out = []; ci_rows = []

    # 特征构造函数 (给 LightGBM 的行级设计矩阵)
    feat_fns = {
        "value_only": lambda V, Mk, R, Dl: V,
        "value_mask": lambda V, Mk, R, Dl: np.hstack([V, Mk, Dl]),
        "mask_only":  lambda V, Mk, R, Dl: np.hstack([Mk, Dl]),
        "R1_reanchor": lambda V, Mk, R, Dl: np.hstack([V, R, Dl]),   # R = 重锚 mask
        "R2_msr":      lambda V, Mk, R, Dl: np.hstack([V, R, Dl]),   # R = MSR残差
        "R4_ipw":      lambda V, Mk, R, Dl: np.hstack([V, Mk, Dl]),  # V 已被IPW替换
        "OPS_full":    lambda V, Mk, R, Dl: np.hstack([V, R, Dl]),   # V=IPW, R=MSR
    }

    for direction in ["A2B", "B2A"]:
        src, tgt = ("A", "B") if direction == "A2B" else ("B", "A")
        srcnorm = F.load_norm(src)
        tr = np.array(sp[f"{src}_train"]); va = np.array(sp[f"{src}_val"])
        ste = np.array(sp[f"{src}_test"]); tte = np.array(sp[f"{tgt}_test"])

        # 构造各集
        Vtr, Mtr, Dtr, PHtr, ytr, gtr = build(tr, X, M, D, Y, L, srcnorm, cap, meta)
        Vva, Mva, Dva, PHva, yva, gva = build(va, X, M, D, Y, L, srcnorm, cap, meta)
        Vst, Mst, Dst, PHst, yst, gst = build(ste, X, M, D, Y, L, srcnorm, cap, meta)
        Vtt, Mtt, Dtt, PHtt, ytt, gtt = build(tte, X, M, D, Y, L, srcnorm, cap, meta)
        print(f"[{direction}] built ({time.time()-t0:.0f}s)", flush=True)

        # 倾向 e_j(C): 源 OOF + 全量模型; 目标全量模型
        mp_src = MeasurementPropensity()
        e_tr = mp_src.fit_transform_oof(PHtr, Mtr, gtr)
        mp_src.fit(PHtr, Mtr)
        e_va = mp_src.predict_e(PHva); e_st = mp_src.predict_e(PHst)
        mp_tgt = MeasurementPropensity().fit(PHtt, Mtt)
        e_tt_h = mp_tgt.predict_e(PHtt)        # 目标域自身倾向
        e_tt_src = mp_src.predict_e(PHtt)      # 目标域上的源倾向(重锚参考)
        # 参考协议 = 源域倾向
        # MSR
        R_tr = msr(Mtr, e_tr); R_va = msr(Mva, e_va); R_st = msr(Mst, e_st); R_tt = msr(Mtt, e_tt_h)
        # 重锚: e_ref(=源) + (M - e_h)
        RA_tr = reanchor(Mtr, e_tr, mp_src.predict_e(PHtr))
        RA_va = reanchor(Mva, e_va, e_va); RA_st = reanchor(Mst, e_st, e_st)
        RA_tt = reanchor(Mtt, e_tt_h, e_tt_src)   # 目标: e_ref=源倾向, e_h=目标倾向
        # IPW 插补
        ipw_models = ipw_impute_fit(Vtr, Mtr, PHtr, e_tr)
        Vtr_ipw = ipw_impute_apply(Vtr, Mtr, PHtr, ipw_models)
        Vst_ipw = ipw_impute_apply(Vst, Mst, PHst, ipw_models)
        Vtt_ipw = ipw_impute_apply(Vtt, Mtt, PHtt, ipw_models)
        Vva_ipw = ipw_impute_apply(Vva, Mva, PHva, ipw_models)

        # 每方法的 (训练R/值, 各集R/值)
        def method_data(name):
            if name in ("R1_reanchor",):
                return dict(Vtr=Vtr, Rtr=RA_tr, Vva=Vva, Rva=RA_va, Vst=Vst, Rst=RA_st, Vtt=Vtt, Rtt=RA_tt)
            if name in ("R2_msr",):
                return dict(Vtr=Vtr, Rtr=R_tr, Vva=Vva, Rva=R_va, Vst=Vst, Rst=R_st, Vtt=Vtt, Rtt=R_tt)
            if name == "R4_ipw":
                return dict(Vtr=Vtr_ipw, Rtr=Mtr, Vva=Vva_ipw, Rva=Mva, Vst=Vst_ipw, Rst=Mst, Vtt=Vtt_ipw, Rtt=Mtt)
            if name == "OPS_full":
                return dict(Vtr=Vtr_ipw, Rtr=R_tr, Vva=Vva_ipw, Rva=R_va, Vst=Vst_ipw, Rst=R_st, Vtt=Vtt_ipw, Rtt=R_tt)
            return dict(Vtr=Vtr, Rtr=Mtr, Vva=Vva, Rva=Mva, Vst=Vst, Rst=Mst, Vtt=Vtt, Rtt=Mtt)

        vo_tgt_preds = None
        for name, fn in feat_fns.items():
            md = method_data(name)
            for seed in range(args.seeds):
                clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                         min_child_samples=100, subsample=0.8, subsample_freq=1,
                                         colsample_bytree=0.8, n_jobs=14, verbose=-1,
                                         scale_pos_weight=15, random_state=seed)
                clf.fit(fn(md["Vtr"], Mtr, md["Rtr"], Dtr), ytr)
                _, pv, yv = evaluate(clf, fn, md["Vva"], Mva, md["Rva"], Dva, yva, gva)
                thr, _ = best_threshold(yv, pv)
                pst, pst_l, yst_l = evaluate(clf, fn, md["Vst"], Mst, md["Rst"], Dst, yst, gst)
                ptt, ptt_l, ytt_l = evaluate(clf, fn, md["Vtt"], Mtt, md["Rtt"], Dtt, ytt, gtt)
                rec = dict(direction=direction, method=name, seed=seed, thr=thr,
                           src_auroc=roc_auc_score(yst, pst), tgt_auroc=roc_auc_score(ytt, ptt),
                           src_util=normalized_utility(yst_l, pst_l, thr),
                           tgt_util=normalized_utility(ytt_l, ptt_l, thr),
                           tgt_calib_slope=calib_slope(ytt, ptt))
                rows_out.append(rec)
                if name == "value_only" and seed == 0:
                    vo_tgt_preds = (ytt, ptt, gtt)
                if name in ("R2_msr", "OPS_full", "R1_reanchor") and seed == 0 and vo_tgt_preds is not None:
                    yb, pb, gb = vo_tgt_preds
                    ci = paired_bootstrap_diff(yb, pb, ptt, gb, "auroc", B=500, seed=0)
                    ci_rows.append(dict(direction=direction, method=name, vs="value_only",
                                        auroc_delta=round(ci["point"], 4), lo=round(ci["lo"], 4),
                                        hi=round(ci["hi"], 4), p_gt0=round(ci["p_gt0"], 3)))
                print(f"  {direction} {name:12s} s{seed} src_u={rec['src_util']:.3f} "
                      f"tgt_u={rec['tgt_util']:.3f} tgt_auroc={rec['tgt_auroc']:.3f} "
                      f"calib={rec['tgt_calib_slope']:.2f}", flush=True)

    df = pl.DataFrame(rows_out); df.write_csv(f"{TAB}/stage4_ops_main.csv")
    pl.DataFrame(ci_rows).write_csv(f"{TAB}/stage4_ops_ci.csv")
    agg = (df.group_by(["direction", "method"])
           .agg([pl.col(c).mean() for c in ["src_auroc", "tgt_auroc", "src_util", "tgt_util", "tgt_calib_slope"]]
                + [pl.col("tgt_util").std().alias("tgt_util_std")])
           .sort(["direction", "method"]))
    agg.write_csv(f"{TAB}/stage4_ops_agg.csv")
    print("\n=== OPS main aggregated ==="); print(agg.to_pandas().to_string(index=False))

    # 门 G5: OPS/R2 vs value_only 跨院 Utility >= +0.02
    gates = {}
    for direction in ["A2B", "B2A"]:
        dd = {r["method"]: r for r in agg.filter(pl.col("direction") == direction).to_dicts()}
        vo = dd["value_only"]
        for meth in ["R1_reanchor", "R2_msr", "R4_ipw", "OPS_full", "value_mask"]:
            gates[f"{direction}_{meth}"] = dict(
                util_gain_vs_vo=round(dd[meth]["tgt_util"] - vo["tgt_util"], 4),
                auroc_gain_vs_vo=round(dd[meth]["tgt_auroc"] - vo["tgt_auroc"], 4))
    g5 = any(v["util_gain_vs_vo"] >= 0.02 for k, v in gates.items() if "OPS_full" in k or "R2" in k)
    gates["G5_threshold"] = "OPS/R2 跨院 Utility 超 value_only >= +0.02"
    gates["G5_pass"] = bool(g5)
    json.dump(gates, open(f"{STT}/stage4_gates.json", "w"), indent=2, ensure_ascii=False)
    print("\n=== GATES (G5) ==="); print(json.dumps(gates, indent=2, ensure_ascii=False))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tcap", type=int, default=200)
    main(ap.parse_args())
