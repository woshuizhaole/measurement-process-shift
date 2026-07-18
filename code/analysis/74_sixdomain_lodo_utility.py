#!/usr/bin/env python
"""Block 5 :: Utility 空间「六域 LODO + ICUType 院内正对照」(LightGBM backbone)

新科学问题: 主终点(40c, A<->B)确立的「Utility 空间崩溃 + 值通道(R4_ipw)方向性修复」模式,
在更广的六域 LODO(留一域) 与更温和的院内单元漂移(ICUType 正对照) 下是否成立/如何变化。

严格复刻 40c_ops_lgb_rigorous.py 的方法学:
  * 值归一化统计只在**源域训练集**上计算(compute_norm 复刻 03_splits_and_norm.py 配方),
    原样套用到目标域; 绝不在目标域上计算任何"值"统计量(§Stage0 铁律)。
  * 倾向 e_j(C) 用 build_phi(仅永远可得通道) 源域 OOF 交叉拟合; MSR 残差按 方案 §6.4
    用局部(域内)倾向剥离协议、保留不变 gamma*G (与 40c 完全一致: mp_tgt 在目标 mask 上拟合)。
  * IPW 条件插补(R4)在源拟合、两域套用。阈值在源域 val 上选(source-domain threshold selection)。
  * 官方 Utility(主) + AUROC(辅); 病人级(cluster)配对 bootstrap CI (util_boot, B>=2000, >=5 seeds)。

方法(4): value_only(哑基线/killer) | value_mask(ERM) | R4_ipw(值通道修复) | OPS_full。

折(fold):
  LODO 六折: 每折留一域, 其余 5 域并集作源训练(role=train), 源 val(role=val) 选阈, held-out 域**整体**作目标测试。
  ICU 院内正对照 4 折: A-MICU<->A-SICU, B-MICU<->B-SICU (同院单元间, 更温和漂移正对照)。
    源=单域(train/val/test 分层), 目标=同院另一单元**整体**。

输出:
  results/stats/stage7_lodo_utility.json   (点估计 + 配对CI + 判定门)
  results/tables/stage7_lodo_utility.csv   (每折每方法一行)
  results/logits/74_{fold}_{method}.npz    (病人级 tgt_fg/tgt_fy/tgt_fp_ens/thr_ens, 便于离线复算)
  results/stats/stage7_lodo_progress.jsonl (逐折 append, 断点可续)
"""
import os, sys, json, time, argparse
import numpy as np, polars as pl
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

P = "."
sys.path.insert(0, f"{P}/code/utils")
import features as F
from ops_repair import MeasurementPropensity, msr
from models import normalized_utility, best_threshold           # 保留(对拍用)
from boot import paired_bootstrap_diff, patient_bootstrap_ci
from util_boot import paired_bootstrap_util_diff, bootstrap_util_ci
from util_boot import _cpu as _CPU, DT_EARLY, DT_OPTIMAL, DT_LATE

TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"; LOG = f"{P}/results/logits"
GRID = np.linspace(0.01, 0.6, 60)     # 与 models.best_threshold 默认网格一致


# ---- fast Utility: 缓存 u_best/u_inact(仅依赖 y, 与阈值/预测无关), 复用于全网格/全方法/全 seed ----
# 与 models.normalized_utility 逐位等价(u_best/u_inact 由 y 决定; util_boot __main__ 已对拍官方)。
def uparts_yfixed(y_list):
    ub = np.empty(len(y_list)); ui = np.empty(len(y_list))
    for k, y in enumerate(y_list):
        y = np.asarray(y).astype(int); m = len(y)
        best = np.zeros(m); inact = np.zeros(m)
        if np.any(y):
            ts = int(np.argmax(y)) - DT_OPTIMAL
            best[max(0, ts + DT_EARLY): min(ts + int(DT_LATE) + 1, m)] = 1
        ub[k] = _CPU(y, best); ui[k] = _CPU(y, inact)
    return ub, ui


def fast_util(y_list, p_list, thr, ub, ui):
    uo = 0.0
    for y, p in zip(y_list, p_list):
        uo += _CPU(np.asarray(y).astype(int), (np.asarray(p) >= thr).astype(int))
    denom = ub.sum() - ui.sum()
    return 0.0 if denom == 0 else float((uo - ui.sum()) / denom)


def fast_best_threshold(y_list, p_list, ub, ui):
    best_u, best_t = -9.0, 0.5
    for t in GRID:
        u = fast_util(y_list, p_list, t, ub, ui)
        if u > best_u:
            best_u, best_t = u, float(t)
    return best_t, best_u

METHODS = ["value_only", "value_mask", "R4_ipw", "OPS_full"]
FEAT_FNS = {
    "value_only": lambda V, Mk, R, Dl: V,
    "value_mask": lambda V, Mk, R, Dl: np.hstack([V, Mk, Dl]),
    "R4_ipw":     lambda V, Mk, R, Dl: np.hstack([V, Mk, Dl]),   # V 已被 IPW 条件插补替换
    "OPS_full":   lambda V, Mk, R, Dl: np.hstack([V, R, Dl]),    # V=IPW, R=MSR
}


def compute_norm(rows, X, M, L):
    """在给定(源)行集合上计算每通道归一化统计, 逐位复刻 03_splits_and_norm.py。
    winsorize[0.1,99.9] observed 值 -> mean/std; median/iqr 用未截断; 附 p01/p999。"""
    T_CAP = X.shape[1]
    valid = np.arange(T_CAP)[None, :] < L[rows][:, None]
    Xr = np.asarray(X[rows]); Mr = np.asarray(M[rows])
    stats = {}
    for j, name in enumerate(F.TV):
        m = Mr[:, :, j].astype(bool) & valid
        vals = Xr[:, :, j][m]; vals = vals[np.isfinite(vals)]
        denom = max(int(valid.sum()), 1)
        if vals.size < 10:
            stats[name] = dict(n=int(vals.size), mean=0.0, std=1.0, median=0.0, iqr=1.0,
                               p01=0.0, p999=1.0, observed_frac=float(m.sum() / denom))
            continue
        p01, p999 = np.percentile(vals, [0.1, 99.9])
        clipped = np.clip(vals, p01, p999)
        q1, q3 = np.percentile(vals, [25, 75])
        stats[name] = dict(n=int(vals.size), mean=float(clipped.mean()), std=float(clipped.std() + 1e-8),
                           median=float(np.median(vals)), iqr=float(q3 - q1 + 1e-8),
                           p01=float(p01), p999=float(p999), observed_frac=float(m.sum() / denom))
    return stats


def build(rows, X, M, D, Y, L, srcnorm, cap, meta):
    """行级拼接 (val=源norm归一化ffill值, msk=raw mask, dlt=log1p(delta), ph=phi(C), yy, gg=病人行索引)。"""
    Z = F.ffill_impute(np.asarray(X[rows][:, :cap]), np.asarray(M[rows][:, :cap]), srcnorm, channels=F.TV)
    phi, _ = F.build_phi(X[rows], M[rows], L[rows], meta[list(map(int, rows))], src=None, _norm=srcnorm)
    val, msk, dlt, ph, yy, gg = [], [], [], [], [], []
    for k, i in enumerate(rows):
        n = int(min(L[i], cap))
        val.append(Z[k, :n]); msk.append(np.asarray(M[i, :n]).astype(np.float32))
        dlt.append(np.log1p(np.asarray(D[i, :n]))); ph.append(phi[k, :n])
        yy.append(np.asarray(Y[i, :n])); gg.append(np.full(n, i))
    return (np.concatenate(val), np.concatenate(msk), np.concatenate(dlt),
            np.concatenate(ph), np.concatenate(yy).astype(int), np.concatenate(gg))


def ipw_impute_fit(val, msk, ph, e_src):
    models = []
    for j in range(34):
        obs = msk[:, j] > 0
        if obs.sum() < 100:
            models.append(None); continue
        w = 1.0 / np.clip(e_src[obs, j], 0.05, 1)
        models.append(Ridge(alpha=1.0).fit(ph[obs], val[obs, j], sample_weight=w))
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
    uniq, inv = np.unique(g, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    sinv = inv[order]
    bnd = np.searchsorted(sinv, np.arange(len(uniq) + 1))
    return uniq, [order[bnd[c]:bnd[c + 1]] for c in range(len(uniq))]


def to_lists(v, idx_list):
    return [v[ix] for ix in idx_list]


def run_fold(fold, src_tr, src_va, src_te, tgt, X, M, D, Y, L, meta, args, t0):
    """跑单折: 返回 (per_method_metrics dict, ci_rows list, ens_tt/thr for save)。"""
    srcnorm = compute_norm(src_tr, X, M, L)
    cap = args.tcap
    Vtr, Mtr, Dtr, PHtr, ytr, gtr = build(src_tr, X, M, D, Y, L, srcnorm, cap, meta)
    Vva, Mva, Dva, PHva, yva, gva = build(src_va, X, M, D, Y, L, srcnorm, cap, meta)
    Vst, Mst, Dst, PHst, yst, gst = build(src_te, X, M, D, Y, L, srcnorm, cap, meta)
    Vtt, Mtt, Dtt, PHtt, ytt, gtt = build(tgt, X, M, D, Y, L, srcnorm, cap, meta)
    print(f"[{fold}] rows tr={len(ytr)} va={len(yva)} st={len(yst)} tt={len(ytt)} "
          f"n_pat tgt={len(np.unique(gtt))} tgt_prev={ytt.mean():.4f} ({time.time()-t0:.0f}s)", flush=True)

    _, ixva = patient_index(gva); yva_L = to_lists(yva, ixva)
    _, ixst = patient_index(gst); yst_L = to_lists(yst, ixst)
    _, ixtt = patient_index(gtt); ytt_L = to_lists(ytt, ixtt)
    # 每评估集 u_best/u_inact 预算一次(阈值/预测无关), 全方法全 seed 复用
    ub_va, ui_va = uparts_yfixed(yva_L)
    ub_st, ui_st = uparts_yfixed(yst_L)
    ub_tt, ui_tt = uparts_yfixed(ytt_L)

    # 倾向 e_j(C): 源 OOF + 源全量模型; MSR 残差用局部(域内)倾向(§6.4, 同 40c)
    mp_src = MeasurementPropensity()
    e_tr = mp_src.fit_transform_oof(PHtr, Mtr, gtr)
    mp_src.fit(PHtr, Mtr)
    mp_tgt = MeasurementPropensity().fit(PHtt, Mtt)
    e_tt_h = mp_tgt.predict_e(PHtt)
    R_tr = msr(Mtr, e_tr)
    R_va = msr(Mva, mp_src.predict_e(PHva))
    R_st = msr(Mst, mp_src.predict_e(PHst))
    R_tt = msr(Mtt, e_tt_h)
    # IPW 条件插补(源拟合, 两域套用)
    ipw_models = ipw_impute_fit(Vtr, Mtr, PHtr, e_tr)
    Vtr_ipw = ipw_impute_apply(Vtr, Mtr, PHtr, ipw_models)
    Vva_ipw = ipw_impute_apply(Vva, Mva, PHva, ipw_models)
    Vst_ipw = ipw_impute_apply(Vst, Mst, PHst, ipw_models)
    Vtt_ipw = ipw_impute_apply(Vtt, Mtt, PHtt, ipw_models)
    print(f"[{fold}] propensity+IPW done ({time.time()-t0:.0f}s)", flush=True)

    def method_data(name):
        if name == "R4_ipw":
            return dict(Vtr=Vtr_ipw, Rtr=Mtr, Vva=Vva_ipw, Rva=Mva, Vst=Vst_ipw, Rst=Mst, Vtt=Vtt_ipw, Rtt=Mtt)
        if name == "OPS_full":
            return dict(Vtr=Vtr_ipw, Rtr=R_tr, Vva=Vva_ipw, Rva=R_va, Vst=Vst_ipw, Rst=R_st, Vtt=Vtt_ipw, Rtt=R_tt)
        return dict(Vtr=Vtr, Rtr=Mtr, Vva=Vva, Rva=Mva, Vst=Vst, Rst=Mst, Vtt=Vtt, Rtt=Mtt)

    ens_tt = {}; ens_thr = {}; ens_metrics = {}; seed_rows = []
    for name in METHODS:
        fn = FEAT_FNS[name]; md = method_data(name)
        Xtr = fn(md["Vtr"], Mtr, md["Rtr"], Dtr)
        Xva = fn(md["Vva"], Mva, md["Rva"], Dva)
        Xst = fn(md["Vst"], Mst, md["Rst"], Dst)
        Xtt = fn(md["Vtt"], Mtt, md["Rtt"], Dtt)
        sum_va = np.zeros(len(yva)); sum_st = np.zeros(len(yst)); sum_tt = np.zeros(len(ytt))
        for seed in range(args.seeds):
            clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                     min_child_samples=100, subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.8, n_jobs=args.jobs, verbose=-1,
                                     scale_pos_weight=15, random_state=seed)
            clf.fit(Xtr, ytr)
            pva = clf.predict_proba(Xva)[:, 1]; pst = clf.predict_proba(Xst)[:, 1]; ptt = clf.predict_proba(Xtt)[:, 1]
            sum_va += pva; sum_st += pst; sum_tt += ptt
            thr_s, _ = fast_best_threshold(yva_L, to_lists(pva, ixva), ub_va, ui_va)
            seed_rows.append(dict(fold=fold, method=name, seed=seed, thr=round(thr_s, 4),
                                  src_auroc=float(roc_auc_score(yst, pst)), tgt_auroc=float(roc_auc_score(ytt, ptt)),
                                  src_util=fast_util(yst_L, to_lists(pst, ixst), thr_s, ub_st, ui_st),
                                  tgt_util=fast_util(ytt_L, to_lists(ptt, ixtt), thr_s, ub_tt, ui_tt)))
        eva = sum_va / args.seeds; est = sum_st / args.seeds; ett = sum_tt / args.seeds
        thr_e, _ = fast_best_threshold(yva_L, to_lists(eva, ixva), ub_va, ui_va)
        ens_tt[name] = ett; ens_thr[name] = thr_e
        ens_metrics[name] = dict(fold=fold, method=name, ens_thr=round(thr_e, 4),
                                 ens_src_auroc=float(roc_auc_score(yst, est)), ens_tgt_auroc=float(roc_auc_score(ytt, ett)),
                                 ens_src_util=fast_util(yst_L, to_lists(est, ixst), thr_e, ub_st, ui_st),
                                 ens_tgt_util=fast_util(ytt_L, to_lists(ett, ixtt), thr_e, ub_tt, ui_tt))
        # 保存病人级 logits (集成)
        np.savez(f"{LOG}/74_{fold}_{name}.npz", tgt_fg=gtt, tgt_fy=ytt, tgt_fp_ens=ett, thr_ens=thr_e)
        em = ens_metrics[name]
        print(f"  [{fold}] {name:11s} src_u={em['ens_src_util']:.3f} tgt_u={em['ens_tgt_util']:.3f} "
              f"tgt_auroc={em['ens_tgt_auroc']:.3f} thr={thr_e:.3f}", flush=True)

    # ---- 单方法绝对 Utility/AUROC CI + 配对 CI(R4_ipw/OPS_full vs value_only & value_mask) ----
    ptt_L = {m: to_lists(ens_tt[m], ixtt) for m in METHODS}
    abs_ci = {}
    for m in METHODS:
        uci = bootstrap_util_ci(ytt_L, ptt_L[m], ens_thr[m], B=args.B, seed=0)
        aci = patient_bootstrap_ci(ytt, ens_tt[m], gtt, "auroc", B=args.B, seed=0)
        abs_ci[m] = dict(util_lo=round(uci["lo"], 4), util_hi=round(uci["hi"], 4),
                         auroc_lo=round(aci["lo"], 4), auroc_hi=round(aci["hi"], 4))
    ci_rows = []
    for name in ["R4_ipw", "OPS_full"]:
        for base in ["value_only", "value_mask"]:
            u = paired_bootstrap_util_diff(ytt_L, ptt_L[base], ptt_L[name], ens_thr[base], ens_thr[name],
                                           B=args.B, seed=0)
            a = paired_bootstrap_diff(ytt, ens_tt[base], ens_tt[name], gtt, "auroc", B=args.B, seed=0)
            ci_rows.append(dict(fold=fold, method=name, vs=base,
                                util_delta=round(u["point"], 4), util_lo=round(u["lo"], 4), util_hi=round(u["hi"], 4),
                                util_p_gt0=round(u["p_gt0"], 3), n_pat=u["n_patients"],
                                auroc_delta=round(a["point"], 4), auroc_lo=round(a["lo"], 4), auroc_hi=round(a["hi"], 4)))
    print(f"  [{fold}] CI done ({time.time()-t0:.0f}s)", flush=True)
    return ens_metrics, abs_ci, ci_rows


def main(args):
    t0 = time.time()
    os.makedirs(LOG, exist_ok=True)
    X, M, D, Y, L = F.load_arrays()
    meta = F.load_meta(); sp = F.load_splits()
    N = meta.height
    dom = meta["domain"].to_numpy()
    DOMAINS = ["A-MICU", "A-SICU", "A-unknown", "B-MICU", "B-SICU", "B-unknown"]
    # role: 每行 train/val/test (splits_idx 是行索引)
    role = np.array(["none"] * N, dtype=object)
    for h in ["A", "B"]:
        for r in ["train", "val", "test"]:
            role[np.array(sp[f"{h}_{r}"])] = r

    # 折定义: (fold_name, kind, selector) selector 返回 (src_tr,src_va,src_te,tgt)
    folds = []
    for d in DOMAINS:
        folds.append((f"LODO_{d}", "lodo", d))
    ICU_PAIRS = [("A-MICU", "A-SICU"), ("A-SICU", "A-MICU"), ("B-MICU", "B-SICU"), ("B-SICU", "B-MICU")]
    for sd, td in ICU_PAIRS:
        folds.append((f"ICU_{sd}2{td}", "icu", (sd, td)))
    if args.only:
        keep = set(args.only.split(","))
        folds = [f for f in folds if f[0] in keep]

    canonical = [(f"LODO_{d}", "lodo") for d in DOMAINS] + \
                [(f"ICU_{sd}2{td}", "icu") for sd, td in ICU_PAIRS]

    prog = open(f"{STT}/stage7_lodo_progress.jsonl", "a")
    for fold, kind, sel in folds:
        if kind == "lodo":
            d = sel
            src_tr = np.where((dom != d) & (role == "train"))[0]
            src_va = np.where((dom != d) & (role == "val"))[0]
            src_te = np.where((dom != d) & (role == "test"))[0]
            tgt = np.where(dom == d)[0]
        else:
            sd, td = sel
            src_tr = np.where((dom == sd) & (role == "train"))[0]
            src_va = np.where((dom == sd) & (role == "val"))[0]
            src_te = np.where((dom == sd) & (role == "test"))[0]
            tgt = np.where(dom == td)[0]
        if args.nfit and len(src_tr) > args.nfit:
            src_tr = np.random.default_rng(20260712).choice(src_tr, args.nfit, replace=False)
        em, absci, ci = run_fold(fold, src_tr, src_va, src_te, tgt, X, M, D, Y, L, meta, args, t0)
        prog.write(json.dumps(dict(fold=fold, kind=kind, metrics={m: em[m] for m in METHODS},
                                   abs_ci=absci, paired_ci=ci), ensure_ascii=False) + "\n"); prog.flush()
    prog.close()

    # ===== 从 progress.jsonl 重建全量结果(按 fold 去重取最新, 支持断点续跑聚合) =====
    latest = {}
    for line in open(f"{STT}/stage7_lodo_progress.jsonl"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line); latest[rec["fold"]] = rec
    agg_folds = [(f, k) for f, k in canonical if f in latest]
    all_metrics = [latest[f]["metrics"][m] for f, _ in agg_folds for m in METHODS]
    all_absci = {f: latest[f]["abs_ci"] for f, _ in agg_folds}
    all_ci = [c for f, _ in agg_folds for c in latest[f]["paired_ci"]]
    emap = {(r["fold"], r["method"]): r for r in all_metrics}
    cimap = {(r["fold"], r["method"], r["vs"]): r for r in all_ci}

    # ===== 落盘表 + 判定门 =====
    rows = []
    for fold, kind in agg_folds:
        vo = emap[(fold, "value_only")]; vm = emap[(fold, "value_mask")]
        base_max = max(vo["ens_tgt_util"], vm["ens_tgt_util"])
        base_name = "value_mask" if vm["ens_tgt_util"] >= vo["ens_tgt_util"] else "value_only"
        for m in METHODS:
            e = emap[(fold, m)]; ab = all_absci[fold][m]
            row = dict(fold=fold, kind=kind, method=m,
                       ens_src_util=round(e["ens_src_util"], 4), ens_tgt_util=round(e["ens_tgt_util"], 4),
                       ens_src_auroc=round(e["ens_src_auroc"], 4), ens_tgt_auroc=round(e["ens_tgt_auroc"], 4),
                       ens_thr=e["ens_thr"], tgt_util_lo=ab["util_lo"], tgt_util_hi=ab["util_hi"],
                       tgt_auroc_lo=ab["auroc_lo"], tgt_auroc_hi=ab["auroc_hi"],
                       best_base=round(base_max, 4), best_base_name=base_name,
                       gain_vs_max_base=round(e["ens_tgt_util"] - base_max, 4))
            if m in ("R4_ipw", "OPS_full"):
                cvo = cimap[(fold, m, "value_only")]; cvm = cimap[(fold, m, "value_mask")]
                # 相对 max 基线的配对 CI(取 argmax 基线那条)
                cmax = cvm if base_name == "value_mask" else cvo
                row.update(util_delta_vs_max=cmax["util_delta"], util_lo_vs_max=cmax["util_lo"],
                           util_hi_vs_max=cmax["util_hi"], util_p_gt0_vs_max=cmax["util_p_gt0"],
                           util_lo_vs_vo=cvo["util_lo"], util_hi_vs_vo=cvo["util_hi"],
                           util_lo_vs_vm=cvm["util_lo"], util_hi_vs_vm=cvm["util_hi"],
                           repair_holds=bool(cmax["util_lo"] > 0),
                           repair_holds_both=bool(cvo["util_lo"] > 0 and cvm["util_lo"] > 0))
            rows.append(row)
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(f"{TAB}/stage7_lodo_utility.csv")

    # 汇总 JSON
    out = dict(spec=dict(methods=METHODS, seeds=args.seeds, B=args.B, tcap=args.tcap,
                         backbone="LightGBM row-level", threshold="source-val selection",
                         norm="source-train only (compute_norm, 复刻03)",
                         msr_note="MSR 残差用局部域内倾向(§6.4, 同40c); 值统计严格源域",
                         target_test="held-out 域整体(全部病人, 因该域完全未进训练)"),
               folds={}, metrics=all_metrics, abs_ci=all_absci, paired_ci=all_ci)
    for fold, kind in agg_folds:
        vo = emap[(fold, "value_only")]; vm = emap[(fold, "value_mask")]
        base_max = max(vo["ens_tgt_util"], vm["ens_tgt_util"])
        base_name = "value_mask" if vm["ens_tgt_util"] >= vo["ens_tgt_util"] else "value_only"
        fd = dict(kind=kind, base_max=round(base_max, 4), base_name=base_name,
                  tgt_util={m: round(emap[(fold, m)]["ens_tgt_util"], 4) for m in METHODS},
                  tgt_auroc={m: round(emap[(fold, m)]["ens_tgt_auroc"], 4) for m in METHODS})
        for m in ("R4_ipw", "OPS_full"):
            cmax = cimap[(fold, m, base_name)]
            fd[f"{m}_gain_vs_max"] = round(emap[(fold, m)]["ens_tgt_util"] - base_max, 4)
            fd[f"{m}_paired_util_vs_max"] = [cmax["util_lo"], cmax["util_hi"]]
            fd[f"{m}_repair_holds"] = bool(cmax["util_lo"] > 0)
        out["folds"][fold] = fd
    # 汇总判定
    lodo_folds = [f for f, k in agg_folds if k == "lodo"]
    icu_folds = [f for f, k in agg_folds if k == "icu"]
    def holds_count(mth, flist):
        return sum(1 for f in flist if out["folds"][f].get(f"{mth}_repair_holds", False))
    out["summary"] = dict(
        lodo_folds=lodo_folds, icu_folds=icu_folds,
        R4_lodo_holds=f"{holds_count('R4_ipw', lodo_folds)}/{len(lodo_folds)}",
        OPS_lodo_holds=f"{holds_count('OPS_full', lodo_folds)}/{len(lodo_folds)}",
        R4_icu_holds=f"{holds_count('R4_ipw', icu_folds)}/{len(icu_folds)}",
        OPS_icu_holds=f"{holds_count('OPS_full', icu_folds)}/{len(icu_folds)}",
        note="repair_holds = 配对 Utility CI(vs max(value_only,value_mask)) 下界 > 0")
    json.dump(out, open(f"{STT}/stage7_lodo_utility.json", "w"), indent=2, ensure_ascii=False)

    print("\n=== FOLD SUMMARY (tgt_util) ===")
    print(df.select(["fold", "method", "ens_src_util", "ens_tgt_util", "gain_vs_max_base",
                     "tgt_util_lo", "tgt_util_hi"]).to_pandas().to_string(index=False))
    print("\n=== repair holds ==="); print(json.dumps(out["summary"], indent=2, ensure_ascii=False))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--tcap", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--nfit", type=int, default=0)      # 0=全量源训练; >0 仅 smoke
    ap.add_argument("--only", type=str, default="")     # 逗号分隔 fold 名, 仅跑子集(smoke/断点)
    main(ap.parse_args())
