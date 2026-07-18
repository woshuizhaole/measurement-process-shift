#!/usr/bin/env python
"""Block 4 :: 严谨主终点 —— GRU/GRU-D backbone (方案 §7.1 双 backbone 之二, §9.1/§9.2/§8.3)

在已验证的 40b_ops_gpu.py 基础上, 修全部预注册缺陷 (对齐 40c_ops_lgb_rigorous):
  1. 方法加 R4_ipw 独立项 (value+mask, 值通道 IPW 条件插补) + count_only/value_count (D7 GRU 复核)。
  2. >=10 种子 headline; CI 用"种子集成"(目标域概率跨种子平均, 源 val 集成选阈)。
  3. 配对 CI 覆盖官方 Utility (util_boot.paired_bootstrap_util_diff, B>=2000, 病人级) 与 AUROC,
     且对 value_only 与 value_mask 两个基线都算。
  4. G5 门: OPS_full 与 R4_ipw 双向 tgt_util(集成) 超 max(value_only,value_mask) >= +0.02
     且 vs 两基线 Utility 配对 CI 下界 > 0; §9.2 院内不退化。
  5. --save-logits: 落盘每 (方向,方法) 的目标/源病人级预测 (P0-6 闸门, 解锁全部预测级 CPU 重分析)。
GRUD 只需 meas_override 即可切换测量通道 (raw mask / MSR / 重锚 / count), 不改共享模块。
"""
import os, sys, json, time, argparse
import numpy as np, polars as pl
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
import statsmodels.api as sm

P = "."
sys.path.insert(0, f"{P}/code/utils")
import features as F
from models import GRUD, normalized_utility, best_threshold
from dualchannel import DualChannel
from ops_repair import MeasurementPropensity, msr, reanchor
from boot import paired_bootstrap_diff, patient_bootstrap_ci
from util_boot import paired_bootstrap_util_diff, bootstrap_util_ci

TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"; LOG = f"{P}/results/logits"

# 方法 -> (GRUD feat_mode, 测量通道 key, 用IPW值, 用DANN)
# feat_mode: value=纯值; value_mask=值+mask通道(可被meas_override替换)+delta; mask=仅mask通道+delta
METHOD_CFG = {
    "value_only":  ("value",      None,        False, False),
    "value_mask":  ("value_mask",  None,        False, False),   # meas_override None -> raw mask
    "mask_only":   ("mask",        None,        False, False),   # [raw mask, delta] 诊断/D7对照
    "R1_reanchor": ("value_mask", "REANCHOR",  False, False),
    "R2_msr":      ("value_mask", "MSR",       False, False),
    "R3_dann_msr": ("value_mask", "MSR",       False, True),
    "R4_ipw":      ("value_mask",  None,        True,  False),   # raw mask, 值通道IPW插补
    "OPS_full":    ("value_mask", "MSR",       True,  False),
    "count_only":  ("mask",       "COUNT",     False, False),    # D7: [count, delta]
    "value_count": ("value_mask", "COUNT",     False, False),    # D7: 值 + count + delta
}
METHODS = list(METHOD_CFG.keys())


def calib_slope(y, p):
    eps = 1e-6; lp = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    try:
        return float(sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1])
    except Exception:
        return np.nan


def build_arrays(rows, X, M, D, Y, L, srcnorm, cap, meta, mp_src=None, mp_h=None, ipw_models=None):
    """padded 张量 + 测量通道变体 MSR/REANCHOR/COUNT + Vipw。返回 dict of np arrays。"""
    N = len(rows); T = cap
    Xv = np.zeros((N, T, 34), np.float32); Mk = np.zeros((N, T, 34), np.float32)
    Dl = np.zeros((N, T, 34), np.float32); Yy = np.zeros((N, T), np.float32); Ln = np.zeros(N, np.int32)
    Ct = np.zeros((N, T, 34), np.float32)                       # 8h 回看观测计数
    Zc = F.ffill_impute(np.asarray(X[rows][:, :cap]), np.asarray(M[rows][:, :cap]), srcnorm, channels=F.TV)
    phi, _ = F.build_phi(X[rows], M[rows], L[rows], meta[list(map(int, rows))], src=None, _norm=srcnorm)
    for k, i in enumerate(rows):
        n = int(min(L[i], cap)); Ln[k] = n
        m = np.asarray(M[i, :n]).astype(np.float32)
        Xv[k, :n] = Zc[k, :n]; Mk[k, :n] = m; Dl[k, :n] = np.log1p(np.asarray(D[i, :n])); Yy[k, :n] = np.asarray(Y[i, :n])
        for t in range(n):
            Ct[k, t] = m[max(0, t - 7):t + 1].sum(axis=0)
    age = meta["age"].to_numpy()[list(map(int, rows))]; gen = meta["gender"].to_numpy()[list(map(int, rows))]
    st = np.stack([(age - np.nanmean(age)) / (np.nanstd(age) + 1e-8), gen], 1).astype(np.float32)
    out = dict(Xv=Xv, Mk=Mk, Dl=Dl, Yy=Yy, Ln=Ln, st=st, rows=np.asarray(rows), COUNT=Ct / 8.0)  # count 归一到~[0,1]
    flat_ph = np.concatenate([phi[k, :Ln[k]] for k in range(N)])
    flat_m = np.concatenate([Mk[k, :Ln[k]] for k in range(N)]).astype(int)

    def scatter(flat):
        A = np.zeros((N, T, 34), np.float32); off = 0
        for k in range(N):
            A[k, :Ln[k]] = flat[off:off + Ln[k]]; off += Ln[k]
        return A
    if mp_h is not None:
        e_h = mp_h.predict_e(flat_ph)
        out["MSR"] = scatter(msr(flat_m, e_h))
        if mp_src is not None:
            e_src = mp_src.predict_e(flat_ph)
            out["REANCHOR"] = scatter(reanchor(flat_m, e_h, e_src))
    if ipw_models is not None:
        flat_v = np.concatenate([Xv[k, :Ln[k]] for k in range(N)])
        for j in range(34):
            if ipw_models[j] is None: continue
            miss = flat_m[:, j] == 0
            flat_v[miss, j] = ipw_models[j].predict(flat_ph[miss])
        out["Vipw"] = scatter(flat_v)
    return out


class DS:
    def __init__(s, d, dev, meas=None, use_ipw=False):
        s.x = torch.tensor(d["Vipw"] if (use_ipw and "Vipw" in d) else d["Xv"], device=dev)
        s.m = torch.tensor(d["Mk"], device=dev); s.dl = torch.tensor(d["Dl"], device=dev)
        s.st = torch.tensor(d["st"], device=dev); s.y = torch.tensor(d["Yy"], device=dev)
        s.L = d["Ln"]; s.rows = d["rows"]
        s.meas = torch.tensor(d[meas], device=dev) if (meas and meas in d) else None


def mbce(logit, y, L, pw):
    T = logit.shape[1]
    mk = (torch.arange(T, device=logit.device)[None] < torch.as_tensor(L, device=logit.device)[:, None])
    l = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none", pos_weight=pw)
    return (l * mk).sum() / mk.sum()


def train(cfg, tr, envs_tr, dev, epochs, lr, bs, pw, seed, lam):
    fmode, meas_key, use_ipw, use_dann = cfg
    torch.manual_seed(seed); np.random.seed(seed)
    nE = int(envs_tr.max()) + 1 if len(envs_tr) else 1
    model = DualChannel(grl_lambda=lam, n_hosp=nE).to(dev) if use_dann else GRUD(feat_mode=fmode, hidden=64).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    pwt = torch.tensor(pw, device=dev); envlab = torch.as_tensor(envs_tr, device=dev)
    for ep in range(epochs):
        model.train()
        idx = np.random.default_rng(seed + ep).permutation(len(tr.rows))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; opt.zero_grad()
            mi = tr.meas[b] if tr.meas is not None else None
            if use_dann:
                o = model(tr.x[b], tr.m[b], tr.dl[b], tr.st[b], meas_in=mi, hosp=envlab[b])
                loss = mbce(o["logit"], tr.y[b], tr.L[b], pwt) + lam * nn.functional.cross_entropy(o["hosp_logit"], envlab[b])
            else:
                logit = (model(tr.x[b], tr.m[b], tr.dl[b], tr.st[b]) if fmode == "value"
                         else model(tr.x[b], tr.m[b], tr.dl[b], tr.st[b], meas_override=mi))
                loss = mbce(logit, tr.y[b], tr.L[b], pwt)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()

    def ev(ds):
        model.eval()
        with torch.no_grad():
            probs, ys, fp, fy, fg = [], [], [], [], []
            for i in range(0, len(ds.rows), 512):
                sl = slice(i, i + 512); mi = ds.meas[sl] if ds.meas is not None else None
                if use_dann:
                    lg = model(ds.x[sl], ds.m[sl], ds.dl[sl], ds.st[sl], meas_in=mi)["logit"]
                elif fmode == "value":
                    lg = model(ds.x[sl], ds.m[sl], ds.dl[sl], ds.st[sl])
                else:
                    lg = model(ds.x[sl], ds.m[sl], ds.dl[sl], ds.st[sl], meas_override=mi)
                pr = torch.sigmoid(lg).cpu().numpy()
                for k in range(pr.shape[0]):
                    n = int(ds.L[i + k])
                    probs.append(pr[k, :n]); ys.append(ds.y[i + k, :n].cpu().numpy().astype(int))
                    fp.append(pr[k, :n]); fy.append(ds.y[i + k, :n].cpu().numpy().astype(int)); fg.append(np.full(n, ds.rows[i + k]))
            return dict(probs=probs, ys=ys, fp=np.concatenate(fp), fy=np.concatenate(fy), fg=np.concatenate(fg))
    return {k: ev(v) for k, v in [("va", None), ("st", None), ("tg", None)]}  # placeholder, replaced in main


def main(args):
    t0 = time.time(); dev = args.device
    os.makedirs(LOG, exist_ok=True)
    X, M, D, Y, L = F.load_arrays(); meta = F.load_meta(); sp = F.load_splits()
    dom = meta["domain"].to_numpy()
    rows_out = []; ens_rows = []; ci_rows = []; count_rows = []

    for direction in args.dirs:
        src, tgt = ("A", "B") if direction == "A2B" else ("B", "A")
        srcnorm = F.load_norm(src)
        tr_r = np.array(sp[f"{src}_train"]); va_r = np.array(sp[f"{src}_val"])
        st_r = np.array(sp[f"{src}_test"]); tg_r = np.array(sp[f"{tgt}_test"])
        # 倾向 (源 OOF+full) + IPW 值插补模型 (源拟合)
        phi_tr, _ = F.build_phi(X[tr_r], M[tr_r], L[tr_r], meta[list(map(int, tr_r))], _norm=srcnorm)
        lens = [int(min(L[tr_r[k]], args.tcap)) for k in range(len(tr_r))]
        fph = np.concatenate([phi_tr[k, :lens[k]] for k in range(len(tr_r))])
        Ztr = F.ffill_impute(np.asarray(X[tr_r][:, :args.tcap]), np.asarray(M[tr_r][:, :args.tcap]), srcnorm, channels=F.TV)
        fm = np.concatenate([np.asarray(M[tr_r[k], :lens[k]]) for k in range(len(tr_r))]).astype(int)
        fg = np.concatenate([np.full(lens[k], tr_r[k]) for k in range(len(tr_r))])
        fv = np.concatenate([Ztr[k, :lens[k]] for k in range(len(tr_r))])
        mp_src = MeasurementPropensity(); e_tr = mp_src.fit_transform_oof(fph, fm, fg, also_full=True)
        ipw = []
        for j in range(34):
            obs = fm[:, j] > 0
            if obs.sum() < 100: ipw.append(None); continue
            w = 1.0 / np.clip(e_tr[obs, j], 0.05, 1)
            ipw.append(Ridge(alpha=1.0).fit(fph[obs], fv[obs, j], sample_weight=w))
        phi_tg, _ = F.build_phi(X[tg_r], M[tg_r], L[tg_r], meta[list(map(int, tg_r))], _norm=srcnorm)  # §Stage0: 目标phi也用源域norm(与40c一致)
        ltg = [int(min(L[tg_r[k]], args.tcap)) for k in range(len(tg_r))]
        ftm = np.concatenate([np.asarray(M[tg_r[k], :ltg[k]]) for k in range(len(tg_r))]).astype(int)
        ftph = np.concatenate([phi_tg[k, :ltg[k]] for k in range(len(tg_r))])
        mp_tgt = MeasurementPropensity().fit(ftph, ftm)
        print(f"[{direction}] propensity+ipw done ({time.time()-t0:.0f}s)", flush=True)

        A_tr = build_arrays(tr_r, X, M, D, Y, L, srcnorm, args.tcap, meta, mp_src, mp_src, ipw)
        A_va = build_arrays(va_r, X, M, D, Y, L, srcnorm, args.tcap, meta, mp_src, mp_src, ipw)
        A_st = build_arrays(st_r, X, M, D, Y, L, srcnorm, args.tcap, meta, mp_src, mp_src, ipw)
        A_tg = build_arrays(tg_r, X, M, D, Y, L, srcnorm, args.tcap, meta, mp_src, mp_tgt, ipw)  # §Stage0: 源域norm套用目标(部署忠实, 与40c一致)
        envs_tr = np.array([{d: i for i, d in enumerate(sorted(set(dom[tr_r])))}[dom[r]] for r in tr_r])
        # 目标/源病人分组 (用于 Utility list + CI)
        gtt = A_tg["rows"]; ytt_flat = None
        print(f"[{direction}] arrays built ({time.time()-t0:.0f}s)", flush=True)

        ens_tt = {}; ens_thr = {}; ens_metrics = {}
        ens_st = {}
        for name in METHODS:
            cfg = METHOD_CFG[name]; meas_key = cfg[1]; use_ipw = cfg[2]
            tr = DS(A_tr, dev, meas_key, use_ipw); va = DS(A_va, dev, meas_key, use_ipw)
            st = DS(A_st, dev, meas_key, use_ipw); tg = DS(A_tg, dev, meas_key, use_ipw)
            sum_va_fp = None; sum_st_fp = None; sum_tt_fp = None
            va_probs_seedmean = None
            perseed_tt = []
            ref = {}
            for seed in range(args.seeds):
                model_out = _train_eval(cfg, tr, va, st, tg, envs_tr, dev, args, seed)
                out = model_out
                thr, _ = best_threshold(out["va"]["ys"], out["va"]["probs"])
                rec = dict(direction=direction, method=name, seed=seed, thr=round(thr, 4),
                           src_auroc=roc_auc_score(out["st"]["fy"], out["st"]["fp"]),
                           tgt_auroc=roc_auc_score(out["tg"]["fy"], out["tg"]["fp"]),
                           src_util=normalized_utility(out["st"]["ys"], out["st"]["probs"], thr),
                           tgt_util=normalized_utility(out["tg"]["ys"], out["tg"]["probs"], thr),
                           tgt_calib=calib_slope(out["tg"]["fy"], out["tg"]["fp"]))
                rows_out.append(rec)
                sum_va_fp = out["va"]["fp"].copy() if sum_va_fp is None else sum_va_fp + out["va"]["fp"]
                sum_st_fp = out["st"]["fp"].copy() if sum_st_fp is None else sum_st_fp + out["st"]["fp"]
                sum_tt_fp = out["tg"]["fp"].copy() if sum_tt_fp is None else sum_tt_fp + out["tg"]["fp"]
                perseed_tt.append(out["tg"]["fp"])
                ref = out  # 保留最后 seed 的分组结构 (fy/fg 与 seed 无关)
                print(f"  {direction} {name:12s} s{seed} src_u={rec['src_util']:.3f} tgt_u={rec['tgt_util']:.3f} "
                      f"tgt_auroc={rec['tgt_auroc']:.3f} calib={rec['tgt_calib']:.2f} ({time.time()-t0:.0f}s)", flush=True)
                torch.cuda.empty_cache()
            # 种子集成
            eva = sum_va_fp / args.seeds; est = sum_st_fp / args.seeds; ett = sum_tt_fp / args.seeds
            # 病人级 list 重建 (基于 ref 的 fg 分组; fy 与 seed 无关)
            va_lists = _relist(ref["va"]["fg"], ref["va"]["fy"], eva)
            st_lists = _relist(ref["st"]["fg"], ref["st"]["fy"], est)
            tt_lists = _relist(ref["tg"]["fg"], ref["tg"]["fy"], ett)
            thr_e, _ = best_threshold(va_lists["ys"], va_lists["ps"])
            ens_tt[name] = dict(fp=ett, fy=ref["tg"]["fy"], fg=ref["tg"]["fg"], ps=tt_lists["ps"], ys=tt_lists["ys"])
            ens_st[name] = dict(ps=st_lists["ps"], ys=st_lists["ys"], fp=est, fy=ref["st"]["fy"])
            ens_thr[name] = thr_e
            emet = dict(direction=direction, method=name, ens_thr=round(thr_e, 4),
                        ens_src_auroc=roc_auc_score(ref["st"]["fy"], est), ens_tgt_auroc=roc_auc_score(ref["tg"]["fy"], ett),
                        ens_src_util=normalized_utility(st_lists["ys"], st_lists["ps"], thr_e),
                        ens_tgt_util=normalized_utility(tt_lists["ys"], tt_lists["ps"], thr_e),
                        ens_tgt_calib=calib_slope(ref["tg"]["fy"], ett))
            ens_metrics[name] = emet; ens_rows.append(emet)
            print(f"  [{direction}] ENSEMBLE {name:12s} src_u={emet['ens_src_util']:.3f} "
                  f"tgt_u={emet['ens_tgt_util']:.3f} tgt_auroc={emet['ens_tgt_auroc']:.3f}", flush=True)
            if args.save_logits:
                np.savez_compressed(f"{LOG}/40d_{direction}_{name}.npz",
                                    tgt_fp_ens=ett, tgt_fy=ref["tg"]["fy"], tgt_fg=ref["tg"]["fg"],
                                    tgt_fp_perseed=np.stack(perseed_tt),
                                    src_fp_ens=est, src_fy=ref["st"]["fy"], src_fg=ref["st"]["fg"],
                                    thr_ens=thr_e)

        if args.skip_ci:      # 只训练+落logits, CI/agg/gates 由 offline_ci_from_logits.py 算(避开慢 roc_auc bootstrap)
            continue
        # ---- 配对 CI: 每方法 vs value_only 与 value_mask (Utility + AUROC) ----
        for name in METHODS:
            if name == "value_only": continue
            for base in ["value_only", "value_mask"]:
                if name == base: continue
                u = paired_bootstrap_util_diff(ens_tt[base]["ys"], ens_tt[base]["ps"], ens_tt[name]["ps"],
                                               ens_thr[base], ens_thr[name], B=args.B, seed=0)
                ci_rows.append(dict(direction=direction, method=name, vs=base, metric="utility",
                                    delta=round(u["point"], 4), lo=round(u["lo"], 4), hi=round(u["hi"], 4),
                                    p_gt0=round(u["p_gt0"], 3), n_pat=u["n_patients"]))
                a = paired_bootstrap_diff(ens_tt[base]["fy"], ens_tt[base]["fp"], ens_tt[name]["fp"],
                                          ens_tt[base]["fg"], "auroc", B=args.B, seed=0)
                ci_rows.append(dict(direction=direction, method=name, vs=base, metric="auroc",
                                    delta=round(a["point"], 4), lo=round(a["lo"], 4), hi=round(a["hi"], 4),
                                    p_gt0=round(a["p_gt0"], 3), n_pat=len(ens_tt[base]["ps"])))
            print(f"  [{direction}] CI done: {name} ({time.time()-t0:.0f}s)", flush=True)

        for feat in ["count_only", "value_count", "mask_only", "value_mask", "value_only"]:
            if feat not in ens_tt: continue
            aci = patient_bootstrap_ci(ens_tt[feat]["fy"], ens_tt[feat]["fp"], ens_tt[feat]["fg"], "auroc", B=args.B, seed=0)
            uci = bootstrap_util_ci(ens_tt[feat]["ys"], ens_tt[feat]["ps"], ens_thr[feat], B=args.B, seed=0)
            em = ens_metrics[feat]
            count_rows.append(dict(direction=direction, feat=feat, ens_src_auroc=round(em["ens_src_auroc"], 4),
                                   ens_tgt_auroc=round(em["ens_tgt_auroc"], 4), tgt_auroc_lo=round(aci["lo"], 4),
                                   tgt_auroc_hi=round(aci["hi"], 4), ens_tgt_util=round(em["ens_tgt_util"], 4),
                                   tgt_util_lo=round(uci["lo"], 4), tgt_util_hi=round(uci["hi"], 4)))

    if args.skip_ci:
        pl.DataFrame(rows_out).write_csv(f"{TAB}/stage4_ops_gru_main_{''.join(args.dirs)}.csv")
        print("[skip-ci] 训练完成, logits 已落盘; agg/CI/gates 由 offline_ci_from_logits.py 从 logits 算")
    else:
        _write_and_gate(rows_out, ens_rows, ci_rows, count_rows, args)
    print(f"total {time.time()-t0:.0f}s")


def _train_eval(cfg, tr, va, st, tg, envs_tr, dev, args, seed):
    """训练一个 seed 并在 va/st/tg 上评估 (复用 train 的 ev 内核)。"""
    fmode, meas_key, use_ipw, use_dann = cfg
    torch.manual_seed(seed); np.random.seed(seed)
    nE = int(envs_tr.max()) + 1 if len(envs_tr) else 1
    model = DualChannel(grl_lambda=args.lam, n_hosp=nE).to(dev) if use_dann else GRUD(feat_mode=fmode, hidden=64).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    pwt = torch.tensor(15.0, device=dev); envlab = torch.as_tensor(envs_tr, device=dev)
    for ep in range(args.epochs):
        model.train()
        idx = np.random.default_rng(seed + ep).permutation(len(tr.rows))
        for i in range(0, len(idx), args.bs):
            b = idx[i:i + args.bs]; opt.zero_grad()
            mi = tr.meas[b] if tr.meas is not None else None
            if use_dann:
                o = model(tr.x[b], tr.m[b], tr.dl[b], tr.st[b], meas_in=mi, hosp=envlab[b])
                loss = mbce(o["logit"], tr.y[b], tr.L[b], pwt) + args.lam * nn.functional.cross_entropy(o["hosp_logit"], envlab[b])
            else:
                logit = (model(tr.x[b], tr.m[b], tr.dl[b], tr.st[b]) if fmode == "value"
                         else model(tr.x[b], tr.m[b], tr.dl[b], tr.st[b], meas_override=mi))
                loss = mbce(logit, tr.y[b], tr.L[b], pwt)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()

    def ev(ds):
        model.eval()
        with torch.no_grad():
            probs, ys, fp, fy, fg = [], [], [], [], []
            for i in range(0, len(ds.rows), 512):
                sl = slice(i, i + 512); mi = ds.meas[sl] if ds.meas is not None else None
                if use_dann:
                    lg = model(ds.x[sl], ds.m[sl], ds.dl[sl], ds.st[sl], meas_in=mi)["logit"]
                elif fmode == "value":
                    lg = model(ds.x[sl], ds.m[sl], ds.dl[sl], ds.st[sl])
                else:
                    lg = model(ds.x[sl], ds.m[sl], ds.dl[sl], ds.st[sl], meas_override=mi)
                pr = torch.sigmoid(lg).cpu().numpy()
                for k in range(pr.shape[0]):
                    n = int(ds.L[i + k])
                    probs.append(pr[k, :n]); ys.append(ds.y[i + k, :n].cpu().numpy().astype(int))
                    fp.append(pr[k, :n]); fy.append(ds.y[i + k, :n].cpu().numpy().astype(int)); fg.append(np.full(n, ds.rows[i + k]))
            return dict(probs=probs, ys=ys, fp=np.concatenate(fp), fy=np.concatenate(fy), fg=np.concatenate(fg))
    return {"va": ev(va), "st": ev(st), "tg": ev(tg)}


def _relist(fg, fy, fp):
    """把 flat (fg 病人分组, fy 标签, fp 概率) 还原成每病人 list (顺序按首次出现)。"""
    order = []
    seen = {}
    for g in fg:
        if g not in seen:
            seen[g] = True; order.append(g)
    ps, ys = [], []
    for g in order:
        sel = fg == g
        ps.append(fp[sel]); ys.append(fy[sel])
    return dict(ps=ps, ys=ys)


def _write_and_gate(rows_out, ens_rows, ci_rows, count_rows, args):
    pl.DataFrame(rows_out).write_csv(f"{TAB}/stage4_ops_gru_main.csv")
    pl.DataFrame(ci_rows).write_csv(f"{TAB}/stage4_ops_gru_ci.csv")
    ens_df = pl.DataFrame(ens_rows)
    df = pl.DataFrame(rows_out)
    stat_cols = ["src_auroc", "tgt_auroc", "src_util", "tgt_util", "tgt_calib"]
    agg = (df.group_by(["direction", "method"])
           .agg([pl.col(c).mean().alias(f"{c}_mean") for c in stat_cols]
                + [pl.col(c).median().alias(f"{c}_median") for c in stat_cols]
                + [pl.col(c).std().alias(f"{c}_std") for c in stat_cols]))
    agg = agg.join(ens_df, on=["direction", "method"], how="left").sort(["direction", "method"])
    agg.write_csv(f"{TAB}/stage4_ops_gru_agg.csv")
    pl.DataFrame(count_rows).write_csv(f"{TAB}/stage4_count_collapse_gru.csv")

    ci_lookup = {(r["direction"], r["method"], r["vs"], r["metric"]): r for r in ci_rows}
    emap = {(r["direction"], r["method"]): r for r in ens_rows}
    gates = {"spec": {"G5_primary": "OPS_full 与 R4_ipw 双向 tgt_util(集成) 超 max(value_only,value_mask) >= +0.02 "
                      "且 vs 两基线 Utility 配对 CI 下界 > 0 (§9.1)", "backbone": "GRU-D 时序",
                      "seeds": args.seeds, "B": args.B}}
    per = {}; dirs = sorted(set(r["direction"] for r in ens_rows))
    for direction in dirs:
        vo = emap[(direction, "value_only")]; vm = emap[(direction, "value_mask")]
        best_base = max(vo["ens_tgt_util"], vm["ens_tgt_util"])
        for meth in ["value_mask", "R1_reanchor", "R2_msr", "R3_dann_msr", "R4_ipw", "OPS_full"]:
            if (direction, meth) not in emap: continue
            e = emap[(direction, meth)]
            u_vo = ci_lookup.get((direction, meth, "value_only", "utility"))
            u_vm = ci_lookup.get((direction, meth, "value_mask", "utility"))
            ci_vo_lo = u_vo["lo"] if u_vo else None
            ci_vm_lo = u_vm["lo"] if u_vm else (0.0 if meth == "value_mask" else None)
            ci_both = (ci_vo_lo is not None and ci_vo_lo > 0) and (meth == "value_mask" or (ci_vm_lo is not None and ci_vm_lo > 0))
            gain = e["ens_tgt_util"] - best_base
            per[f"{direction}_{meth}"] = dict(ens_tgt_util=round(e["ens_tgt_util"], 4), best_base=round(best_base, 4),
                gain_vs_max_base=round(gain, 4), util_ci_vs_value_only=[u_vo["lo"], u_vo["hi"]] if u_vo else None,
                util_ci_vs_value_mask=[u_vm["lo"], u_vm["hi"]] if u_vm else None,
                ens_src_util=round(e["ens_src_util"], 4), value_mask_src_util=round(vm["ens_src_util"], 4),
                indomain_ok=bool(e["ens_src_util"] >= vm["ens_src_util"] - 0.01),
                margin_pass=bool(gain >= 0.02), ci_both_lo_gt0=bool(ci_both),
                G5_pass_this=bool(gain >= 0.02 and ci_both))
    gates["per_direction_method"] = per
    for key in ["OPS_full", "R4_ipw"]:
        gates[f"G5_{key}_bidirectional_pass"] = bool(all(per.get(f"{d}_{key}", {}).get("G5_pass_this", False) for d in dirs))
    gates["G5_overall_pass"] = bool(gates.get("G5_OPS_full_bidirectional_pass") or gates.get("G5_R4_ipw_bidirectional_pass"))
    gates["candidate_C_supported"] = gates["G5_overall_pass"]
    json.dump(gates, open(f"{STT}/stage4_ops_gru_gates.json", "w"), indent=2, ensure_ascii=False)
    print("\n=== ENSEMBLE ==="); print(ens_df.sort(["direction", "method"]).to_pandas().to_string(index=False))
    print("\n=== G5 GATES ==="); print(json.dumps({k: v for k, v in gates.items() if k != "per_direction_method"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--tcap", type=int, default=200)
    ap.add_argument("--B", type=int, default=2000); ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--dirs", nargs="+", default=["A2B", "B2A"]); ap.add_argument("--save-logits", action="store_true")
    ap.add_argument("--skip-ci", action="store_true")
    main(ap.parse_args())
