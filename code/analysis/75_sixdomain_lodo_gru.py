#!/usr/bin/env python
"""Block 5-GPU :: 六域 LODO + ICUType 院内正对照, **GRU-D backbone** (improvement.md P0-GPU 决定性实验)

科学问题: 主终点(40d, A<->B)得到的「时序 GRU-D 比行级 LGB 更可修 (R4_ipw>value_mask)」机制,
  现只由 GRU 的 A2B/B2A 两个跨院方向 (n=2) 支撑, 对 6 个 LGB LODO 折 (74) 严重不对称/功效不足。
  本脚本补齐 **GRU-D 在六域 LODO** 上的结果, 与 74_sixdomain_lodo_utility.py 同域(fold)配对对比,
  判定 backbone 机制能否泛化 → 决定 candidate C-mainline 升主线 (≥3/6 折 R4>value_mask 且 CI>0
  而 LGB 打平) 还是收敛 candidate D。

融合来源 (只做 backbone 替换, 方法学严格不变):
  * 折设计 / 源域选阈 / util_boot CI  参照 74_sixdomain_lodo_utility.py:
      - DOMAINS 六域 LODO (留一域: 源=其余5域并集, 目标=held-out 域整体) + ICU_PAIRS 4 院内对。
      - compute_norm 只在 src_tr 上算值归一化统计 (复刻 03_splits_and_norm.py), 原样套目标域。
      - 阈值只在源域 val 上选 (source-domain threshold selection)。
      - 病人级 cluster 配对 bootstrap CI (util_boot / boot, B>=2000)。
      - --only 子集断点, --seeds。
  * GRU-D backbone / 方法 / 源域 norm 套目标 / save-logits  参照 40d_ops_gru_rigorous.py:
      - METHOD_CFG (本脚本取 4 个: value_only/value_mask/R4_ipw/OPS_full)。
      - build_arrays: 源域 norm(此处=compute_norm(src_tr)) 原样套目标集 (§Stage0, 与 40d L184/194 一致)。
      - MeasurementPropensity 源 OOF+full; MSR 残差用目标域内局部倾向(§6.4); IPW 条件插补源拟合两域套用。
      - GRU-D 训练 + 多种子集成 (目标域概率跨种子平均, 源 val 集成选阈)。
      - --save-logits: 落盘 40d 的 8 字段到 results/logits/{tag}_{fold}_{method}.npz。

方法固定 4: value_only | value_mask | R4_ipw(值通道 IPW 条件插补) | OPS_full(IPW 值 + MSR 测量通道)。

§Stage0 铁律 (40d 曾栽, 本脚本严格遵守):
  归一化统计只在源域 src_tr 上算、原样套目标域; 阈值只源 val 选; IPW Ridge 只源拟合、套目标 phi;
  目标集绝不用目标域 norm。build_arrays 内部 phi 亦用 srcnorm 建 (含目标)。

输出:
  results/logits/{tag}_{fold}_{method}.npz              (8 字段, 与 40d 一致)
  results/tables/stage7_lodo_gru_ens_{tag}.csv          (每折每方法集成指标)
  results/tables/stage7_lodo_gru_seed_{tag}.csv         (每折每方法每 seed)
  results/stats/stage7_lodo_gru_progress_{tag}.jsonl    (逐折 append, 断点可续聚合)
  results/tables/stage7_lodo_gru_utility_{tag}.csv      (非 skip-ci: 每折每方法 + CI)
  results/stats/stage7_lodo_gru_utility_{tag}.json      (非 skip-ci: 判定门)
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
from ops_repair import MeasurementPropensity, msr
from boot import paired_bootstrap_diff, patient_bootstrap_ci
from util_boot import paired_bootstrap_util_diff, bootstrap_util_ci

TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"; LOG = f"{P}/results/logits"

# 方法 -> (GRUD feat_mode, 测量通道 key, 用IPW值). 取自 40d METHOD_CFG 的 4 个 (无 DANN 分支)。
METHOD_CFG = {
    "value_only":  ("value",      None,   False),
    "value_mask":  ("value_mask", None,   False),   # meas_override None -> raw mask
    "R4_ipw":      ("value_mask", None,   True),    # raw mask, 值通道 IPW 条件插补
    "OPS_full":    ("value_mask", "MSR",  True),    # MSR 测量通道 + IPW 值
}
METHODS = list(METHOD_CFG.keys())


# ---------------- 值归一化: 只在源域训练集上算 (74.compute_norm, 逐位复刻 03_splits_and_norm.py) ----------------
def compute_norm(rows, X, M, L):
    """在给定(源)行集合上计算每通道归一化统计。winsorize[0.1,99.9] observed 值 -> mean/std;
    median/iqr 用未截断; 附 p01/p999。输出 dict 与 F.load_norm 同构, 可直接喂 F.ffill_impute/F.build_phi。"""
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


def calib_slope(y, p):
    eps = 1e-6; lp = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    try:
        return float(sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1])
    except Exception:
        return np.nan


# ---------------- build_arrays: 40d 版, 源域 norm 套目标 (§Stage0); 仅保留 4 方法所需通道 ----------------
def build_arrays(rows, X, M, D, Y, L, srcnorm, cap, meta, mp_h=None, ipw_models=None):
    """padded 张量 (Xv/Mk/Dl/Yy/Ln/st) + MSR(值通道 override, 需 mp_h) + Vipw(IPW 条件插补, 需 ipw_models)。
    srcnorm 一律为源域统计 (compute_norm(src_tr)); 对目标集亦原样套用 (部署忠实, 与 40d L194 一致)。
    去掉 40d 的 COUNT(慢的逐时双循环) 与 REANCHOR(本 4 方法不用)。"""
    N = len(rows); T = cap
    Xv = np.zeros((N, T, 34), np.float32); Mk = np.zeros((N, T, 34), np.float32)
    Dl = np.zeros((N, T, 34), np.float32); Yy = np.zeros((N, T), np.float32); Ln = np.zeros(N, np.int32)
    # §Stage0: 值 ffill 用 srcnorm; phi 亦用 srcnorm (含目标集)
    Zc = F.ffill_impute(np.asarray(X[rows][:, :cap]), np.asarray(M[rows][:, :cap]), srcnorm, channels=F.TV)
    phi, _ = F.build_phi(X[rows], M[rows], L[rows], meta[list(map(int, rows))], src=None, _norm=srcnorm)
    for k, i in enumerate(rows):
        n = int(min(L[i], cap)); Ln[k] = n
        Xv[k, :n] = Zc[k, :n]; Mk[k, :n] = np.asarray(M[i, :n]).astype(np.float32)
        Dl[k, :n] = np.log1p(np.asarray(D[i, :n])); Yy[k, :n] = np.asarray(Y[i, :n])
    age = meta["age"].to_numpy()[list(map(int, rows))]; gen = meta["gender"].to_numpy()[list(map(int, rows))]
    st = np.stack([(age - np.nanmean(age)) / (np.nanstd(age) + 1e-8), gen], 1).astype(np.float32)
    out = dict(Xv=Xv, Mk=Mk, Dl=Dl, Yy=Yy, Ln=Ln, st=st, rows=np.asarray(rows))
    flat_ph = np.concatenate([phi[k, :Ln[k]] for k in range(N)])
    flat_m = np.concatenate([Mk[k, :Ln[k]] for k in range(N)]).astype(int)

    def scatter(flat):
        A = np.zeros((N, T, 34), np.float32); off = 0
        for k in range(N):
            A[k, :Ln[k]] = flat[off:off + Ln[k]]; off += Ln[k]
        return A
    if mp_h is not None:                                    # OPS_full 用: MSR 残差 (mp_h=源或目标局部倾向)
        e_h = mp_h.predict_e(flat_ph)
        out["MSR"] = scatter(msr(flat_m, e_h))
    if ipw_models is not None:                              # R4_ipw/OPS_full 用: 值通道 IPW 条件插补
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


def _train_eval(cfg, tr, va, st, tg, dev, args, seed):
    """训练一个 seed 的 GRU-D 并在 va/st/tg 上评估 (40d 内核, 无 DANN 分支)。"""
    fmode, meas_key, use_ipw = cfg
    torch.manual_seed(seed); np.random.seed(seed)
    model = GRUD(feat_mode=fmode, hidden=64).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    pwt = torch.tensor(15.0, device=dev)                    # 与 LGB scale_pos_weight=15 一致
    for ep in range(args.epochs):
        model.train()
        idx = np.random.default_rng(seed + ep).permutation(len(tr.rows))
        for i in range(0, len(idx), args.bs):
            b = idx[i:i + args.bs]; opt.zero_grad()
            mi = tr.meas[b] if tr.meas is not None else None
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
                if fmode == "value":
                    lg = model(ds.x[sl], ds.m[sl], ds.dl[sl], ds.st[sl])
                else:
                    lg = model(ds.x[sl], ds.m[sl], ds.dl[sl], ds.st[sl], meas_override=mi)
                pr = torch.sigmoid(lg).cpu().numpy()
                for k in range(pr.shape[0]):
                    n = int(ds.L[i + k])
                    probs.append(pr[k, :n]); ys.append(ds.y[i + k, :n].cpu().numpy().astype(int))
                    fp.append(pr[k, :n]); fy.append(ds.y[i + k, :n].cpu().numpy().astype(int))
                    fg.append(np.full(n, ds.rows[i + k]))
            return dict(probs=probs, ys=ys, fp=np.concatenate(fp), fy=np.concatenate(fy), fg=np.concatenate(fg))
    return {"va": ev(va), "st": ev(st), "tg": ev(tg)}


def _relist(fg, fy, fp):
    """flat (fg 病人分组, fy 标签, fp 概率) -> 每病人 list (顺序按首次出现)。"""
    order, seen = [], {}
    for g in fg:
        if g not in seen:
            seen[g] = True; order.append(g)
    ps, ys = [], []
    for g in order:
        sel = fg == g; ps.append(fp[sel]); ys.append(fy[sel])
    return dict(ps=ps, ys=ys)


# ---------------- 单折: GRU-D 训练 + 多种子集成 + 源域选阈 + (可选)配对 CI ----------------
def run_fold(fold, src_tr, src_va, src_te, tgt, X, M, D, Y, L, meta, dom, args, t0, prog):
    dev = args.device; cap = args.tcap
    srcnorm = compute_norm(src_tr, X, M, L)                 # §Stage0: 只在 src_tr 算
    # 倾向 e_j(C): 源 OOF+full (build_phi 用 srcnorm) + IPW 值插补模型 (源拟合)
    phi_tr, _ = F.build_phi(X[src_tr], M[src_tr], L[src_tr], meta[list(map(int, src_tr))], _norm=srcnorm)
    lens = [int(min(L[src_tr[k]], cap)) for k in range(len(src_tr))]
    fph = np.concatenate([phi_tr[k, :lens[k]] for k in range(len(src_tr))])
    Ztr = F.ffill_impute(np.asarray(X[src_tr][:, :cap]), np.asarray(M[src_tr][:, :cap]), srcnorm, channels=F.TV)
    fm = np.concatenate([np.asarray(M[src_tr[k], :lens[k]]) for k in range(len(src_tr))]).astype(int)
    fg = np.concatenate([np.full(lens[k], src_tr[k]) for k in range(len(src_tr))])
    fv = np.concatenate([Ztr[k, :lens[k]] for k in range(len(src_tr))])
    mp_src = MeasurementPropensity(); e_tr = mp_src.fit_transform_oof(fph, fm, fg, also_full=True)
    ipw = []
    for j in range(34):
        obs = fm[:, j] > 0
        if obs.sum() < 100: ipw.append(None); continue
        w = 1.0 / np.clip(e_tr[obs, j], 0.05, 1)
        ipw.append(Ridge(alpha=1.0).fit(fph[obs], fv[obs, j], sample_weight=w))
    # 目标域局部倾向 (MSR §6.4). phi 用 srcnorm (§Stage0: 目标 phi 也用源域 norm, 与 40d L184 一致)
    phi_tg, _ = F.build_phi(X[tgt], M[tgt], L[tgt], meta[list(map(int, tgt))], _norm=srcnorm)
    ltg = [int(min(L[tgt[k]], cap)) for k in range(len(tgt))]
    ftm = np.concatenate([np.asarray(M[tgt[k], :ltg[k]]) for k in range(len(tgt))]).astype(int)
    ftph = np.concatenate([phi_tg[k, :ltg[k]] for k in range(len(tgt))])
    mp_tgt = MeasurementPropensity().fit(ftph, ftm)
    print(f"[{fold}] propensity+IPW done n_pat tgt={len(tgt)} ({time.time()-t0:.0f}s)", flush=True)

    A_tr = build_arrays(src_tr, X, M, D, Y, L, srcnorm, cap, meta, mp_src, ipw)   # 源集: MSR 用源倾向
    A_va = build_arrays(src_va, X, M, D, Y, L, srcnorm, cap, meta, mp_src, ipw)
    A_st = build_arrays(src_te, X, M, D, Y, L, srcnorm, cap, meta, mp_src, ipw)
    A_tg = build_arrays(tgt,    X, M, D, Y, L, srcnorm, cap, meta, mp_tgt, ipw)   # 目标集: MSR 用目标局部倾向
    print(f"[{fold}] arrays built ({time.time()-t0:.0f}s)", flush=True)

    ens_tt = {}; ens_thr = {}; ens_metrics = {}; seed_rows = []
    for name in METHODS:
        cfg = METHOD_CFG[name]; meas_key = cfg[1]; use_ipw = cfg[2]
        tr = DS(A_tr, dev, meas_key, use_ipw); va = DS(A_va, dev, meas_key, use_ipw)
        st = DS(A_st, dev, meas_key, use_ipw); tg = DS(A_tg, dev, meas_key, use_ipw)
        sum_va = sum_st = sum_tt = None; perseed_tt = []; ref = {}
        for seed in range(args.seeds):
            out = _train_eval(cfg, tr, va, st, tg, dev, args, seed)
            thr, _ = best_threshold(out["va"]["ys"], out["va"]["probs"])          # 源 val 选阈
            seed_rows.append(dict(fold=fold, method=name, seed=seed, thr=round(thr, 4),
                                  src_auroc=float(roc_auc_score(out["st"]["fy"], out["st"]["fp"])),
                                  tgt_auroc=float(roc_auc_score(out["tg"]["fy"], out["tg"]["fp"])),
                                  src_util=normalized_utility(out["st"]["ys"], out["st"]["probs"], thr),
                                  tgt_util=normalized_utility(out["tg"]["ys"], out["tg"]["probs"], thr),
                                  tgt_calib=calib_slope(out["tg"]["fy"], out["tg"]["fp"])))
            sum_va = out["va"]["fp"].copy() if sum_va is None else sum_va + out["va"]["fp"]
            sum_st = out["st"]["fp"].copy() if sum_st is None else sum_st + out["st"]["fp"]
            sum_tt = out["tg"]["fp"].copy() if sum_tt is None else sum_tt + out["tg"]["fp"]
            perseed_tt.append(out["tg"]["fp"]); ref = out
            print(f"  {fold} {name:11s} s{seed} src_u={seed_rows[-1]['src_util']:.3f} "
                  f"tgt_u={seed_rows[-1]['tgt_util']:.3f} tgt_auroc={seed_rows[-1]['tgt_auroc']:.3f} "
                  f"calib={seed_rows[-1]['tgt_calib']:.2f} ({time.time()-t0:.0f}s)", flush=True)
            torch.cuda.empty_cache()
        eva = sum_va / args.seeds; est = sum_st / args.seeds; ett = sum_tt / args.seeds
        va_L = _relist(ref["va"]["fg"], ref["va"]["fy"], eva)
        st_L = _relist(ref["st"]["fg"], ref["st"]["fy"], est)
        tt_L = _relist(ref["tg"]["fg"], ref["tg"]["fy"], ett)
        thr_e, _ = best_threshold(va_L["ys"], va_L["ps"])                          # 集成源 val 选阈
        ens_tt[name] = dict(fp=ett, fy=ref["tg"]["fy"], fg=ref["tg"]["fg"], ps=tt_L["ps"], ys=tt_L["ys"])
        ens_thr[name] = thr_e
        ens_metrics[name] = dict(fold=fold, method=name, ens_thr=round(thr_e, 4),
                                 ens_src_auroc=float(roc_auc_score(ref["st"]["fy"], est)),
                                 ens_tgt_auroc=float(roc_auc_score(ref["tg"]["fy"], ett)),
                                 ens_src_util=normalized_utility(st_L["ys"], st_L["ps"], thr_e),
                                 ens_tgt_util=normalized_utility(tt_L["ys"], tt_L["ps"], thr_e),
                                 ens_tgt_calib=calib_slope(ref["tg"]["fy"], ett))
        em = ens_metrics[name]
        print(f"  [{fold}] ENS {name:11s} src_u={em['ens_src_util']:.3f} tgt_u={em['ens_tgt_util']:.3f} "
              f"tgt_auroc={em['ens_tgt_auroc']:.3f} thr={thr_e:.3f}", flush=True)
        if args.save_logits:                                                      # 40d 的 8 字段
            np.savez_compressed(f"{LOG}/{args.tag}_{fold}_{name}.npz",
                                tgt_fp_ens=ett, tgt_fy=ref["tg"]["fy"], tgt_fg=ref["tg"]["fg"],
                                tgt_fp_perseed=np.stack(perseed_tt),
                                src_fp_ens=est, src_fy=ref["st"]["fy"], src_fg=ref["st"]["fg"],
                                thr_ens=thr_e)

    abs_ci = None; ci_rows = None
    if not args.skip_ci:
        abs_ci = {}
        for m in METHODS:
            uci = bootstrap_util_ci(ens_tt[m]["ys"], ens_tt[m]["ps"], ens_thr[m], B=args.B, seed=0)
            aci = patient_bootstrap_ci(ens_tt[m]["fy"], ens_tt[m]["fp"], ens_tt[m]["fg"], "auroc", B=args.B, seed=0)
            abs_ci[m] = dict(util_lo=round(uci["lo"], 4), util_hi=round(uci["hi"], 4),
                             auroc_lo=round(aci["lo"], 4), auroc_hi=round(aci["hi"], 4))
        ci_rows = []
        for name in ["R4_ipw", "OPS_full"]:
            for base in ["value_only", "value_mask"]:
                u = paired_bootstrap_util_diff(ens_tt[base]["ys"], ens_tt[base]["ps"], ens_tt[name]["ps"],
                                               ens_thr[base], ens_thr[name], B=args.B, seed=0)
                a = paired_bootstrap_diff(ens_tt[base]["fy"], ens_tt[base]["fp"], ens_tt[name]["fp"],
                                          ens_tt[base]["fg"], "auroc", B=args.B, seed=0)
                ci_rows.append(dict(fold=fold, method=name, vs=base,
                                    util_delta=round(u["point"], 4), util_lo=round(u["lo"], 4), util_hi=round(u["hi"], 4),
                                    util_p_gt0=round(u["p_gt0"], 3), n_pat=u["n_patients"],
                                    auroc_delta=round(a["point"], 4), auroc_lo=round(a["lo"], 4), auroc_hi=round(a["hi"], 4)))
        print(f"  [{fold}] CI done ({time.time()-t0:.0f}s)", flush=True)

    # 逐折 append (断点可续聚合)
    prog.write(json.dumps(dict(fold=fold, metrics={m: ens_metrics[m] for m in METHODS},
                               seed_rows=seed_rows, abs_ci=abs_ci, paired_ci=ci_rows),
                          ensure_ascii=False) + "\n"); prog.flush()
    return ens_metrics, abs_ci, ci_rows, seed_rows


def main(args):
    t0 = time.time()
    torch.set_num_threads(args.jobs)
    os.makedirs(LOG, exist_ok=True); os.makedirs(TAB, exist_ok=True); os.makedirs(STT, exist_ok=True)
    X, M, D, Y, L = F.load_arrays(); meta = F.load_meta(); sp = F.load_splits()
    N = meta.height; dom = meta["domain"].to_numpy()
    DOMAINS = ["A-MICU", "A-SICU", "A-unknown", "B-MICU", "B-SICU", "B-unknown"]
    role = np.array(["none"] * N, dtype=object)
    for h in ["A", "B"]:
        for r in ["train", "val", "test"]:
            role[np.array(sp[f"{h}_{r}"])] = r

    folds = [(f"LODO_{d}", "lodo", d) for d in DOMAINS]
    ICU_PAIRS = [("A-MICU", "A-SICU"), ("A-SICU", "A-MICU"), ("B-MICU", "B-SICU"), ("B-SICU", "B-MICU")]
    folds += [(f"ICU_{sd}2{td}", "icu", (sd, td)) for sd, td in ICU_PAIRS]
    canonical = [(f"LODO_{d}", "lodo") for d in DOMAINS] + [(f"ICU_{sd}2{td}", "icu") for sd, td in ICU_PAIRS]
    if args.only:
        keep = set(args.only.split(",")); folds = [f for f in folds if f[0] in keep]

    prog = open(f"{STT}/stage7_lodo_gru_progress_{args.tag}.jsonl", "a")
    all_seed = []; all_ens = []
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
        if args.nfit and len(src_tr) > args.nfit:                # 仅 smoke: 抽稀源训练
            src_tr = np.random.default_rng(20260713).choice(src_tr, args.nfit, replace=False)
        em, absci, ci, seed_rows = run_fold(fold, src_tr, src_va, src_te, tgt,
                                            X, M, D, Y, L, meta, dom, args, t0, prog)
        all_seed += seed_rows; all_ens += [em[m] for m in METHODS]
    prog.close()

    # 逐折表 (总是落) —— 便于 skip-ci 下直接看 tgt_util 量级
    if all_seed:
        pl.DataFrame(all_seed).write_csv(f"{TAB}/stage7_lodo_gru_seed_{args.tag}.csv")
        pl.DataFrame(all_ens).write_csv(f"{TAB}/stage7_lodo_gru_ens_{args.tag}.csv")
    print("\n=== ENSEMBLE (tgt_util) ===")
    print(pl.DataFrame(all_ens).select(["fold", "method", "ens_src_util", "ens_tgt_util",
          "ens_tgt_auroc", "ens_tgt_calib"]).to_pandas().to_string(index=False))

    if args.skip_ci:
        print("\n[skip-ci] 训练完成, logits 已落盘; 配对 CI / 判定门可由 logits 离线算 (offline_ci_74.py 同款)。")
        print(f"total {time.time()-t0:.0f}s"); return

    _aggregate_and_gate(canonical, args, t0)


def _aggregate_and_gate(canonical, args, t0):
    """从 progress.jsonl 重建全量 (按 fold 去重取最新), 落 utility csv/json + 判定门 (74 同款)。"""
    latest = {}
    for line in open(f"{STT}/stage7_lodo_gru_progress_{args.tag}.jsonl"):
        line = line.strip()
        if line:
            rec = json.loads(line); latest[rec["fold"]] = rec
    agg = [(f, k) for f, k in canonical if f in latest]
    emap = {(f, m): latest[f]["metrics"][m] for f, _ in agg for m in METHODS}
    cimap = {(f, r["method"], r["vs"]): r for f, _ in agg if latest[f]["paired_ci"] for r in latest[f]["paired_ci"]}
    absmap = {f: latest[f]["abs_ci"] for f, _ in agg}

    rows = []; folds_out = {}
    for fold, kind in agg:
        vo = emap[(fold, "value_only")]; vm = emap[(fold, "value_mask")]
        base_max = max(vo["ens_tgt_util"], vm["ens_tgt_util"])
        base_name = "value_mask" if vm["ens_tgt_util"] >= vo["ens_tgt_util"] else "value_only"
        for m in METHODS:
            e = emap[(fold, m)]; ab = (absmap[fold] or {}).get(m, {})
            row = dict(fold=fold, kind=kind, method=m,
                       ens_src_util=round(e["ens_src_util"], 4), ens_tgt_util=round(e["ens_tgt_util"], 4),
                       ens_src_auroc=round(e["ens_src_auroc"], 4), ens_tgt_auroc=round(e["ens_tgt_auroc"], 4),
                       ens_tgt_calib=round(e["ens_tgt_calib"], 4) if e["ens_tgt_calib"] == e["ens_tgt_calib"] else None,
                       ens_thr=e["ens_thr"], tgt_util_lo=ab.get("util_lo"), tgt_util_hi=ab.get("util_hi"),
                       tgt_auroc_lo=ab.get("auroc_lo"), tgt_auroc_hi=ab.get("auroc_hi"),
                       best_base=round(base_max, 4), best_base_name=base_name,
                       gain_vs_max_base=round(e["ens_tgt_util"] - base_max, 4))
            if m in ("R4_ipw", "OPS_full") and (fold, m, "value_only") in cimap:
                cvo = cimap[(fold, m, "value_only")]; cvm = cimap[(fold, m, "value_mask")]
                cmax = cvm if base_name == "value_mask" else cvo
                row.update(util_delta_vs_max=cmax["util_delta"], util_lo_vs_max=cmax["util_lo"],
                           util_hi_vs_max=cmax["util_hi"], util_p_gt0_vs_max=cmax["util_p_gt0"],
                           util_lo_vs_vo=cvo["util_lo"], util_lo_vs_vm=cvm["util_lo"],
                           repair_holds=bool(cmax["util_lo"] > 0),
                           repair_holds_both=bool(cvo["util_lo"] > 0 and cvm["util_lo"] > 0))
            rows.append(row)
        fd = dict(kind=kind, base_max=round(base_max, 4), base_name=base_name,
                  tgt_util={m: round(emap[(fold, m)]["ens_tgt_util"], 4) for m in METHODS},
                  tgt_auroc={m: round(emap[(fold, m)]["ens_tgt_auroc"], 4) for m in METHODS})
        for m in ("R4_ipw", "OPS_full"):
            if (fold, m, base_name) in cimap:
                cmax = cimap[(fold, m, base_name)]
                fd[f"{m}_gain_vs_max"] = round(emap[(fold, m)]["ens_tgt_util"] - base_max, 4)
                fd[f"{m}_paired_util_vs_max"] = [cmax["util_lo"], cmax["util_hi"]]
                fd[f"{m}_repair_holds"] = bool(cmax["util_lo"] > 0)
        folds_out[fold] = fd
    pl.DataFrame(rows, infer_schema_length=None).write_csv(f"{TAB}/stage7_lodo_gru_utility_{args.tag}.csv")

    lodo_folds = [f for f, k in agg if k == "lodo"]; icu_folds = [f for f, k in agg if k == "icu"]
    def holds(mth, flist): return sum(1 for f in flist if folds_out[f].get(f"{mth}_repair_holds", False))
    out = dict(spec=dict(methods=METHODS, seeds=args.seeds, B=args.B, tcap=args.tcap,
                         backbone="GRU-D 时序 (hidden=64, 多种子集成)", threshold="source-val selection",
                         norm="source-train only (compute_norm, 复刻03)",
                         msr_note="MSR 残差用目标域内局部倾向(§6.4, 同40c/74); 值统计严格源域",
                         target_test="held-out 域整体(全部病人)"),
               folds=folds_out, metrics=list(emap.values()),
               summary=dict(lodo_folds=lodo_folds, icu_folds=icu_folds,
                            R4_lodo_holds=f"{holds('R4_ipw', lodo_folds)}/{len(lodo_folds)}",
                            OPS_lodo_holds=f"{holds('OPS_full', lodo_folds)}/{len(lodo_folds)}",
                            R4_icu_holds=f"{holds('R4_ipw', icu_folds)}/{len(icu_folds)}",
                            OPS_icu_holds=f"{holds('OPS_full', icu_folds)}/{len(icu_folds)}",
                            note="repair_holds = 配对 Utility CI(vs max(value_only,value_mask)) 下界 > 0",
                            candidate_C_mainline_rule="≥3/6 LODO 折 R4_ipw repair_holds (与 74 LGB 同域对比)"))
    json.dump(out, open(f"{STT}/stage7_lodo_gru_utility_{args.tag}.json", "w"), indent=2, ensure_ascii=False)
    print("\n=== repair holds ==="); print(json.dumps(out["summary"], indent=2, ensure_ascii=False))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tcap", type=int, default=200)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=4)                # CPU 线程帽 (torch); BLAS 由 env 帽
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--only", type=str, default="")               # 逗号分隔 fold 名, 子集/断点
    ap.add_argument("--nfit", type=int, default=0)                # 0=全量源训练; >0 仅 smoke 抽稀
    ap.add_argument("--tag", type=str, default="75")              # logits/表/进度 前缀 (smoke 用 75smoke)
    ap.add_argument("--save-logits", action="store_true")
    ap.add_argument("--skip-ci", action="store_true")
    main(ap.parse_args())
