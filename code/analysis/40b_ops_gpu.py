#!/usr/bin/env python
"""Stage 4 :: OPS-Repair 主实验 —— GRU/GPU backbone (避开 CPU 争抢; 方案 §7.2 双backbone之一)

用 GRUD(value_maskchan) 的 meas_override 喂不同的测量通道输入:
  B0 value_only : feat_mode='value' (纯值)
  B1 value_mask : meas_override=raw mask
  R1 reanchor   : meas_override=重锚 mask
  R2 msr        : meas_override=MSR残差
  R3 dann_msr   : R2 + 仅测量通道 GRL 对抗(6域中源3子域为环境)
  OPS_full      : R2 + IPW值通道 (+可选GRL)
倾向 e_j(C) 用 build_phi OOF(源)/full(目标)。阈值源域val选。官方Utility+AUROC+校准斜率+配对CI。
门 G5: OPS/R2 跨院 Utility 超 value_only >= +0.02 且配对CI下界>0。
"""
import os, sys, json, time, argparse
import numpy as np, polars as pl
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

P = "."
sys.path.insert(0, f"{P}/code/utils")
import features as F
from models import GRUD, normalized_utility, best_threshold
from dualchannel import DualChannel, grad_reverse
from ops_repair import MeasurementPropensity, msr, reanchor
from boot import paired_bootstrap_diff

TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"


def calib_slope(y, p):
    eps = 1e-6; lp = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    try:
        return float(sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1])
    except Exception:
        return np.nan


def build_arrays(rows, X, M, D, Y, L, srcnorm, cap, meta, mp_src=None, mp_h=None, ipw_models=None):
    """构造 padded 张量 + phi + 各测量通道变体。返回 dict of np arrays (N,T,·)."""
    N = len(rows); T = cap
    Xv = np.full((N, T, 34), 0.0, np.float32); Mk = np.zeros((N, T, 34), np.float32)
    Dl = np.zeros((N, T, 34), np.float32); Yy = np.zeros((N, T), np.float32)
    Ln = np.zeros(N, np.int32)
    Zc = F.ffill_impute(np.asarray(X[rows][:, :cap]), np.asarray(M[rows][:, :cap]), srcnorm, channels=F.TV)
    phi, _ = F.build_phi(X[rows], M[rows], L[rows], meta[list(rows)], src=None, _norm=srcnorm)
    for k, i in enumerate(rows):
        n = int(min(L[i], cap)); Ln[k] = n
        Xv[k, :n] = Zc[k, :n]; Mk[k, :n] = np.asarray(M[i, :n]); Dl[k, :n] = np.log1p(np.asarray(D[i, :n]))
        Yy[k, :n] = np.asarray(Y[i, :n])
    age = meta["age"].to_numpy()[list(rows)]; gen = meta["gender"].to_numpy()[list(rows)]
    st = np.stack([(age - np.nanmean(age)) / (np.nanstd(age) + 1e-8), gen], 1).astype(np.float32)
    out = dict(Xv=Xv, Mk=Mk, Dl=Dl, Yy=Yy, Ln=Ln, st=st, rows=np.asarray(rows))
    # 测量通道变体 (逐行 e, 需展平->还原)
    flat_ph = np.concatenate([phi[k, :Ln[k]] for k in range(N)])
    flat_m = np.concatenate([Mk[k, :Ln[k]] for k in range(N)]).astype(int)
    e_h = mp_h.predict_e(flat_ph) if mp_h is not None else None
    e_src = mp_src.predict_e(flat_ph) if mp_src is not None else None
    def scatter(flat):
        A = np.zeros((N, T, 34), np.float32); off = 0
        for k in range(N):
            A[k, :Ln[k]] = flat[off:off + Ln[k]]; off += Ln[k]
        return A
    if e_h is not None:
        out["MSR"] = scatter(msr(flat_m, e_h))
        if e_src is not None:
            out["REANCHOR"] = scatter(reanchor(flat_m, e_h, e_src))
    if ipw_models is not None:
        # IPW 值通道: 缺失处用 Ridge(C) 预测
        Vipw = Xv.copy()
        off = 0
        flat_v = np.concatenate([Xv[k, :Ln[k]] for k in range(N)])
        for j in range(34):
            if ipw_models[j] is None: continue
            miss = flat_m[:, j] == 0
            flat_v[miss, j] = ipw_models[j].predict(flat_ph[miss])
        out["Vipw"] = scatter(flat_v)
    return out


class DS:
    def __init__(s, d, dev, meas=None, use_ipw=False):
        s.x = torch.tensor(d["Vipw"] if use_ipw else d["Xv"], device=dev)
        s.m = torch.tensor(d["Mk"], device=dev); s.dl = torch.tensor(d["Dl"], device=dev)
        s.st = torch.tensor(d["st"], device=dev); s.y = torch.tensor(d["Yy"], device=dev)
        s.L = d["Ln"]; s.rows = d["rows"]
        s.meas = torch.tensor(d[meas], device=dev) if meas else None


def mbce(logit, y, L, pw):
    T = logit.shape[1]
    mk = (torch.arange(T, device=logit.device)[None] < torch.as_tensor(L, device=logit.device)[:, None])
    l = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none", pos_weight=pw)
    return (l * mk).sum() / mk.sum()


def train(method, tr, envs_tr, va, st, tg, dev, epochs=15, lr=1e-3, bs=256, pw=15.0, seed=0, lam=1.0):
    torch.manual_seed(seed); np.random.seed(seed)
    fmode = "value" if method == "value_only" else "value_mask"
    use_dann = method == "R3_dann_msr"
    nE = int(envs_tr.max()) + 1 if len(envs_tr) else 1
    if use_dann:
        model = DualChannel(grl_lambda=lam, n_hosp=nE).to(dev)
    else:
        model = GRUD(feat_mode=fmode, hidden=64).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    pwt = torch.tensor(pw, device=dev)
    envlab = torch.as_tensor(envs_tr, device=dev)
    for ep in range(epochs):
        model.train()
        idx = np.random.default_rng(seed + ep).permutation(len(tr.rows))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            opt.zero_grad()
            if use_dann:
                o = model(tr.x[b], tr.m[b], tr.dl[b], tr.st[b],
                          meas_in=tr.meas[b] if tr.meas is not None else None,
                          hosp=envlab[b])
                loss = mbce(o["logit"], tr.y[b], tr.L[b], pwt) + \
                    lam * nn.functional.cross_entropy(o["hosp_logit"], envlab[b])
            else:
                mi = tr.meas[b] if tr.meas is not None else None
                if fmode == "value":
                    logit = model(tr.x[b], tr.m[b], tr.dl[b], tr.st[b])
                else:
                    logit = model(tr.x[b], tr.m[b], tr.dl[b], tr.st[b], meas_override=mi)
                loss = mbce(logit, tr.y[b], tr.L[b], pwt)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()

    def ev(ds):
        model.eval()
        with torch.no_grad():
            probs, ys, fp, fy, fg = [], [], [], [], []
            for i in range(0, len(ds.rows), 512):
                sl = slice(i, i + 512)
                mi = ds.meas[sl] if ds.meas is not None else None
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
    return model, {k: ev(v) for k, v in [("va", va), ("st", st), ("tg", tg)]}


def main(args):
    t0 = time.time(); dev = args.device
    X, M, D, Y, L = F.load_arrays(); meta = F.load_meta(); sp = F.load_splits()
    dom = meta["domain"].to_numpy()
    from sklearn.linear_model import Ridge
    methods = ["value_only", "value_mask", "R1_reanchor", "R2_msr", "R3_dann_msr", "OPS_full"]
    rows_out = []; ci_rows = []
    for direction in args.dirs:
        src, tgt = ("A", "B") if direction == "A2B" else ("B", "A")
        srcnorm = F.load_norm(src)
        tr_r = np.array(sp[f"{src}_train"]); va_r = np.array(sp[f"{src}_val"])
        st_r = np.array(sp[f"{src}_test"]); tg_r = np.array(sp[f"{tgt}_test"])
        # 倾向: 源 OOF+full, 目标 full
        phi_tr, _ = F.build_phi(X[tr_r], M[tr_r], L[tr_r], meta[list(tr_r)], _norm=srcnorm)
        fph = np.concatenate([phi_tr[k, :int(min(L[tr_r[k]], args.tcap))] for k in range(len(tr_r))])
        Ztr = F.ffill_impute(np.asarray(X[tr_r][:, :args.tcap]), np.asarray(M[tr_r][:, :args.tcap]), srcnorm, channels=F.TV)
        lens = [int(min(L[tr_r[k]], args.tcap)) for k in range(len(tr_r))]
        fm = np.concatenate([np.asarray(M[tr_r[k], :lens[k]]) for k in range(len(tr_r))]).astype(int)
        fg = np.concatenate([np.full(lens[k], tr_r[k]) for k in range(len(tr_r))])
        fv = np.concatenate([Ztr[k, :lens[k]] for k in range(len(tr_r))])
        mp_src = MeasurementPropensity(); e_tr = mp_src.fit_transform_oof(fph, fm, fg, also_full=True)
        # IPW 值插补模型
        ipw = []
        for j in range(34):
            obs = fm[:, j] > 0
            if obs.sum() < 100: ipw.append(None); continue
            w = 1.0 / np.clip(e_tr[obs, j], 0.05, 1)
            ipw.append(Ridge(alpha=1.0).fit(fph[obs], fv[obs, j], sample_weight=w))
        # 目标倾向
        phi_tg, _ = F.build_phi(X[tg_r], M[tg_r], L[tg_r], meta[list(tg_r)], _norm=F.load_norm(tgt))
        ftm = np.concatenate([np.asarray(M[tg_r[k], :int(min(L[tg_r[k]], args.tcap))]) for k in range(len(tg_r))]).astype(int)
        ftph = np.concatenate([phi_tg[k, :int(min(L[tg_r[k]], args.tcap))] for k in range(len(tg_r))])
        mp_tgt = MeasurementPropensity().fit(ftph, ftm)
        print(f"[{direction}] propensity+ipw done ({time.time()-t0:.0f}s)", flush=True)

        # 构造各集数组 (含 MSR/REANCHOR/Vipw)
        A_tr = build_arrays(tr_r, X, M, D, Y, L, srcnorm, args.tcap, meta, mp_src, mp_src, ipw)
        A_va = build_arrays(va_r, X, M, D, Y, L, srcnorm, args.tcap, meta, mp_src, mp_src, ipw)
        A_st = build_arrays(st_r, X, M, D, Y, L, srcnorm, args.tcap, meta, mp_src, mp_src, ipw)
        A_tg = build_arrays(tg_r, X, M, D, Y, L, F.load_norm(tgt), args.tcap, meta, mp_src, mp_tgt, ipw)
        envs_tr = np.array([{d: i for i, d in enumerate(sorted(set(dom[tr_r])))}[dom[r]] for r in tr_r])
        print(f"[{direction}] arrays built ({time.time()-t0:.0f}s)", flush=True)

        vo_tg = None
        for method in methods:
            meas = None; use_ipw = False
            if method in ("R1_reanchor",): meas = "REANCHOR"
            if method in ("R2_msr", "R3_dann_msr"): meas = "MSR"
            if method == "OPS_full": meas = "MSR"; use_ipw = True
            for seed in range(args.seeds):
                tr = DS(A_tr, dev, meas, use_ipw); va = DS(A_va, dev, meas, use_ipw)
                st = DS(A_st, dev, meas, use_ipw); tg = DS(A_tg, dev, meas, use_ipw)
                _, out = train(method, tr, envs_tr, va, st, tg, dev, epochs=args.epochs,
                               lr=args.lr, bs=args.bs, seed=seed, lam=args.lam)
                thr, _ = best_threshold(out["va"]["ys"], out["va"]["probs"])
                rec = dict(direction=direction, method=method, seed=seed, thr=thr,
                           src_auroc=roc_auc_score(out["st"]["fy"], out["st"]["fp"]),
                           tgt_auroc=roc_auc_score(out["tg"]["fy"], out["tg"]["fp"]),
                           src_util=normalized_utility(out["st"]["ys"], out["st"]["probs"], thr),
                           tgt_util=normalized_utility(out["tg"]["ys"], out["tg"]["probs"], thr),
                           tgt_calib=calib_slope(out["tg"]["fy"], out["tg"]["fp"]))
                rows_out.append(rec)
                if method == "value_only" and seed == 0:
                    vo_tg = (out["tg"]["fy"], out["tg"]["fp"], out["tg"]["fg"])
                if method in ("R2_msr", "OPS_full", "value_mask") and seed == 0 and vo_tg is not None:
                    ci = paired_bootstrap_diff(vo_tg[0], vo_tg[1], out["tg"]["fp"], vo_tg[2], "auroc", B=500, seed=0)
                    ci_rows.append(dict(direction=direction, method=method, auroc_delta_vs_vo=round(ci["point"], 4),
                                        lo=round(ci["lo"], 4), hi=round(ci["hi"], 4), p_gt0=round(ci["p_gt0"], 3)))
                print(f"  {direction} {method:12s} s{seed} src_u={rec['src_util']:.3f} tgt_u={rec['tgt_util']:.3f} "
                      f"tgt_auroc={rec['tgt_auroc']:.3f} calib={rec['tgt_calib']:.2f}", flush=True)
                torch.cuda.empty_cache()

    df = pl.DataFrame(rows_out); df.write_csv(f"{TAB}/stage4_ops_main{args.tag}.csv")
    pl.DataFrame(ci_rows).write_csv(f"{TAB}/stage4_ops_ci{args.tag}.csv")
    if not args.tag:
        finalize(df)
    print(f"total {time.time()-t0:.0f}s")


def finalize(df):
    agg = (df.group_by(["direction", "method"]).agg(
        [pl.col(c).mean() for c in ["src_auroc", "tgt_auroc", "src_util", "tgt_util", "tgt_calib"]]
        + [pl.col("tgt_util").std().alias("tgt_util_std")]).sort(["direction", "method"]))
    agg.write_csv(f"{TAB}/stage4_ops_agg.csv")
    print("\n=== OPS main aggregated ==="); print(agg.to_pandas().to_string(index=False))
    gates = {}
    for direction in ["A2B", "B2A"]:
        dd = {r["method"]: r for r in agg.filter(pl.col("direction") == direction).to_dicts()}
        if "value_only" not in dd: continue
        vo = dd["value_only"]; vm = dd.get("value_mask")
        for meth in dd:
            if meth == "value_only": continue
            gates[f"{direction}_{meth}"] = dict(
                util_gain_vs_vo=round(dd[meth]["tgt_util"] - vo["tgt_util"], 4),
                util_gain_vs_value_mask=round(dd[meth]["tgt_util"] - vm["tgt_util"], 4) if vm else None,
                auroc_gain_vs_vo=round(dd[meth]["tgt_auroc"] - vo["tgt_auroc"], 4),
                calib=round(dd[meth]["tgt_calib"], 3))
    # 诚实判定 (写进代码, 避免手工注解被重跑覆盖 —— 自检 caveat 2)
    r2_beats_vo = all(gates.get(f"{d}_R2_msr", {}).get("util_gain_vs_vo", -9) >= 0.02 for d in ["A2B", "B2A"])
    any_beats_vm = any(gates.get(f"{d}_{m}", {}).get("util_gain_vs_value_mask", -9) is not None and
                       gates[f"{d}_{m}"]["util_gain_vs_value_mask"] >= 0.02
                       for d in ["A2B", "B2A"] for m in ["OPS_full", "R2_msr", "R1_reanchor", "R3_dann_msr"])
    both_dir_beats_vm = any(all(gates.get(f"{d}_{m}", {}).get("util_gain_vs_value_mask", -9) >= 0.02
                                for d in ["A2B", "B2A"]) for m in ["OPS_full", "R2_msr", "R1_reanchor", "R3_dann_msr"])
    gates["G5_threshold"] = "预注册主要终点: OPS/R2 跨院 Utility 超 value_only >= +0.02, 且打赢 value_mask, 双向"
    gates["G5_technical_pass_vs_value_only"] = bool(any(v.get("util_gain_vs_vo", -9) >= 0.02
                                                        for k, v in gates.items() if isinstance(v, dict) and ("OPS_full" in k or "R2_msr" in k)))
    gates["R2_msr_beats_value_only_both_dir"] = bool(r2_beats_vo)
    gates["any_method_beats_value_mask_some_dir"] = bool(any_beats_vm)
    gates["any_method_beats_value_mask_both_dir"] = bool(both_dir_beats_vm)
    gates["G5_pass_prereg"] = bool(both_dir_beats_vm)   # 预注册严格判据: 双向打赢 value_mask
    gates["HONEST_VERDICT"] = ("纯 MSR 校正(R2)未双向打赢 value_only(CI含0); 无方法双向稳定超越 value_mask(简单用mask). "
                               "OPS_full 对 value_only 技术通过但增益主要来自 IPW 值插补, 非测量通道校正. "
                               "符合 Stage2(测量占跨院缺口~20%)/Stage3(gamma多共享但非主导)预期. 触发预注册 Plan-B: 主结论=诊断+边界, 非修复胜利.")
    json.dump(gates, open(f"{STT}/stage4_gates.json", "w"), indent=2, ensure_ascii=False)
    print("\n=== GATES (G5) ==="); print(json.dumps(gates, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--tcap", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dirs", nargs="+", default=["A2B", "B2A"]); ap.add_argument("--tag", default="")
    main(ap.parse_args())
