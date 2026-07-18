#!/usr/bin/env python
"""Stage 4 :: φ 敏感性 (自检 caveat 1) —— 用富化 φ(rich=含观测强度史 + 逐变量滞后) 重跑 MSR

检验负结论(R2_msr 打不过 value_only/value_mask)在更完整的倾向模型 φ 下是否依然成立。
只跑 value_only / value_mask / R2_msr, GRU/GPU, 双向, 3种子。
"""
import os, sys, json, time, argparse
import numpy as np, polars as pl
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

P = "."
sys.path.insert(0, f"{P}/code/utils")
import features as F
from models import GRUD, normalized_utility, best_threshold
from ops_repair import MeasurementPropensity, msr

TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"


def flat_build(rows, X, M, D, Y, L, srcnorm, cap, meta, mp=None, lagfn=None):
    """构造 padded 张量 + rich phi + (可选)MSR 测量通道。返回 dict。"""
    N = len(rows); T = cap
    Xv = np.zeros((N, T, 34), np.float32); Mk = np.zeros((N, T, 34), np.float32)
    Dl = np.zeros((N, T, 34), np.float32); Yy = np.zeros((N, T), np.float32); Ln = np.zeros(N, np.int32)
    Zc = F.ffill_impute(np.asarray(X[rows][:, :cap]), np.asarray(M[rows][:, :cap]), srcnorm, channels=F.TV)
    phi, _ = F.build_phi(X[rows], M[rows], L[rows], meta[list(rows)], _norm=srcnorm, rich=True)
    for k, i in enumerate(rows):
        n = int(min(L[i], cap)); Ln[k] = n
        Xv[k, :n] = Zc[k, :n]; Mk[k, :n] = np.asarray(M[i, :n])
        Dl[k, :n] = np.log1p(np.asarray(D[i, :n])); Yy[k, :n] = np.asarray(Y[i, :n])
    age = meta["age"].to_numpy()[list(rows)]; gen = meta["gender"].to_numpy()[list(rows)]
    st = np.stack([(age - np.nanmean(age)) / (np.nanstd(age) + 1e-8), gen], 1).astype(np.float32)
    out = dict(Xv=Xv, Mk=Mk, Dl=Dl, Yy=Yy, Ln=Ln, st=st, rows=np.asarray(rows))
    # 展平 phi/mask/lag(=log1p delta 逐变量)
    fph = np.concatenate([phi[k, :Ln[k]] for k in range(N)])
    fm = np.concatenate([Mk[k, :Ln[k]] for k in range(N)]).astype(int)
    flag = np.concatenate([Dl[k, :Ln[k]] for k in range(N)])   # 逐变量 log1p(距上次观测)
    fg = np.concatenate([np.full(Ln[k], rows[k]) for k in range(N)])
    if mp is not None:
        e = mp.predict_e(fph, lag=flag)
        R = msr(fm, e)
        # scatter 回 (N,T,34)
        A = np.zeros((N, T, 34), np.float32); off = 0
        for k in range(N):
            A[k, :Ln[k]] = R[off:off + Ln[k]]; off += Ln[k]
        out["MSR"] = A
    out["_flat"] = dict(ph=fph, m=fm, lag=flag, g=fg)
    return out


class DS:
    def __init__(s, d, dev, meas=None):
        s.x = torch.tensor(d["Xv"], device=dev); s.m = torch.tensor(d["Mk"], device=dev)
        s.dl = torch.tensor(d["Dl"], device=dev); s.st = torch.tensor(d["st"], device=dev)
        s.y = torch.tensor(d["Yy"], device=dev); s.L = d["Ln"]; s.rows = d["rows"]
        s.meas = torch.tensor(d[meas], device=dev) if meas else None


def mbce(logit, y, L, pw):
    T = logit.shape[1]
    mk = (torch.arange(T, device=logit.device)[None] < torch.as_tensor(L, device=logit.device)[:, None])
    return (nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none", pos_weight=pw) * mk).sum() / mk.sum()


def train(method, tr, va, st, tg, dev, epochs=15, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    fmode = "value" if method == "value_only" else "value_mask"
    model = GRUD(feat_mode=fmode, hidden=64).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5); pw = torch.tensor(15.0, device=dev)
    for ep in range(epochs):
        model.train(); idx = np.random.default_rng(seed + ep).permutation(len(tr.rows))
        for i in range(0, len(idx), 256):
            b = idx[i:i + 256]; opt.zero_grad()
            mi = tr.meas[b] if tr.meas is not None else None
            logit = model(tr.x[b], tr.m[b], tr.dl[b], tr.st[b]) if fmode == "value" else \
                model(tr.x[b], tr.m[b], tr.dl[b], tr.st[b], meas_override=mi)
            loss = mbce(logit, tr.y[b], tr.L[b], pw); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()

    def ev(ds):
        model.eval()
        with torch.no_grad():
            probs, ys, fp, fy = [], [], [], []
            for i in range(0, len(ds.rows), 512):
                sl = slice(i, i + 512); mi = ds.meas[sl] if ds.meas is not None else None
                lg = model(ds.x[sl], ds.m[sl], ds.dl[sl], ds.st[sl]) if ds.meas is None else \
                    model(ds.x[sl], ds.m[sl], ds.dl[sl], ds.st[sl], meas_override=mi)
                pr = torch.sigmoid(lg).cpu().numpy()
                for k in range(pr.shape[0]):
                    n = int(ds.L[i + k]); probs.append(pr[k, :n]); ys.append(ds.y[i + k, :n].cpu().numpy().astype(int))
                    fp.append(pr[k, :n]); fy.append(ds.y[i + k, :n].cpu().numpy().astype(int))
            return dict(probs=probs, ys=ys, fp=np.concatenate(fp), fy=np.concatenate(fy))
    return {k: ev(v) for k, v in [("va", va), ("st", st), ("tg", tg)]}


def main(args):
    t0 = time.time(); dev = args.device
    X, M, D, Y, L = F.load_arrays(); meta = F.load_meta(); sp = F.load_splits()
    rows_out = []
    for direction in args.dirs:
        src, tgt = ("A", "B") if direction == "A2B" else ("B", "A")
        srcnorm = F.load_norm(src)
        tr_r = np.array(sp[f"{src}_train"]); va_r = np.array(sp[f"{src}_val"])
        st_r = np.array(sp[f"{src}_test"]); tg_r = np.array(sp[f"{tgt}_test"])
        # rich 倾向: OOF + 逐变量 lag
        d0 = flat_build(tr_r, X, M, D, Y, L, srcnorm, args.tcap, meta)
        fl = d0["_flat"]
        mp = MeasurementPropensity()
        mp.fit_transform_oof(fl["ph"], fl["m"], fl["g"], also_full=True, lag=fl["lag"])
        print(f"[{direction}] rich propensity (phi_dim={fl['ph'].shape[1]}+lag) done ({time.time()-t0:.0f}s)", flush=True)
        A_tr = flat_build(tr_r, X, M, D, Y, L, srcnorm, args.tcap, meta, mp)
        A_va = flat_build(va_r, X, M, D, Y, L, srcnorm, args.tcap, meta, mp)
        A_st = flat_build(st_r, X, M, D, Y, L, srcnorm, args.tcap, meta, mp)
        A_tg = flat_build(tg_r, X, M, D, Y, L, F.load_norm(tgt), args.tcap, meta, mp)
        for method in ["value_only", "value_mask", "R2_msr"]:
            meas = "MSR" if method == "R2_msr" else None
            for seed in range(args.seeds):
                tr = DS(A_tr, dev, meas); va = DS(A_va, dev, meas); st = DS(A_st, dev, meas); tg = DS(A_tg, dev, meas)
                out = train(method, tr, va, st, tg, dev, epochs=args.epochs, seed=seed)
                thr, _ = best_threshold(out["va"]["ys"], out["va"]["probs"])
                rows_out.append(dict(direction=direction, method=method, seed=seed,
                                     tgt_auroc=roc_auc_score(out["tg"]["fy"], out["tg"]["fp"]),
                                     tgt_util=normalized_utility(out["tg"]["ys"], out["tg"]["probs"], thr)))
                print(f"  {direction} {method:11s} s{seed} tgt_u={rows_out[-1]['tgt_util']:.3f} "
                      f"tgt_auroc={rows_out[-1]['tgt_auroc']:.3f}", flush=True)
                torch.cuda.empty_cache()
    df = pl.DataFrame(rows_out); df.write_csv(f"{TAB}/stage4_phi_sensitivity{args.tag}.csv")
    if not args.tag:
        agg = df.group_by(["direction", "method"]).agg([pl.col("tgt_util").mean(), pl.col("tgt_auroc").mean()]).sort(["direction", "method"])
        print("\n=== rich-phi 敏感性 ==="); print(agg.to_pandas().to_string(index=False))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--tcap", type=int, default=200); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dirs", nargs="+", default=["A2B", "B2A"]); ap.add_argument("--tag", default="")
    main(ap.parse_args())
