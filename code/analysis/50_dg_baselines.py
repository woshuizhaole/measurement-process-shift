#!/usr/bin/env python
"""Stage 5 :: 域泛化/域适应基线大盘 (方案 §Stage5, §7.3-7.4)

在 GRU backbone 上跑 DomainBed 系 DG 算法 + on-thesis DA 基线, 匹配调参预算,
证明"通用 DG 关不上跨院缺口, 且 value-only(丢mask) 仍具竞争力"(ClinicalDG 的诅咒)。

方法(GRU value+mask backbone, 6域=A/B×MICU/SICU/unknown 作 environment):
  ERM, IRM, VREx, GroupDRO, CORAL, DANN  (DomainBed 系)
  + on-thesis: value_only(drop mask), mask_dropout_aug, DAMS_reweight, importance_weight, TENT/AdaBN
双向 A2B/B2A(源域用其3个子域作environment), 多种子, 源域选模, 官方 Utility + CI。
"""
import os, sys, json, time, argparse
import numpy as np, polars as pl
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

P = "."
sys.path.insert(0, f"{P}/code/utils")
import features as F
from trainer import Cache
from models import GRUD, normalized_utility, best_threshold
from dualchannel import grad_reverse
from boot import paired_bootstrap_diff

TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"


def masked_bce(logit, y, L, pw):
    T = logit.shape[1]
    mk = (torch.arange(T, device=logit.device)[None, :] < torch.as_tensor(L, device=logit.device)[:, None])
    l = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none", pos_weight=pw)
    return (l * mk).sum() / mk.sum()


def train_dg(algo, c_tr, envs, c_va, c_st, c_tg, feat_mode="value_mask", hidden=64,
             epochs=15, lr=1e-3, bs=256, pw=15.0, seed=0, device="cuda:0", lam=1.0):
    """envs: list of 病人环境索引(6域中源域的3个子域). algo in ERM/IRM/VREx/GroupDRO/CORAL/DANN."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = GRUD(feat_mode=feat_mode, hidden=hidden).to(device)
    disc = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, len(envs))).to(device) if algo == "DANN" else None
    # 简化: 直接用整个源域 cache, environment 标签用于 IRM/VREx/GroupDRO/DANN 的分组
    x, m, d, st, y = [c_tr.x.to(device), c_tr.m.to(device), c_tr.d.to(device),
                      c_tr.static.to(device), c_tr.y.to(device)]
    envlab = torch.as_tensor(c_tr.envlab, device=device) if hasattr(c_tr, "envlab") else torch.zeros(len(c_tr.rows), device=device, dtype=torch.long)
    params = list(model.parameters()) + (list(disc.parameters()) if disc else [])
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    pwt = torch.tensor(pw, device=device)
    nE = len(envs)
    for ep in range(epochs):
        model.train()
        idx = np.random.default_rng(seed + ep).permutation(len(c_tr.rows))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            opt.zero_grad()
            # GRUD.forward: (x_hat,mask,delta,static)
            gout = model.gru if False else None
            logit = model(x[b], m[b], d[b], st[b])
            base = masked_bce(logit, y[b], c_tr.L[b], pwt)
            loss = base
            eb = envlab[b]
            if algo in ("IRM", "VREx", "GroupDRO"):
                # 逐环境 loss
                penalties = []
                elosses = []
                Lb = c_tr.L[b]
                for e in range(nE):
                    sel = (eb == e)
                    if sel.sum() < 2:
                        continue
                    selc = sel.detach().cpu().numpy()
                    le = masked_bce(logit[sel], y[b][sel], Lb[selc], pwt)
                    elosses.append(le)
                    if algo == "IRM":
                        # IRM penalty: grad wrt dummy scale
                        scale = torch.tensor(1.0, device=device, requires_grad=True)
                        lg = masked_bce(logit[sel] * scale, y[b][sel], Lb[selc], pwt)
                        gpen = torch.autograd.grad(lg, [scale], create_graph=True)[0]
                        penalties.append((gpen ** 2))
                if algo == "IRM" and penalties:
                    loss = base + lam * torch.stack(penalties).mean()
                elif algo == "VREx" and len(elosses) > 1:
                    loss = base + lam * torch.stack(elosses).var()
                elif algo == "GroupDRO" and elosses:
                    loss = torch.stack(elosses).max()
            elif algo == "DANN":
                h = model_last_hidden(model, x[b], m[b], d[b], st[b])
                dl = disc(grad_reverse(h, lam))
                loss = base + nn.functional.cross_entropy(dl, eb)
            elif algo == "CORAL" and nE > 1:
                h = model_last_hidden(model, x[b], m[b], d[b], st[b])
                cpen = 0.0; cnt = 0
                for e1 in range(nE):
                    for e2 in range(e1 + 1, nE):
                        s1, s2 = (eb == e1), (eb == e2)
                        if s1.sum() < 2 or s2.sum() < 2:
                            continue
                        cpen = cpen + coral(h[s1], h[s2]); cnt += 1
                if cnt:
                    loss = base + lam * cpen / cnt
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
    return model


def model_last_hidden(model, x, m, d, st):
    B, T, _ = m.shape
    g = torch.exp(-torch.relu(model.decay(d)))
    x_dec = m * x + (1 - m) * (g * x)
    parts = [x_dec, m, d, st.unsqueeze(1).expand(-1, T, -1)]
    z = torch.cat(parts, -1)
    h, _ = model.gru(z)
    return h.mean(1)


def coral(h1, h2):
    c1 = torch.cov(h1.T); c2 = torch.cov(h2.T)
    return ((c1 - c2) ** 2).sum() / (4 * h1.shape[1] ** 2)


def eval_model(model, ec, device, thr=None):
    model.eval()
    with torch.no_grad():
        ex, em, ed, es = ec.x.to(device), ec.m.to(device), ec.d.to(device), ec.static.to(device)
        probs, ys, fp, fy, fg = [], [], [], [], []
        for i in range(0, len(ec.rows), 512):
            sl = slice(i, i + 512)
            lg = model(ex[sl], em[sl], ed[sl], es[sl])
            pr = torch.sigmoid(lg).cpu().numpy()
            for k in range(pr.shape[0]):
                n = int(ec.L[i + k])
                probs.append(pr[k, :n]); ys.append(ec.y[i + k, :n].numpy().astype(int))
                fp.append(pr[k, :n]); fy.append(ec.y[i + k, :n].numpy().astype(int)); fg.append(np.full(n, ec.rows[i + k]))
    fp = np.concatenate(fp); fy = np.concatenate(fy); fg = np.concatenate(fg)
    return dict(probs=probs, ys=ys, flat_p=fp, flat_y=fy, flat_g=fg, auroc=roc_auc_score(fy, fp))


def main(args):
    t0 = time.time()
    dev = args.device
    X, M, D, Y, L = F.load_arrays(); meta = F.load_meta(); sp = F.load_splits()
    dom = meta["domain"].to_numpy()
    algos = args.algos
    rows_out = []
    for direction in args.dirs:
        src, tgt = ("A", "B") if direction == "A2B" else ("B", "A")
        norm = F.load_norm(src)
        kw = dict(src_norm=norm, device=dev, X=X, M=M, D=D, Y=Y, L=L, meta=meta, Tcap=args.tcap)
        tr_rows = np.array(sp[f"{src}_train"])
        c_tr = Cache(tr_rows, **kw)
        # environment 标签: 源域3子域
        src_doms = sorted(set(dom[tr_rows]))
        dmap = {dd: i for i, dd in enumerate(src_doms)}
        c_tr.envlab = np.array([dmap[dom[r]] for r in tr_rows])
        envs = list(range(len(src_doms)))
        c_va = Cache(np.array(sp[f"{src}_val"]), **kw)
        c_st = Cache(np.array(sp[f"{src}_test"]), **kw)
        c_tg = Cache(np.array(sp[f"{tgt}_test"]), **kw)
        print(f"[{direction}] envs={len(envs)} built ({time.time()-t0:.0f}s)", flush=True)
        for algo in algos:
            fm = "value" if algo == "value_only" else "value_mask"
            a = "ERM" if algo == "value_only" else algo
            for seed in range(args.seeds):
                model = train_dg(a, c_tr, envs, c_va, c_st, c_tg, feat_mode=fm, hidden=args.hidden,
                                 epochs=args.epochs, lr=args.lr, bs=args.bs, seed=seed, device=dev, lam=args.lam)
                ov = eval_model(model, c_va, dev)
                thr, _ = best_threshold(ov["ys"], ov["probs"])
                ost = eval_model(model, c_st, dev); otg = eval_model(model, c_tg, dev)
                rows_out.append(dict(direction=direction, algo=algo, seed=seed,
                                     src_auroc=ost["auroc"], tgt_auroc=otg["auroc"],
                                     src_util=normalized_utility(ost["ys"], ost["probs"], thr),
                                     tgt_util=normalized_utility(otg["ys"], otg["probs"], thr)))
                print(f"  {direction} {algo:12s} s{seed} src_u={rows_out[-1]['src_util']:.3f} "
                      f"tgt_u={rows_out[-1]['tgt_util']:.3f} tgt_auroc={otg['auroc']:.3f}", flush=True)
                del model; torch.cuda.empty_cache()
    df = pl.DataFrame(rows_out); df.write_csv(f"{TAB}/stage5_dg_baselines{args.tag}.csv")
    if not args.tag:
        agg = (df.group_by(["direction", "algo"]).agg([pl.col(c).mean() for c in
               ["src_auroc", "tgt_auroc", "src_util", "tgt_util"]]).sort(["direction", "algo"]))
        agg.write_csv(f"{TAB}/stage5_dg_agg.csv")
        print("\n=== DG baselines aggregated ==="); print(agg.to_pandas().to_string(index=False))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+",
                    default=["value_only", "ERM", "IRM", "VREx", "GroupDRO", "CORAL", "DANN"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tcap", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dirs", nargs="+", default=["A2B", "B2A"])
    ap.add_argument("--tag", default="")
    main(ap.parse_args())
