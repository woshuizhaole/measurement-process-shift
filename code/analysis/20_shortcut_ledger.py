#!/usr/bin/env python
"""Stage 2 :: 捷径账本 (方案 §Stage2.1) —— 全量、GRU、双向、多种子

对每个 (特征集) × (方向 A->B, B->A) × (种子)，训练 GRU，报告院内(源域留出)与跨院的
AUROC/AUPRC/归一化Utility。定义并报告捷径指数 SI。门 G2 在此判定。

特征集: value_only, value_mask, mask_only。
阈值在源域 val 上选(使源域 val Utility 最大)，套用到院内 test 与跨院。
"""
import os, sys, json, time, argparse
import numpy as np, polars as pl

P = "."
sys.path.insert(0, f"{P}/code/utils")
import features as F
from trainer import Cache, train_eval
from models import normalized_utility, best_threshold
from boot import patient_bootstrap_ci, paired_bootstrap_diff
import torch

TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"
os.makedirs(TAB, exist_ok=True); os.makedirs(STT, exist_ok=True)


def run(args):
    t0 = time.time()
    dev = args.device
    X, M, D, Y, L = F.load_arrays()
    meta = F.load_meta()
    sp = F.load_splits()
    feats = ["value_only", "value_mask", "mask_only"]
    fmode = {"value_only": "value", "value_mask": "value_mask", "mask_only": "mask"}

    rows_out = []
    ci_rows = []
    for direction in args.dirs:
        src, tgt = ("A", "B") if direction == "A2B" else ("B", "A")
        norm = F.load_norm(src)
        tr_rows = np.array(sp[f"{src}_train"])
        va_rows = np.array(sp[f"{src}_val"])
        srcte_rows = np.array(sp[f"{src}_test"])
        tgt_rows = np.array(sp[f"{tgt}_test"])
        # 预建 caches (值通道用源域 norm)
        cache_kw = dict(src_norm=norm, device=dev, X=X, M=M, D=D, Y=Y, L=L, meta=meta, Tcap=args.tcap)
        c_tr = Cache(tr_rows, **cache_kw)
        c_va = Cache(va_rows, **cache_kw)
        c_st = Cache(srcte_rows, **cache_kw)
        c_tg = Cache(tgt_rows, **cache_kw)
        print(f"[{direction}] caches built ({time.time()-t0:.0f}s)", flush=True)

        for arch in args.archs:
            # 保存 value_only / value_mask 的跨院逐病人预测, 供配对 bootstrap CI
            xdir_preds = {}
            for feat in feats:
                for seed in range(args.seeds):
                    model, out = train_eval(
                        c_tr, {"val": c_va, "src_test": c_st, "tgt_test": c_tg},
                        feat_mode=fmode[feat], hidden=args.hidden, epochs=args.epochs,
                        lr=args.lr, bs=args.bs, pos_weight=args.pos_weight, seed=seed,
                        device=dev, arch=arch)
                    thr, _ = best_threshold(out["val"]["ys"], out["val"]["probs"])
                    rec = dict(direction=direction, arch=arch, feat=feat, seed=seed, thr=thr)
                    for split in ["src_test", "tgt_test"]:
                        o = out[split]
                        rec[f"{split}_auroc"] = o["auroc"]
                        rec[f"{split}_auprc"] = o["auprc"]
                        rec[f"{split}_utility"] = normalized_utility(o["ys"], o["probs"], thr)
                    rows_out.append(rec)
                    if feat in ("value_only", "value_mask") and seed == 0:
                        xdir_preds[feat] = (out["tgt_test"]["flat_y"], out["tgt_test"]["flat_p"],
                                            out["tgt_test"]["flat_g"])
                    print(f"  {direction} {arch:11s} {feat:11s} s{seed} "
                          f"in={rec['src_test_auroc']:.3f} cross={rec['tgt_test_auroc']:.3f} "
                          f"in_u={rec['src_test_utility']:.3f} cross_u={rec['tgt_test_utility']:.3f}",
                          flush=True)
                    del model
                    torch.cuda.empty_cache()
            # 配对 bootstrap CI: value_mask - value_only 的跨院 AUROC 差 (seed0)
            if "value_only" in xdir_preds and "value_mask" in xdir_preds:
                y0, p0, g0 = xdir_preds["value_only"]
                y1, p1, g1 = xdir_preds["value_mask"]
                ci = paired_bootstrap_diff(y0, p0, p1, g0, "auroc", B=2000, seed=0)
                ci_rows.append(dict(direction=direction, arch=arch,
                                    mask_cross_delta=ci["point"], lo=ci["lo"], hi=ci["hi"],
                                    p_gt0=ci["p_gt0"]))
                print(f"  [{direction} {arch}] mask cross ΔAUROC = {ci['point']:+.4f} "
                      f"[{ci['lo']:+.4f},{ci['hi']:+.4f}] p(>0)={ci['p_gt0']:.2f}", flush=True)

    df = pl.DataFrame(rows_out)
    tag = args.tag
    df.write_csv(f"{TAB}/stage2_shortcut_ledger{tag}.csv")
    pl.DataFrame(ci_rows).write_csv(f"{TAB}/stage2_ledger_ci{tag}.csv")
    if tag:   # 分片模式: 只写分片文件, 不做全局聚合/门(由 merge 脚本做)
        print(f"shard {tag} done, {len(rows_out)} rows, {time.time()-t0:.0f}s")
        return

    # ---- 聚合(按 arch×feat, 多种子 mean±std) + 门 G2 ----
    agg = (df.group_by(["direction", "arch", "feat"])
           .agg([pl.col(c).mean().alias(c) for c in
                 ["src_test_auroc", "tgt_test_auroc", "src_test_utility", "tgt_test_utility"]]
                + [pl.col("tgt_test_auroc").std().alias("tgt_auroc_std"),
                   pl.col("seed").count().alias("n_seed")])
           .sort(["direction", "arch", "feat"]))
    agg.write_csv(f"{TAB}/stage2_ledger_agg.csv")
    print("\n=== aggregated (arch×feat, multi-seed) ==="); print(agg.to_pandas().to_string(index=False))

    gates = {}
    for direction in ["A2B", "B2A"]:
        for arch in args.archs:
            dd = {r["feat"]: r for r in agg.filter((pl.col("direction") == direction) &
                                                   (pl.col("arch") == arch)).to_dicts()}
            vm, vo = dd["value_mask"], dd["value_only"]
            gates[f"{direction}_{arch}"] = dict(
                mask_in_gain=float(vm["src_test_auroc"] - vo["src_test_auroc"]),
                mask_cross_gain=float(vm["tgt_test_auroc"] - vo["tgt_test_auroc"]),
                in_util_gain=float(vm["src_test_utility"] - vo["src_test_utility"]),
                cross_util_gain=float(vm["tgt_test_utility"] - vo["tgt_test_utility"]),
                n_seed=int(vm["n_seed"]))
    ci_by = {f"{r['direction']}_{r['arch']}": r for r in ci_rows}
    for k in gates:
        if k in ci_by:
            gates[k]["cross_delta_ci"] = [ci_by[k]["lo"], ci_by[k]["hi"]]
            gates[k]["cross_delta_p_gt0"] = ci_by[k]["p_gt0"]
    # G2: 跨院 ΔAUROC<=-0.03 且 CI 上界<0 (任一 arch×方向); 全量下预期不通过
    g2_pass = any(g["mask_cross_gain"] <= -0.03 and g.get("cross_delta_ci", [0, 9])[1] < 0
                  for g in gates.values())
    gates["G2_threshold"] = dict(in_gain=">=0.01", cross_gain="<=-0.03 且 CI 上界<0")
    gates["G2_pass"] = bool(g2_pass)
    json.dump(gates, open(f"{STT}/stage2_gates.json", "w"), indent=2)
    print("\n=== GATES (G2) ==="); print(json.dumps(gates, indent=2))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos_weight", type=float, default=15.0)
    ap.add_argument("--tcap", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--archs", nargs="+", default=["gru", "transformer"])
    ap.add_argument("--dirs", nargs="+", default=["A2B", "B2A"])
    ap.add_argument("--tag", default="")     # 分片标签; 非空则只写分片文件
    run(ap.parse_args())
