#!/usr/bin/env python
"""Stage 2 :: 病人级配对 bootstrap CI (value_mask - value_only 跨院 AUROC)。
单独小任务: 每 (dir,arch) 重训 value_only+value_mask(seed0), 保存跨院逐病人预测, 快速 CI。
高效 bootstrap: 预建 patient->rows, B=1000, 病人级重采样。"""
import os, sys, json, time
import numpy as np, polars as pl
import torch

P = "."
sys.path.insert(0, f"{P}/code/utils")
import features as F
from trainer import Cache, train_eval
from boot import paired_bootstrap_diff

TAB = f"{P}/results/tables"


def main():
    t0 = time.time()
    dev = "cuda:0"
    X, M, D, Y, L = F.load_arrays()
    meta = F.load_meta(); sp = F.load_splits()
    rows = []
    for direction in ["A2B", "B2A"]:
        src, tgt = ("A", "B") if direction == "A2B" else ("B", "A")
        norm = F.load_norm(src)
        kw = dict(src_norm=norm, device=dev, X=X, M=M, D=D, Y=Y, L=L, meta=meta, Tcap=200)
        c_tr = Cache(np.array(sp[f"{src}_train"]), **kw)
        c_va = Cache(np.array(sp[f"{src}_val"]), **kw)
        c_tg = Cache(np.array(sp[f"{tgt}_test"]), **kw)
        for arch in ["gru", "transformer"]:
            preds = {}
            for feat, fm in [("value_only", "value"), ("value_mask", "value_mask")]:
                _, out = train_eval(c_tr, {"tgt": c_tg}, feat_mode=fm, hidden=64, epochs=15,
                                    lr=1e-3, bs=256, pos_weight=15.0, seed=0, device=dev, arch=arch)
                preds[feat] = (out["tgt"]["flat_y"], out["tgt"]["flat_p"], out["tgt"]["flat_g"])
                torch.cuda.empty_cache()
            y0, p0, g0 = preds["value_only"]; _, p1, _ = preds["value_mask"]
            ci = paired_bootstrap_diff(y0, p0, p1, g0, "auroc", B=500, seed=0)
            rows.append(dict(direction=direction, arch=arch, mask_cross_delta=round(ci["point"], 4),
                             lo=round(ci["lo"], 4), hi=round(ci["hi"], 4), p_gt0=round(ci["p_gt0"], 3)))
            print(f"[{direction} {arch}] mask cross ΔAUROC={ci['point']:+.4f} "
                  f"[{ci['lo']:+.4f},{ci['hi']:+.4f}] p(>0)={ci['p_gt0']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    pl.DataFrame(rows).write_csv(f"{TAB}/stage2_ledger_ci.csv")
    print("\n=== CI 表 ==="); print(pl.DataFrame(rows).to_pandas().to_string(index=False))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
