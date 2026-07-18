#!/usr/bin/env python
"""Stage 0 :: 病人级划分 + 源域归一化统计 (方案 §Stage0.3)

- 划分只按病人, 按 SepsisLabel 分层, 固定 seed, 持久化 id 列表
- 源域 train/val = 80/20; 目标域整体留作 (无标签 mask) + (带标签测试)
- 归一化统计: 每院各算一份 (winsorize [0.1,99.9] pct + z-score, 用 OBSERVED 值)
  训练时按"源域"套用; 绝不在目标域上计算任何"值"统计量。
输出:
  data/processed/splits.json          {A_train,A_val,A_test,B_train,B_val,B_test} -> patient_id 列表
  data/processed/norm_stats_A.json    每通道 {mean,std,median,iqr,p01,p999} (仅 A 训练集 observed)
  data/processed/norm_stats_B.json    同上 (仅 B 训练集 observed)
"""
import json
import numpy as np, polars as pl
from sklearn.model_selection import train_test_split

P = "."
OUT = f"{P}/data/processed"
ARR = f"{OUT}/arrays"
SEED = 20260710

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
        "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
        "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
        "PTT", "WBC", "Fibrinogen", "Platelets"]
TV = VITALS + LABS


def main():
    meta = pl.read_parquet(f"{OUT}/meta.parquet")
    idx = np.arange(meta.height)
    hosp = meta["hospital"].to_numpy()
    ever = meta["sepsis_ever"].to_numpy()
    pid = meta["patient_id"].to_numpy()

    splits = {}
    for h in ["A", "B"]:
        sel = np.where(hosp == h)[0]
        # 源域用途: train/val/test = 64/16/20; 目标域用途: 整体作 test, 但也切出同样比例便于"内部对照"
        tr, tmp = train_test_split(sel, test_size=0.36, random_state=SEED, stratify=ever[sel])
        va, te = train_test_split(tmp, test_size=20.0 / 36.0, random_state=SEED, stratify=ever[tmp])
        splits[f"{h}_train"] = pid[tr].tolist()
        splits[f"{h}_val"] = pid[va].tolist()
        splits[f"{h}_test"] = pid[te].tolist()
        print(f"{h}: train {len(tr)} val {len(va)} test {len(te)}  "
              f"(prev train {ever[tr].mean():.4f} val {ever[va].mean():.4f} test {ever[te].mean():.4f})")

    with open(f"{OUT}/splits.json", "w") as f:
        json.dump(splits, f)
    # 索引版本（供数组按行取用）
    pid2row = {p: i for i, p in enumerate(pid)}
    splits_idx = {k: [pid2row[p] for p in v] for k, v in splits.items()}
    with open(f"{OUT}/splits_idx.json", "w") as f:
        json.dump(splits_idx, f)

    # ---------- 归一化统计: 每院 train 集 observed 值 ----------
    X = np.load(f"{ARR}/X_raw.npy", mmap_mode="r")
    M = np.load(f"{ARR}/M.npy", mmap_mode="r")
    lengths = np.load(f"{ARR}/lengths.npy")
    T_CAP = X.shape[1]
    valid = np.arange(T_CAP)[None, :] < lengths[:, None]

    for h in ["A", "B"]:
        rows = np.array(splits_idx[f"{h}_train"])
        stats = {}
        for j, name in enumerate(TV):
            m = M[rows, :, j].astype(bool) & valid[rows]
            vals = X[rows, :, j][m]
            vals = vals[np.isfinite(vals)]
            if vals.size < 10:
                stats[name] = dict(n=int(vals.size), mean=0.0, std=1.0, median=0.0,
                                   iqr=1.0, p01=0.0, p999=1.0, observed_frac=float(m.sum() / valid[rows].sum()))
                continue
            p01, p999 = np.percentile(vals, [0.1, 99.9])
            clipped = np.clip(vals, p01, p999)
            q1, q3 = np.percentile(vals, [25, 75])
            stats[name] = dict(
                n=int(vals.size),
                mean=float(clipped.mean()), std=float(clipped.std() + 1e-8),
                median=float(np.median(vals)), iqr=float(q3 - q1 + 1e-8),
                p01=float(p01), p999=float(p999),
                observed_frac=float(m.sum() / valid[rows].sum()),
            )
        with open(f"{OUT}/norm_stats_{h}.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"norm_stats_{h}: {len(stats)} channels written")

    print("DONE splits + norm")


if __name__ == "__main__":
    main()
