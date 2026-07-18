#!/usr/bin/env python
"""Stage 0 :: CinC2019 全量预处理 (方案 §Stage0.3)  —— 稳健流式版

修复: polars 多线程运行时在 ProcessPoolExecutor(fork) 后死锁。
改为 numpy.genfromtxt 读取 + ThreadPoolExecutor 做 IO + 主进程流式写 memmap。

输入 : data/raw/cinc2019/training/training_set{A,B}/*.psv   (20336 + 20000)
输出 :
  data/processed/arrays/{X_raw,M,delta,y,lengths}.npy   (N, T=336, 34)
  data/processed/meta.parquet
  data/processed/qc_stage0.json
"""
import os, json, glob, time
import numpy as np
import polars as pl
from concurrent.futures import ThreadPoolExecutor

P = "."
RAW = f"{P}/data/raw/cinc2019/training"
OUT = f"{P}/data/processed"
ARR = f"{OUT}/arrays"
os.makedirs(ARR, exist_ok=True)

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
        "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
        "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
        "PTT", "WBC", "Fibrinogen", "Platelets"]
TV = VITALS + LABS
STATIC = ["Age", "Gender"]
ADMIN = ["Unit1", "Unit2", "HospAdmTime", "ICULOS"]
COLS41 = TV + STATIC + ADMIN + ["SepsisLabel"]
COL_IDX = {c: i for i, c in enumerate(COLS41)}
TV_COLS = [COL_IDX[c] for c in TV]
T_CAP = 336


def read_one(path):
    """返回 (n, 41) float32 数组，NaN 保留。纯 numpy，无 polars。"""
    arr = np.genfromtxt(path, delimiter="|", skip_header=1, dtype=np.float32,
                        filling_values=np.nan, missing_values="NaN")
    if arr.ndim == 1:                       # 只有一行
        arr = arr[None, :]
    return arr


def main():
    t0 = time.time()
    fa = sorted(glob.glob(f"{RAW}/training_setA/*.psv"))
    fb = sorted(glob.glob(f"{RAW}/training_setB/*.psv"))
    files = [(f, "A") for f in fa] + [(f, "B") for f in fb]
    print(f"files: {len(files)} (A={len(fa)}, B={len(fb)})", flush=True)
    assert len(fa) == 20336 and len(fb) == 20000, (len(fa), len(fb))

    hdr = open(files[0][0]).readline().strip().split("|")
    assert hdr == COLS41, f"column order mismatch:\n{hdr}"
    print("[OK] 41 columns in expected order", flush=True)

    N = len(files)
    X = np.lib.format.open_memmap(f"{ARR}/X_raw.npy", mode="w+", dtype=np.float32, shape=(N, T_CAP, 34))
    M = np.lib.format.open_memmap(f"{ARR}/M.npy", mode="w+", dtype=np.uint8, shape=(N, T_CAP, 34))
    D = np.lib.format.open_memmap(f"{ARR}/delta.npy", mode="w+", dtype=np.float32, shape=(N, T_CAP, 34))
    Y = np.lib.format.open_memmap(f"{ARR}/y.npy", mode="w+", dtype=np.uint8, shape=(N, T_CAP))
    X[:] = np.nan
    M[:] = 0
    D[:] = 0
    Y[:] = 0

    lengths = np.zeros(N, dtype=np.int32)
    meta = []
    unit_bad = unit_varying = iculos_bad = 0

    # 线程池读文件（IO-bound；genfromtxt 释放 GIL 期间可并行），主线程写 memmap
    with ThreadPoolExecutor(max_workers=32) as ex:
        for i, (arr, (path, hosp)) in enumerate(zip(ex.map(read_one, [f for f, _ in files]), files)):
            n = arr.shape[0]
            lengths[i] = n
            if n > T_CAP:
                arr = arr[:T_CAP]
                n = T_CAP
            vals = arr[:, TV_COLS]                    # (n,34)
            m = (~np.isnan(vals)).astype(np.uint8)
            X[i, :n] = vals
            M[i, :n] = m
            dd = np.zeros((n, 34), dtype=np.float32)
            for t in range(1, n):
                dd[t] = 1.0 + dd[t - 1] * (1.0 - m[t - 1])
            D[i, :n] = dd
            yv = np.nan_to_num(arr[:, COL_IDX["SepsisLabel"]], nan=0).astype(np.uint8)
            Y[i, :n] = yv

            u1 = arr[:, COL_IDX["Unit1"]]
            u2 = arr[:, COL_IDX["Unit2"]]
            u1v = np.unique(u1[~np.isnan(u1)])
            u2v = np.unique(u2[~np.isnan(u2)])
            if len(u1v) > 1 or len(u2v) > 1:
                unit_varying += 1
            has_unit = len(u1v) == 1 and len(u2v) == 1
            if has_unit and not ((u1v[0] == 1) ^ (u2v[0] == 1)):
                unit_bad += 1
            icul = arr[:, COL_IDX["ICULOS"]]
            if not (icul[0] == 1 and np.all(np.diff(icul) == 1)):
                iculos_bad += 1
            unit = "unknown" if not has_unit else ("MICU" if u1v[0] == 1 else "SICU")

            meta.append(dict(
                patient_id=os.path.basename(path)[:-4], hospital=hosp, unit=unit,
                domain=f"{hosp}-{unit}", n_hours=int(lengths[i]),
                age=float(np.nanmax(arr[:, COL_IDX["Age"]])),
                gender=float(np.nanmax(arr[:, COL_IDX["Gender"]])),
                hosp_adm_time=float(np.nanmax(arr[:, COL_IDX["HospAdmTime"]])),
                sepsis_ever=int(yv.max()),
                first_pos_hour=int(np.argmax(yv)) if yv.max() > 0 else -1,
            ))
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{N}  ({time.time()-t0:.0f}s)", flush=True)

    for a in (X, M, D, Y):
        a.flush()
    np.save(f"{ARR}/lengths.npy", lengths)
    mdf = pl.DataFrame(meta)
    mdf.write_parquet(f"{OUT}/meta.parquet")
    print(f"arrays written in {time.time()-t0:.0f}s", flush=True)

    # ---------- 全量 QC ----------
    isA = np.array([h == "A" for _, h in files])
    L = np.minimum(lengths, T_CAP)
    valid = np.arange(T_CAP)[None, :] < L[:, None]

    def obs_rate(sel):
        return (M[sel].sum(axis=(0, 1)) / valid[sel].sum()).astype(float)

    ra, rb = obs_rate(isA), obs_rate(~isA)
    qc = dict(
        n_patients=dict(A=int(isA.sum()), B=int((~isA).sum())),
        n_rows=dict(A=int(L[isA].sum()), B=int(L[~isA].sum()), total=int(L.sum())),
        max_len=int(lengths.max()), n_truncated=int((lengths > T_CAP).sum()),
        unit_varying_within_patient=int(unit_varying),
        unit_not_mutually_exclusive=int(unit_bad),
        iculos_not_1_to_n=int(iculos_bad),
        obs_rate_A=dict(zip(TV, np.round(ra, 6).tolist())),
        obs_rate_B=dict(zip(TV, np.round(rb, 6).tolist())),
        entry_density=dict(A=float(M[isA].sum() / (valid[isA].sum() * 34)),
                           B=float(M[~isA].sum() / (valid[~isA].sum() * 34))),
        hourly_prevalence=dict(A=float(Y[isA][valid[isA]].mean()),
                               B=float(Y[~isA][valid[~isA]].mean())),
        patient_prevalence=dict(
            A=float(mdf.filter(pl.col("hospital") == "A")["sepsis_ever"].mean()),
            B=float(mdf.filter(pl.col("hospital") == "B")["sepsis_ever"].mean())),
    )
    u, c = np.unique(mdf["domain"].to_numpy(), return_counts=True)
    qc["domain_counts"] = {str(k): int(v) for k, v in zip(u, c)}
    qc["unit_observed_frac"] = dict(
        A=float((mdf.filter(pl.col("hospital") == "A")["unit"] != "unknown").mean()),
        B=float((mdf.filter(pl.col("hospital") == "B")["unit"] != "unknown").mean()))
    qc["structurally_absent"] = [TV[j] for j in range(34)
                                 if (ra[j] == 0 and rb[j] > 0) or (rb[j] == 0 and ra[j] > 0)]
    qc["abs_diff_gt_50pp"] = [TV[j] for j in range(34) if abs(ra[j] - rb[j]) > 0.5]
    qc["abs_diff_gt_20pp"] = [TV[j] for j in range(34) if abs(ra[j] - rb[j]) > 0.2]
    ever = (M.sum(axis=1) > 0)
    qc["ever_obs_frac_A"] = dict(zip(TV, np.round(ever[isA].mean(axis=0), 6).tolist()))
    qc["ever_obs_frac_B"] = dict(zip(TV, np.round(ever[~isA].mean(axis=0), 6).tolist()))

    with open(f"{OUT}/qc_stage0.json", "w") as f:
        json.dump(qc, f, indent=2)
    show = {k: v for k, v in qc.items() if not k.startswith(("obs_rate", "ever_obs"))}
    print("\n=== QC ===")
    print(json.dumps(show, indent=2))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
