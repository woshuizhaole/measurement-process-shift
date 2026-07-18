#!/usr/bin/env python
"""code/utils/features.py — 共享的数据/特征工具 (Stage 1+ 全程使用)

核心约定:
  * 归一化统计一律从"源域"读入 (norm_stats_<src>.json)；绝不在目标域上重新计算值统计。
  * phi(C): 只用永远可得的通道 (密集生命体征 ffill + Age/Gender + ICULOS样条 + 逐变量滞后观测历史)
            —— 供测量倾向 e_j(C) 使用，避免循环性 (方案 §6.5)。
"""
import json
import numpy as np

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
        "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
        "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
        "PTT", "WBC", "Fibrinogen", "Platelets"]
TV = VITALS + LABS                                  # 34
# phi(C) 用的密集、几乎总是可得的生命体征（方案 §6.5：不含化验、不含化验 mask）
PHI_VITALS = ["HR", "O2Sat", "SBP", "MAP", "Resp"]   # DBP/Temp/EtCO2 缺失率高，不入 phi 默认集
IDX = {c: i for i, c in enumerate(TV)}

PROC = "./data/processed"


def load_arrays(proc=PROC, mmap=True):
    mode = "r" if mmap else None
    X = np.load(f"{proc}/arrays/X_raw.npy", mmap_mode=mode)
    M = np.load(f"{proc}/arrays/M.npy", mmap_mode=mode)
    D = np.load(f"{proc}/arrays/delta.npy", mmap_mode=mode)
    Y = np.load(f"{proc}/arrays/y.npy", mmap_mode=mode)
    L = np.load(f"{proc}/arrays/lengths.npy")
    return X, M, D, Y, L


def load_meta(proc=PROC):
    import polars as pl
    return pl.read_parquet(f"{proc}/meta.parquet")


def load_splits(proc=PROC):
    return json.load(open(f"{proc}/splits_idx.json"))


def load_norm(src, proc=PROC):
    return json.load(open(f"{proc}/norm_stats_{src}.json"))


def ffill_impute(X, M, norm, use="mean", channels=None):
    """因果 ffill -> 前导空缺填源域统计量。返回归一化后的值张量 (同形状, NaN 已消除)。
    X:(N,T,C) M:(N,T,C) norm: 源域 norm dict。
    channels: 与 X 最后一维一一对应的通道名列表；缺省时假设 X 覆盖全部 34 个 TV 通道。
    use in {mean,median}."""
    N, T, C = X.shape
    names = channels if channels is not None else TV
    assert len(names) == C, f"channels({len(names)}) != X last dim ({C})"
    Z = np.array(X, dtype=np.float32, copy=True)
    for j in range(C):
        col = Z[:, :, j]
        mask = M[:, :, j].astype(bool)
        idx = np.where(mask, np.arange(T)[None, :], 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        col2 = np.take_along_axis(np.where(mask, col, np.nan), idx, axis=1)
        Z[:, :, j] = col2
    for j, name in enumerate(names):
        center = norm[name]["mean"] if use == "mean" else norm[name]["median"]
        scale = norm[name]["std"] if use == "mean" else norm[name]["iqr"]
        colj = Z[:, :, j]
        colj[np.isnan(colj)] = center
        Z[:, :, j] = (np.clip(colj, norm[name]["p01"], norm[name]["p999"]) - center) / (scale + 1e-8)
    return Z


def build_phi(X, M, L, meta_rows=None, proc=PROC, src="A", _norm=None, rich=False):
    """phi(C): (N, T, d_phi) 供测量倾向。
    含: PHI_VITALS 的 ffill 归一化值 + Age + Gender + ICULOS/24 + log1p(ICULOS)。
    rich=True 追加"全局观测强度历史"(方案 §6.5 迄今观测计数):
      当前小时观测变量数 + 迄今累计观测强度(逐小时观测数的 running mean)。
    逐变量自身的滞后(距上次观测)在 MeasurementPropensity 拟合时按变量追加(见 ops_repair)。
    meta_rows: 已按行子集切好的 polars DataFrame(含 age,gender); None 则 age/gender 置 0。
    _norm: 直接传入的源域 norm dict(优先); 否则按 src 加载。"""
    norm = _norm if _norm is not None else load_norm(src, proc)
    N, T, _ = X.shape
    Zv = ffill_impute(X[:, :, [IDX[c] for c in PHI_VITALS]],
                      M[:, :, [IDX[c] for c in PHI_VITALS]],
                      {c: norm[c] for c in PHI_VITALS}, channels=PHI_VITALS)
    if meta_rows is not None:
        age = meta_rows["age"].to_numpy().astype(np.float32)
        gen = meta_rows["gender"].to_numpy().astype(np.float32)
    else:
        age = np.zeros(N, dtype=np.float32); gen = np.zeros(N, dtype=np.float32)
    age_z = ((age - np.nanmean(age)) / (np.nanstd(age) + 1e-8)) if meta_rows is not None else age
    iculos = np.arange(T, dtype=np.float32)[None, :].repeat(N, 0)
    feats = [Zv,
             np.broadcast_to(age_z[:, None, None], (N, T, 1)),
             np.broadcast_to(gen[:, None, None], (N, T, 1)),
             (iculos / 24.0)[:, :, None],
             np.log1p(iculos)[:, :, None]]
    names = [f"vit_{c}" for c in PHI_VITALS] + ["age", "gender", "iculos_24", "log_iculos"]
    if rich:
        obs_per_hour = np.asarray(M).sum(axis=2).astype(np.float32)            # (N,T) 当前小时观测变量数
        cum_intensity = np.cumsum(obs_per_hour, axis=1) / (iculos + 1.0)       # 迄今平均观测强度
        feats += [(obs_per_hour / 34.0)[:, :, None], (cum_intensity / 34.0)[:, :, None]]
        names += ["obs_intensity_now", "cum_obs_intensity"]
    phi = np.concatenate([f.astype(np.float32) for f in feats], axis=2)
    return phi, names


if __name__ == "__main__":
    X, M, D, Y, L = load_arrays()
    sp = load_splits()
    print("arrays:", X.shape, M.dtype, "splits keys:", list(sp.keys()))
    print("A_train n=", len(sp["A_train"]), "B_test n=", len(sp["B_test"]))
    nrmA = load_norm("A")
    print("norm A HR:", {k: round(v, 3) for k, v in nrmA["HR"].items() if k in ("mean", "std", "observed_frac")})
