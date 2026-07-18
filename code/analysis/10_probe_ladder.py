#!/usr/bin/env python
"""Stage 1 :: 观测过程解剖 —— 探针阶梯 + 逐变量刻画 + 时间指标混淆 + T1 (方案 §Stage1)

产出:
  results/tables/stage1_probe_ladder.csv        L0..L3 病人级/行级 mask->医院 AUC (+CI)
  results/tables/stage1_per_variable.csv        逐变量单变量医院 AUC + 观测率差
  results/tables/stage1_time_confound.csv       G1b: ICULOS/HospAdmTime 医院探针
  results/tables/stage1_T1_slope_stability.csv  T1: beta_j^A vs beta_j^B (无标签)
  results/stats/stage1_gates.json               门 G1 / G1b 判定
所有医院探针: 按病人分组划分, 逻辑回归 + LightGBM, 病人级 bootstrap CI。
"""
import os, sys, json, time
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb

sys.path.insert(0, "./code/utils")
import features as F
from boot import patient_bootstrap_ci

P = "."
TAB = f"{P}/results/tables"
STT = f"{P}/results/stats"
os.makedirs(TAB, exist_ok=True)
os.makedirs(STT, exist_ok=True)
SEED = 20260711
TV = F.TV


def flatten_mask(M, L, rows, cap=None):
    """把 (N,T,34) mask 展平成 (行, 34) + patient_id 向量, 只取 valid 小时。"""
    Ms, pid = [], []
    for i in rows:
        n = int(L[i]) if cap is None else min(int(L[i]), cap)
        Ms.append(M[i, :n])
        pid.append(np.full(n, i))
    return np.concatenate(Ms).astype(np.float32), np.concatenate(pid)


def hospital_probe(Xrow, hosp_row, pid_row, cols_idx, name, clf_kind="lgb", seed=SEED):
    """行级医院探针: 按病人分组 60/40, 返回 test AUC + 病人级 bootstrap CI。"""
    X = Xrow[:, cols_idx]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    tr, te = next(gss.split(X, hosp_row, pid_row))
    if clf_kind == "lr":
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    else:
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                 min_child_samples=50, n_jobs=16, verbose=-1, random_state=seed)
    clf.fit(X[tr], hosp_row[tr])
    p = clf.predict_proba(X[te])[:, 1]
    ci = patient_bootstrap_ci(hosp_row[te], p, pid_row[te], "auroc", B=1000, seed=seed)
    return ci


def main():
    t0 = time.time()
    X, M, D, Y, L = F.load_arrays()
    meta = F.load_meta()
    hosp = (meta["hospital"].to_numpy() == "B").astype(int)   # B=1
    N = X.shape[0]

    # 结构性缺失列 & Δ观测率
    qc = json.load(open(f"{P}/data/processed/qc_stage0.json"))
    ra = np.array([qc["obs_rate_A"][c] for c in TV])
    rb = np.array([qc["obs_rate_B"][c] for c in TV])
    struct = [j for j in range(34) if TV[j] in qc["structurally_absent"]]           # EtCO2
    d50 = [j for j in range(34) if abs(ra[j] - rb[j]) > 0.5]
    d20 = [j for j in range(34) if abs(ra[j] - rb[j]) > 0.2]
    print(f"structural={[TV[j] for j in struct]}, >50pp={[TV[j] for j in d50]}, >20pp={[TV[j] for j in d20]}")

    # ---------- 行级探针阶梯 ----------
    rng = np.random.default_rng(SEED)
    # 为控制内存/时间，行级探针在每院随机抽 8000 病人（分层）
    subA = rng.choice(np.where(hosp == 0)[0], 8000, replace=False)
    subB = rng.choice(np.where(hosp == 1)[0], 8000, replace=False)
    rows = np.concatenate([subA, subB])
    Xrow, pid_row = flatten_mask(M, L, rows)
    hosp_row = hosp[pid_row]
    print(f"row-level probe on {len(rows)} patients -> {len(pid_row)} rows ({time.time()-t0:.0f}s)")

    keep_all = list(range(34))
    keep_L1 = [j for j in keep_all if j not in struct]                               # 去 EtCO2
    keep_L2 = [j for j in keep_L1 if j not in d50]                                    # 再去 >50pp
    idx_unit = None  # Unit1/2 不在 34 通道里（它们是域标签），此处 L2 already schema-shared
    keep_L3 = [j for j in keep_L2 if abs(ra[j] - rb[j]) <= 0.2]                       # schema共享+中等缺失
    LABS_idx = [F.IDX[c] for c in F.LABS]
    ladders = {
        "L0_all34": keep_all,
        "L1_drop_structural": keep_L1,
        "L2_drop_gt50pp": keep_L2,
        "L3_shared_moderate": keep_L3,
        "labs_only": LABS_idx,
    }
    rows_out = []
    for name, cols in ladders.items():
        for kind in ["lr", "lgb"]:
            ci = hospital_probe(Xrow, hosp_row, pid_row, cols, name, kind)
            rows_out.append(dict(level=name, n_cols=len(cols), clf=kind, **ci))
            print(f"  row {name:22s} {kind} AUC={ci['point']:.4f} [{ci['lo']:.4f},{ci['hi']:.4f}]")

    # ---------- 病人级探针 (每变量平均观测率) ----------
    valid = np.arange(X.shape[1])[None, :] < L[:, None]
    obs_per_pt = (M.sum(axis=1) / np.maximum(valid.sum(axis=1, keepdims=True), 1)).astype(np.float32)  # (N,34)
    for name, cols in ladders.items():
        Xp = obs_per_pt[rows][:, cols]
        gss = GroupShuffleSplit(1, test_size=0.4, random_state=SEED)
        tr, te = next(gss.split(Xp, hosp[rows], rows))
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                 min_child_samples=50, n_jobs=16, verbose=-1, random_state=SEED)
        clf.fit(Xp[tr], hosp[rows][tr])
        pte = clf.predict_proba(Xp[te])[:, 1]
        auc = roc_auc_score(hosp[rows][te], pte)
        rows_out.append(dict(level=name, n_cols=len(cols), clf="patient_lgb",
                             point=float(auc), lo=np.nan, hi=np.nan, n_boot=0, mean=np.nan, std=np.nan))
        print(f"  pt  {name:22s} AUC={auc:.4f}")
    pl.DataFrame(rows_out).write_csv(f"{TAB}/stage1_probe_ladder.csv")

    # ---------- 逐变量单变量医院 AUC ----------
    pv = []
    for j in range(34):
        col = Xrow[:, j:j + 1]
        gss = GroupShuffleSplit(1, test_size=0.4, random_state=SEED)
        tr, te = next(gss.split(col, hosp_row, pid_row))
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(col[tr], hosp_row[tr])
        auc = roc_auc_score(hosp_row[te], clf.predict_proba(col[te])[:, 1])
        pv.append(dict(variable=TV[j], obs_rate_A=float(ra[j]), obs_rate_B=float(rb[j]),
                       abs_diff=float(abs(ra[j] - rb[j])), univariate_hospital_auc=float(max(auc, 1 - auc)),
                       is_structural=TV[j] in qc["structurally_absent"]))
    pl.DataFrame(pv).sort("univariate_hospital_auc", descending=True).write_csv(f"{TAB}/stage1_per_variable.csv")
    print("per-variable table written")

    # ---------- G1b: 时间指标混淆 ----------
    hat = meta["hosp_adm_time"].to_numpy().astype(np.float32)[rows]
    # ICULOS: 用每病人的停留长度 + 起点作代理（行级用 hour index）
    iculos_row = np.concatenate([np.arange(int(min(L[i], 336))) for i in rows]).astype(np.float32)
    hat_row = np.concatenate([np.full(int(min(L[i], 336)), hat[k]) for k, i in enumerate(rows)]).astype(np.float32)
    Xtime = np.stack([iculos_row, hat_row], axis=1)
    gss = GroupShuffleSplit(1, test_size=0.4, random_state=SEED)
    tr, te = next(gss.split(Xtime, hosp_row, pid_row))
    clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, n_jobs=16, verbose=-1, random_state=SEED)
    clf.fit(Xtime[tr], hosp_row[tr])
    auc_time = roc_auc_score(hosp_row[te], clf.predict_proba(Xtime[te])[:, 1])
    tc = [dict(probe="ICULOS+HospAdmTime", hospital_auc=float(auc_time))]
    for k, nm in [(0, "ICULOS_only"), (1, "HospAdmTime_only")]:
        clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, n_jobs=16, verbose=-1, random_state=SEED)
        clf.fit(Xtime[tr, k:k + 1], hosp_row[tr])
        a = roc_auc_score(hosp_row[te], clf.predict_proba(Xtime[te, k:k + 1])[:, 1])
        tc.append(dict(probe=nm, hospital_auc=float(a)))
    pl.DataFrame(tc).write_csv(f"{TAB}/stage1_time_confound.csv")
    print(f"G1b ICULOS+HospAdmTime hospital AUC = {auc_time:.4f}")

    # ---------- T1: 斜率稳定性 (无标签) ----------
    # phi(C) = build_phi 完整版 (密集生命体征 ffill + age + gender + iculos样条)。逐院拟合 M_j ~ phi。
    from sklearn.linear_model import LogisticRegression as LR
    t1 = []
    phi_cache = {}
    for h, hn in [(0, "A"), (1, "B")]:
        sub = rows[hosp[rows] == h][:3000]
        phi, phi_names = F.build_phi(X[sub], M[sub], L[sub], meta[sub.tolist()], src=hn)
        # 展平 valid 行 -> (rows_flat, d_phi)，同时记录每行属于哪个 sub-index
        flat_phi, flat_localidx = [], []
        for k, i in enumerate(sub):
            n = int(min(L[i], 336))
            flat_phi.append(phi[k, :n]); flat_localidx.append(np.full(n, k))
        phi_cache[hn] = (sub, np.concatenate(flat_phi), np.concatenate(flat_localidx))
    for j in range(34):
        betas = {}
        ok = True
        for hn in ["A", "B"]:
            sub, PH, loc = phi_cache[hn]
            mj = np.concatenate([M[i, :int(min(L[i], 336)), j] for i in sub])
            if mj.mean() < 0.005 or mj.mean() > 0.995:
                ok = False; break
            lr = LR(max_iter=1000, C=1.0).fit(PH, mj)
            betas[hn] = lr.coef_[0]
        if not ok:
            continue
        dif = np.abs(betas["A"] - betas["B"])
        t1.append(dict(variable=TV[j], mean_abs_beta_A=float(np.abs(betas["A"]).mean()),
                       mean_abs_beta_B=float(np.abs(betas["B"]).mean()),
                       max_abs_slope_diff=float(dif.max()), mean_abs_slope_diff=float(dif.mean())))
    pl.DataFrame(t1).sort("mean_abs_slope_diff", descending=True).write_csv(f"{TAB}/stage1_T1_slope_stability.csv")
    print(f"T1 slope-stability: {len(t1)} variables (phi=build_phi full)")

    # ---------- 安慰剂: 病人级标签置换 (泄漏杀手) ----------
    rng2 = np.random.default_rng(SEED + 7)
    perm_hosp_pt = hosp[rows].copy()
    rng2.shuffle(perm_hosp_pt)                                   # 病人级打乱
    pt2perm = dict(zip(rows, perm_hosp_pt))
    perm_row = np.array([pt2perm[g] for g in pid_row])
    placebo = []
    for name, cols, feathe in [("L0_row", keep_all, Xrow), ("L0_patient", keep_all, None)]:
        if feathe is not None:
            ci = hospital_probe(Xrow, perm_row, pid_row, cols, name, "lgb")
            placebo.append(dict(probe="L0_row_permuted", auc=ci["point"]))
        else:
            Xp = obs_per_pt[rows][:, cols]
            gss = GroupShuffleSplit(1, test_size=0.4, random_state=SEED)
            tr, te = next(gss.split(Xp, perm_hosp_pt, rows))
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                     min_child_samples=50, n_jobs=16, verbose=-1, random_state=SEED)
            clf.fit(Xp[tr], perm_hosp_pt[tr])
            placebo.append(dict(probe="L0_patient_permuted",
                                auc=float(roc_auc_score(perm_hosp_pt[te], clf.predict_proba(Xp[te])[:, 1]))))
    pl.DataFrame(placebo).write_csv(f"{TAB}/stage1_placebo.csv")
    print(f"placebo (permuted): {placebo}")

    # ---------- 门判定 ----------
    ladder_df = pl.read_csv(f"{TAB}/stage1_probe_ladder.csv")
    L3_pt = ladder_df.filter((pl.col("level") == "L3_shared_moderate") & (pl.col("clf") == "patient_lgb"))["point"][0]
    L3_row = ladder_df.filter((pl.col("level") == "L3_shared_moderate") & (pl.col("clf") == "lgb"))["point"][0]
    gates = dict(
        G1_L3_patient_auc=float(L3_pt), G1_L3_row_auc=float(L3_row),
        G1_threshold=0.75, G1_pass=bool(max(L3_pt, L3_row) >= 0.75),
        G1b_time_auc=float(auc_time), G1b_threshold=0.80, G1b_pass=bool(auc_time < 0.80),
    )
    json.dump(gates, open(f"{STT}/stage1_gates.json", "w"), indent=2)
    print("\n=== GATES ==="); print(json.dumps(gates, indent=2))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
