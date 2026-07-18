#!/usr/bin/env python
"""Stage 7 :: CinC2012 阴性对照 + ICUType 院内正对照 (方案 §Stage7.1-2, 门 G7)

CinC2012: 12000 例来自单中心 MIMIC-II(BIDMC), set-a/b/c 是随机三等分 -> 无跨院偏移。
  阴性对照(G7): set-a vs set-b 的 mask->伪医院探针 AUC 必须 <=0.55; 校正应是 no-op。
  院内正对照: ICUType(4类:冠心病监护/心外恢复/内科ICU/外科ICU) 是同院内的协议域,
    真有测量差异但无机构差异 -> mask->ICUType 探针应显著>0.5, 且跨 ICUType 有迁移损失。

CinC2012 格式: 每病人一个 txt (Time,Parameter,Value)。首行块(00:00)是 6 个总体描述子。
"""
import os, sys, json, glob, time
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from concurrent.futures import ThreadPoolExecutor

P = "."
RAW = f"{P}/data/interim/cinc2012"
OUT = f"{P}/data/processed"; TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"
SEED = 20260711

# CinC2012 37 时序变量
TS_VARS = ["Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol", "Creatinine",
           "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K", "Lactate", "Mg",
           "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP", "NISysABP", "PaCO2", "PaO2", "pH",
           "Platelets", "RespRate", "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC", "Weight"]
GEN = ["RecordID", "Age", "Gender", "Height", "ICUType", "Weight"]


def parse_one(path):
    """返回 (record_id, icutype, mask_over_ts(37,), observed_rates dict)。"""
    icutype = np.nan; obs = {v: 0 for v in TS_VARS}
    with open(path) as f:
        next(f)  # header Time,Parameter,Value
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            t, param, val = parts
            if param == "ICUType":
                try:
                    icutype = int(float(val))
                except Exception:
                    pass
            if param in obs and val not in ("", "-1"):
                obs[param] += 1
    rid = os.path.basename(path)[:-4]
    mask = np.array([1 if obs[v] > 0 else 0 for v in TS_VARS], dtype=np.float32)
    rates = np.array([obs[v] for v in TS_VARS], dtype=np.float32)
    return rid, icutype, mask, rates


def load_set(setname):
    files = sorted(glob.glob(f"{RAW}/{setname}/*.txt"))
    with ThreadPoolExecutor(max_workers=24) as ex:
        recs = list(ex.map(parse_one, files))
    return recs


def probe(Xrows, ylab, groups, name):
    gss = GroupShuffleSplit(1, test_size=0.4, random_state=SEED)
    tr, te = next(gss.split(Xrows, ylab, groups))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(Xrows[tr], ylab[tr])
    p = clf.predict_proba(Xrows[te])[:, 1] if len(set(ylab)) == 2 else None
    auc = roc_auc_score(ylab[te], p) if p is not None else np.nan
    return float(auc)


def main():
    t0 = time.time()
    os.makedirs(TAB, exist_ok=True); os.makedirs(STT, exist_ok=True)
    a = load_set("set-a"); b = load_set("set-b"); c = load_set("set-c")
    print(f"loaded a={len(a)} b={len(b)} c={len(c)} ({time.time()-t0:.0f}s)", flush=True)

    # 病人级 mask 剖面(每变量 ever-observed)
    def mat(recs):
        return (np.stack([r[2] for r in recs]),  # mask
                np.array([r[1] for r in recs]),   # icutype
                np.array([r[0] for r in recs]))   # rid

    Ma, ita, ida = mat(a); Mb, itb, idb = mat(b); Mc, itc, idc = mat(c)

    # === 阴性对照: set-a vs set-b mask->伪医院 ===
    Xab = np.vstack([Ma, Mb]); yab = np.r_[np.zeros(len(Ma)), np.ones(len(Mb))]
    gab = np.arange(len(yab))   # 病人级, 每行独立病人
    auc_ab = probe(Xab, yab.astype(int), gab, "a_vs_b")
    Xac = np.vstack([Ma, Mc]); yac = np.r_[np.zeros(len(Ma)), np.ones(len(Mc))]
    auc_ac = probe(Xac, yac.astype(int), np.arange(len(yac)), "a_vs_c")

    # === 院内正对照: mask->ICUType (4类, 用 MICU=3 vs SICU=4 二分或多分) ===
    allM = np.vstack([Ma, Mb, Mc]); allIT = np.r_[ita, itb, itc]
    valid = ~np.isnan(allIT)
    # 3(MICU) vs 4(SICU) 二分类
    sel = valid & np.isin(allIT, [3, 4])
    y_icu = (allIT[sel] == 4).astype(int)
    auc_icu = probe(allM[sel], y_icu, np.arange(sel.sum()), "micu_vs_sicu")
    # 各 ICUType 计数
    it_counts = {int(k): int(v) for k, v in zip(*np.unique(allIT[valid], return_counts=True))}

    res = dict(
        negctrl_a_vs_b_mask_auc=auc_ab, negctrl_a_vs_c_mask_auc=auc_ac,
        poscontrol_micu_vs_sicu_mask_auc=auc_icu,
        icutype_counts=it_counts,
        n_a=len(a), n_b=len(b), n_c=len(c),
        G7_threshold=0.55,
        G7_pass=bool(auc_ab <= 0.55 and auc_ac <= 0.55),
        G7_note="随机划分 mask->伪医院 AUC 必须<=0.55(无跨院偏移); ICUType 院内正对照应>0.5")
    json.dump(res, open(f"{STT}/stage7_cinc2012.json", "w"), indent=2, ensure_ascii=False)
    print("\n=== CinC2012 对照 ===")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
