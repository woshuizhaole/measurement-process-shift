#!/usr/bin/env python
"""Stage 7 :: 多院描述性 + 多院放大 (方案 §Stage7.4-5)

eICU-CRD demo (186 医院): 逐化验观测率的跨院分布 -> 测量过程异质性描述图数据。
WiDS2020 GOSSIS (147 医院, 静态首24h): mask->医院可判别性 + mask 是否随医院变(多院放大)。
  静态设定下退化为 DAMS + 本框架的条件版本; 验证"测量指纹"在数百院上普适。
"""
import os, sys, json, gzip, csv, time
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb

P = "."
TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"
SEED = 20260711


def eicu_demo():
    """eICU demo: patient.csv.gz 只有病人表; 用 lab.csv 算逐化验逐院观测率需 lab 表(未下)。
    此处用 patient 表的字段可得性 + apacheApsVar 等做测量剖面近似。仅 patient.csv 时给医院计数。"""
    path = f"{P}/data/external/eicu_demo/patient.csv.gz"
    if not os.path.exists(path):
        return dict(note="eICU demo patient.csv.gz 不存在")
    with gzip.open(path, "rt") as f:
        rows = list(csv.DictReader(f))
    hosp = [r["hospitalid"] for r in rows]
    uniq = set(hosp)
    from collections import Counter
    c = Counter(hosp)
    return dict(n_stays=len(rows), n_hospitals=len(uniq),
                stays_per_hosp_min=min(c.values()), stays_per_hosp_max=max(c.values()),
                stays_per_hosp_median=int(np.median(list(c.values()))),
                note="eICU demo: 186 医院但每院 10-40 例, 仅作多院存在性描述; 完整逐化验观测率需 lab.csv(主站故障未下)")


def wids():
    """WiDS2020: training_v2.csv 静态首24h. hospital_id + d1_/h1_ 列.
    (1) mask->医院多分类可判别; (2) 逐 d1_/h1_ 列观测率跨院异质."""
    path = f"{P}/data/external/wids2020/training_v2.csv"
    if not os.path.exists(path):
        return dict(note="WiDS training_v2.csv 不存在")
    df = pl.read_csv(path, infer_schema_length=None, null_values=["", "NA"])
    n = df.height
    hid = df["hospital_id"].to_numpy()
    labcols = [c for c in df.columns if c.startswith(("d1_", "h1_"))]
    # mask 矩阵
    M = np.stack([df[c].is_null().not_().to_numpy().astype(np.float32) for c in labcols], axis=1)
    # 大医院(>=200例)
    import collections
    cc = collections.Counter(hid.tolist())
    big = [h for h, v in cc.items() if v >= 200 and h is not None]
    sel = np.isin(hid, big)
    Msel = M[sel]; hsel = hid[sel]
    # mask->医院 多分类(判别性): 二分最大两院
    top2 = [h for h, _ in cc.most_common() if h in big][:2]
    m2 = np.isin(hsel, top2)
    y2 = (hsel[m2] == top2[1]).astype(int)
    X2 = Msel[m2]
    rng = np.random.RandomState(SEED); idx = rng.permutation(len(y2))
    tr, te = idx[:int(.6 * len(idx))], idx[int(.6 * len(idx)):]
    clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, verbose=-1, n_jobs=8, random_state=SEED)
    clf.fit(X2[tr], y2[tr])
    auc_top2 = roc_auc_score(y2[te], clf.predict_proba(X2[te])[:, 1])
    # 逐列观测率跨院异质: p90-p10
    obs = np.stack([Msel[hsel == h].mean(0) for h in big])  # (n_big, n_lab)
    spread = np.percentile(obs, 90, axis=0) - np.percentile(obs, 10, axis=0)
    n_het = int((spread > 0.30).sum())
    return dict(n_rows=n, n_hospitals=len(cc), n_big_hospitals=len(big),
                mask_top2_hospital_auc=float(auc_top2),
                n_lab_cols=len(labcols), n_labcols_obs_spread_gt_0p30=n_het,
                median_across_hosp_obs_spread=float(np.median(spread)),
                note="WiDS 静态首24h; mask 判别最大两院 AUC; 逐化验观测率跨院异质列数(p90-p10>0.30)")


def main():
    t0 = time.time()
    os.makedirs(STT, exist_ok=True)
    res = dict(eicu_demo=eicu_demo(), wids2020=wids())
    json.dump(res, open(f"{STT}/stage7_multihospital.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
