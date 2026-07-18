#!/usr/bin/env python
"""Stage 7 :: eICU demo 186医院 逐化验观测率异质性 (完整 lab.csv, 方案 §Stage7.4)

lab.csv: 每行一次化验(patientunitstayid, labname, ...)。计算每 unit-stay 是否做过某化验 ->
逐 (医院, 化验) 观测率 -> 跨院异质性(常见化验的观测率在186院间的分布跨度)。
"""
import os, sys, json, gzip, csv, time
import numpy as np
from collections import defaultdict, Counter

P = "."
E = f"{P}/data/external/eicu_demo"; STT = f"{P}/results/stats"; TAB = f"{P}/results/tables"


def main():
    t0 = time.time()
    # patient -> hospital
    stay2hosp = {}
    with gzip.open(f"{E}/patient.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            stay2hosp[r["patientunitstayid"]] = r["hospitalid"]
    print(f"stays={len(stay2hosp)} hospitals={len(set(stay2hosp.values()))} ({time.time()-t0:.0f}s)", flush=True)

    # 每 stay 做过哪些 labname
    stay_labs = defaultdict(set)
    labname_count = Counter()
    with gzip.open(f"{E}/lab.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            sid = r["patientunitstayid"]; ln = r.get("labname", "")
            if sid in stay2hosp and ln:
                stay_labs[sid].add(ln); labname_count[ln] += 1
    print(f"lab rows parsed, {len(labname_count)} distinct labnames ({time.time()-t0:.0f}s)", flush=True)

    # 最常见的 30 个化验
    top_labs = [ln for ln, _ in labname_count.most_common(30)]
    # 逐(医院,化验)观测率: 该院有多少 stay 做过该 lab
    hosp_stays = defaultdict(list)
    for sid, h in stay2hosp.items():
        hosp_stays[h].append(sid)
    big_hosp = [h for h, ss in hosp_stays.items() if len(ss) >= 15]   # >=15 stays
    print(f"hospitals with >=15 stays: {len(big_hosp)}", flush=True)

    rows = []
    for ln in top_labs:
        rates = []
        for h in big_hosp:
            ss = hosp_stays[h]
            rate = np.mean([1.0 if ln in stay_labs[s] else 0.0 for s in ss])
            rates.append(rate)
        rates = np.array(rates)
        rows.append(dict(labname=ln, n_hosp=len(big_hosp),
                         mean_obs_rate=float(rates.mean()),
                         p10=float(np.percentile(rates, 10)), p90=float(np.percentile(rates, 90)),
                         spread_p90_p10=float(np.percentile(rates, 90) - np.percentile(rates, 10)),
                         sd_across_hosp=float(rates.std())))
    import polars as pl
    df = pl.DataFrame(rows).sort("spread_p90_p10", descending=True)
    df.write_csv(f"{TAB}/stage7_eicu_lab_heterogeneity.csv")

    n_het = int((df["spread_p90_p10"] > 0.30).sum())
    res = dict(n_hospitals_ge15stays=len(big_hosp), n_top_labs=len(top_labs),
               n_labs_spread_gt_0p30=n_het,
               median_spread=float(df["spread_p90_p10"].median()),
               max_spread_lab=df.row(0, named=True)["labname"],
               max_spread=float(df["spread_p90_p10"].max()),
               note="eICU demo 186医院: 常见化验观测率跨院异质(p90-p10)。真实逐化验观测率(升级自仅patient.csv)")
    json.dump(res, open(f"{STT}/stage7_eicu_labs.json", "w"), indent=2, ensure_ascii=False)
    print("\n=== eICU 逐化验跨院异质 ===")
    print(df.head(12).select(["labname", "mean_obs_rate", "spread_p90_p10", "sd_across_hosp"]).to_pandas().to_string(index=False))
    print(f"\n{json.dumps(res, indent=2, ensure_ascii=False)}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
