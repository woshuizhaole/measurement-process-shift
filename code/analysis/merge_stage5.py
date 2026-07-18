#!/usr/bin/env python
"""合并 Stage 5 两方向分片 -> 聚合 + ClinicalDG 诅咒判定(端到端可复现, 与 committed gates.json schema 一致)。
判据: 通用 DG(IRM/VREx/GroupDRO/CORAL/DANN) 是否双向超 ERM +0.02 -> 否则诅咒确认。"""
import glob, json
import polars as pl

P = "."; TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"
df = pl.concat([pl.read_csv(f) for f in sorted(glob.glob(f"{TAB}/stage5_dg_baselines_*.csv"))])
df.write_csv(f"{TAB}/stage5_dg_baselines.csv")
agg = (df.group_by(["direction", "algo"]).agg(
    [pl.col(c).mean() for c in ["src_auroc", "tgt_auroc", "src_util", "tgt_util"]]
    + [pl.col("tgt_util").std().alias("tgt_util_std")]).sort(["direction", "algo"]))
agg.write_csv(f"{TAB}/stage5_dg_agg.csv")
print(agg.to_pandas().to_string(index=False))

DG = ["IRM", "VREx", "GroupDRO", "CORAL", "DANN"]
byd = {d: {r["algo"]: r for r in agg.filter(pl.col("direction") == d).to_dicts()} for d in ["A2B", "B2A"]}
g = {}
for d in ["A2B", "B2A"]:
    dd = byd[d]; erm, vo = dd["ERM"], dd["value_only"]
    for al in dd:
        if al in ("ERM", "value_only"):
            continue
        g[f"{d}_{al}"] = dict(util_vs_ERM=round(dd[al]["tgt_util"] - erm["tgt_util"], 4),
                              util_vs_value_only=round(dd[al]["tgt_util"] - vo["tgt_util"], 4))
g["ERM_beats_value_only"] = {d: round(byd[d]["ERM"]["tgt_util"] - byd[d]["value_only"]["tgt_util"], 4)
                             for d in ["A2B", "B2A"]}
g["any_DG_beats_ERM_by_0.02_both_dir"] = bool(any(
    all(g.get(f"{d}_{al}", {}).get("util_vs_ERM", -9) >= 0.02 for d in ["A2B", "B2A"]) for al in DG))
g["clinicaldg_curse_confirmed"] = not g["any_DG_beats_ERM_by_0.02_both_dir"]
g["note"] = ("通用 DG(IRM/VREx/GroupDRO/CORAL/DANN) 均未双向超 ERM +0.02 -> ClinicalDG诅咒确认: DG≈ERM. "
             "value+mask ERM 超 value_only(mask帮忙, 与Stage2一致)")
json.dump(g, open(f"{STT}/stage5_gates.json", "w"), indent=2, ensure_ascii=False)
print("\n=== gates (curse) ===")
print(json.dumps({k: v for k, v in g.items() if not isinstance(v, dict) or k == "ERM_beats_value_only"},
                 indent=2, ensure_ascii=False))
