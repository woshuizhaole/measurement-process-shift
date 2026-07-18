#!/usr/bin/env python
"""Stage 3 :: 逐变量"丢/修/留"分诊表 (方案 §Stage3.4) + 重叠/正性诊断

综合:
  - Stage 1 T1 斜率稳定性 (无标签, stage1_T1_slope_stability.csv)
  - Stage 3 T2 gamma 不变性 (需标签, stage3_T2_gamma.csv)
  - 结构性缺失 (qc_stage0.json)
  - 逐变量两院观测率重叠/正性 (e_j 分布, ESS)
产出逐变量决策: 丢(schema不匹配/gamma变) / 修(协议差异gamma共享) / 留(跨院一致信息性) / 弱信号
"""
import os, sys, json
import numpy as np, polars as pl

P = "."
sys.path.insert(0, f"{P}/code/utils")
import features as F
TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"


def main():
    qc = json.load(open(f"{P}/data/processed/qc_stage0.json"))
    struct = set(qc["structurally_absent"])
    ra = qc["obs_rate_A"]; rb = qc["obs_rate_B"]

    t1 = pl.read_csv(f"{TAB}/stage1_T1_slope_stability.csv")
    t1d = {r["variable"]: r for r in t1.to_dicts()}
    t2p = f"{TAB}/stage3_T2_gamma.csv"
    t2d = {}
    if os.path.exists(t2p):
        for r in pl.read_csv(t2p).to_dicts():
            t2d[r["variable"]] = r

    rows = []
    for j, v in enumerate(F.TV):
        oa, ob = ra[v], rb[v]
        rec = dict(variable=v, obs_rate_A=oa, obs_rate_B=ob, abs_diff=abs(oa - ob),
                   is_structural=v in struct,
                   T1_slope_diff=t1d.get(v, {}).get("mean_abs_slope_diff", np.nan),
                   T2_theta_A=t2d.get(v, {}).get("theta_A", np.nan),
                   T2_theta_B=t2d.get(v, {}).get("theta_B", np.nan),
                   T2_p_bh=t2d.get(v, {}).get("p_bh", np.nan),
                   T2_reject_gamma_inv=t2d.get(v, {}).get("reject_gamma_invariant", None))
        # 正性: 两院最小观测率 (太小 -> 违反重叠)
        rec["min_obs_rate"] = min(oa, ob)
        # 决策规则 (方案 §Stage3.4)
        if v in struct or rec["min_obs_rate"] < 0.01:
            rec["decision"] = "DROP_schema_or_positivity"
        elif rec["T2_reject_gamma_inv"] is True:
            rec["decision"] = "DROP_gamma_varies"            # Prop 3: 不可迁移
        elif rec["T2_reject_gamma_inv"] is False:
            # gamma 共享: 若 T1 也拒绝(beta 变) -> 修; 否则留
            rec["decision"] = "REPAIR_protocol" if (not np.isnan(rec["T1_slope_diff"]) and
                                                    rec["T1_slope_diff"] > 0.15) else "KEEP_transportable"
        else:
            rec["decision"] = "WEAK_insufficient_evidence"
        rows.append(rec)

    df = pl.DataFrame(rows).sort("abs_diff", descending=True)
    df.write_csv(f"{TAB}/stage3_triage_table.csv")
    print("=== 逐变量分诊表 ===")
    print(df.select(["variable", "obs_rate_A", "obs_rate_B", "is_structural",
                     "T2_reject_gamma_inv", "decision"]).to_pandas().to_string(index=False))
    counts = df.group_by("decision").len().sort("len", descending=True)
    print("\n=== 决策计数 ==="); print(counts.to_pandas().to_string(index=False))
    json.dump({r["decision"]: int(r["len"]) for r in counts.to_dicts()},
              open(f"{STT}/stage3_triage_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
