#!/usr/bin/env python
"""组装 GRU 最终 G5 门 (从 offline_ci_A2B.json + offline_ci_B2A.json)。"""
import json
P = "."
A = json.load(open(f"{P}/results/stats/offline_ci_A2B.json"))
B = json.load(open(f"{P}/results/stats/offline_ci_B2A.json"))
gates = {"backbone": "GRU-D 时序", "seeds": 10, "B": 2000,
         "spec": "OPS_full/R4_ipw 双向 tgt_util(集成) 超 max(value_only,value_mask)>=+0.02 且 vs 两基线配对 Utility CI 下界>0 (§9.1)"}
per = {}
for dname, d in [("A2B", A), ("B2A", B)]:
    pt = d["point"]; pr = d["paired"]
    vo = pt["value_only"]["tgt_util"]; vm = pt["value_mask"]["tgt_util"]
    base = max(vo, vm)
    for m in ["value_mask", "R1_reanchor", "R2_msr", "R3_dann_msr", "R4_ipw", "OPS_full"]:
        if m not in pt: continue
        u = pt[m]["tgt_util"]; gain = round(u - base, 4)
        ci_vo = pr.get(f"{m}_vs_value_only"); ci_vm = pr.get(f"{m}_vs_value_mask")
        lo_vo = ci_vo["lo"] if ci_vo else None
        lo_vm = ci_vm["lo"] if ci_vm else (0.0 if m == "value_mask" else None)
        ci_both = (lo_vo is not None and lo_vo > 0) and (m == "value_mask" or (lo_vm is not None and lo_vm > 0))
        per[f"{dname}_{m}"] = dict(tgt_util=u, best_base=round(base, 4), gain_vs_max=gain,
                                   ci_vs_value_only=[ci_vo["lo"], ci_vo["hi"]] if ci_vo else None,
                                   ci_vs_value_mask=[ci_vm["lo"], ci_vm["hi"]] if ci_vm else None,
                                   margin_pass=bool(gain >= 0.02), ci_both_lo_gt0=bool(ci_both),
                                   G5_pass_this=bool(gain >= 0.02 and ci_both))
gates["per_direction_method"] = per
for key in ["OPS_full", "R4_ipw"]:
    gates[f"G5_{key}_bidirectional_pass"] = bool(all(per.get(f"{d}_{key}", {}).get("G5_pass_this", False) for d in ["A2B", "B2A"]))
gates["G5_overall_pass"] = bool(gates["G5_OPS_full_bidirectional_pass"] or gates["G5_R4_ipw_bidirectional_pass"])
gates["candidate_C_supported"] = gates["G5_overall_pass"]
json.dump(gates, open(f"{P}/results/stats/stage4_ops_gru_gates.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in gates.items() if k != "per_direction_method"}, indent=1, ensure_ascii=False))
for k, v in per.items():
    if "R4_ipw" in k or "OPS_full" in k:
        print(f"  {k}: util={v['tgt_util']} gain={v['gain_vs_max']} CIvm={v['ci_vs_value_mask']} PASS={v['G5_pass_this']}")
