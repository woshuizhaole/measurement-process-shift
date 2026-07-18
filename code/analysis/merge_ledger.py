#!/usr/bin/env python
"""合并 Stage 2 捷径账本的 4 个分片 -> 全局聚合 + 门 G2。"""
import sys, json, glob
import polars as pl

P = "."
TAB = f"{P}/results/tables"; STT = f"{P}/results/stats"

frames = [pl.read_csv(f) for f in sorted(glob.glob(f"{TAB}/stage2_shortcut_ledger_*.csv"))]
df = pl.concat(frames)
df.write_csv(f"{TAB}/stage2_shortcut_ledger.csv")
ci = pl.concat([pl.read_csv(f) for f in sorted(glob.glob(f"{TAB}/stage2_ledger_ci_*.csv"))])
ci.write_csv(f"{TAB}/stage2_ledger_ci.csv")

agg = (df.group_by(["direction", "arch", "feat"])
       .agg([pl.col(c).mean().alias(c) for c in
             ["src_test_auroc", "tgt_test_auroc", "src_test_utility", "tgt_test_utility"]]
            + [pl.col("tgt_test_auroc").std().alias("tgt_auroc_std"),
               pl.col("seed").count().alias("n_seed")])
       .sort(["direction", "arch", "feat"]))
agg.write_csv(f"{TAB}/stage2_ledger_agg.csv")
print("=== aggregated (arch×feat, multi-seed) ===")
print(agg.to_pandas().to_string(index=False))

gates = {}
for r in agg.iter_rows(named=True):
    pass
for direction in ["A2B", "B2A"]:
    for arch in ["gru", "transformer"]:
        sub = {x["feat"]: x for x in agg.filter((pl.col("direction") == direction) &
                                                (pl.col("arch") == arch)).to_dicts()}
        if "value_mask" not in sub or "value_only" not in sub:
            continue
        vm, vo = sub["value_mask"], sub["value_only"]
        gates[f"{direction}_{arch}"] = dict(
            mask_in_gain=float(vm["src_test_auroc"] - vo["src_test_auroc"]),
            mask_cross_gain=float(vm["tgt_test_auroc"] - vo["tgt_test_auroc"]),
            in_util_gain=float(vm["src_test_utility"] - vo["src_test_utility"]),
            cross_util_gain=float(vm["tgt_test_utility"] - vo["tgt_test_utility"]),
            n_seed=int(vm["n_seed"]))
ci_by = {f"{r['direction']}_{r['arch']}": r for r in ci.to_dicts()}
for k in gates:
    if k in ci_by:
        gates[k]["cross_delta_ci"] = [ci_by[k]["lo"], ci_by[k]["hi"]]
        gates[k]["cross_delta_p_gt0"] = ci_by[k]["p_gt0"]
g2 = any(g["mask_cross_gain"] <= -0.03 and g.get("cross_delta_ci", [0, 9])[1] < 0 for g in gates.values())
gates["G2_threshold"] = dict(cross_gain="<=-0.03 且 CI 上界<0")
gates["G2_pass"] = bool(g2)
json.dump(gates, open(f"{STT}/stage2_gates.json", "w"), indent=2)
print("\n=== GATES (G2) ===")
print(json.dumps(gates, indent=2, ensure_ascii=False))
