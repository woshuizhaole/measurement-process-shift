#!/usr/bin/env python
"""合并 Stage 4 两个方向分片 -> 聚合 + 门 G5。"""
import sys, glob
import polars as pl
sys.path.insert(0, "./code/analysis")
from importlib import import_module
m = import_module("40b_ops_gpu")

P = "."; TAB = f"{P}/results/tables"
df = pl.concat([pl.read_csv(f) for f in sorted(glob.glob(f"{TAB}/stage4_ops_main_*.csv"))])
df.write_csv(f"{TAB}/stage4_ops_main.csv")
ci = pl.concat([pl.read_csv(f) for f in sorted(glob.glob(f"{TAB}/stage4_ops_ci_*.csv"))])
ci.write_csv(f"{TAB}/stage4_ops_ci.csv")
m.finalize(df)
print("\n=== CI vs value_only ===")
print(ci.to_pandas().to_string(index=False))
