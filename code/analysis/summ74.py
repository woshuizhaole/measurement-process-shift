#!/usr/bin/env python
"""轻量汇总 progress.jsonl: 每折 tgt_util(vo/vm/R4/OPS) + gain vs max + 配对CI + repair_holds。无 bootstrap。"""
import json
P = "."
rows = [json.loads(l) for l in open(f"{P}/results/stats/stage7_lodo_progress.jsonl") if l.strip()]
print("%-16s %-4s | vo /  vm /  R4 / OPS  | base | R4 gain  CI            hold | OPS gain  CI            hold" % ("fold", "kind"))
for r in rows:
    m = r["metrics"]; f = r["fold"]
    vo = m["value_only"]["ens_tgt_util"]; vm = m["value_mask"]["ens_tgt_util"]
    r4 = m["R4_ipw"]["ens_tgt_util"]; ops = m["OPS_full"]["ens_tgt_util"]
    base = max(vo, vm); bn = "vm" if vm >= vo else "vo"
    vs = "value_mask" if bn == "vm" else "value_only"
    ci = {(c["method"], c["vs"]): c for c in r["paired_ci"]}
    r4c = ci[("R4_ipw", vs)]; opc = ci[("OPS_full", vs)]
    r4h = "HOLD" if r4c["util_lo"] > 0 else "no"
    oph = "HOLD" if opc["util_lo"] > 0 else "no"
    print("%-16s %-4s | %.3f/%.3f/%.3f/%.3f | %s | %+.3f [%+.3f,%+.3f] %-4s | %+.3f [%+.3f,%+.3f] %-4s"
          % (f, r["kind"], vo, vm, r4, ops, bn, r4 - base, r4c["util_lo"], r4c["util_hi"], r4h,
             ops - base, opc["util_lo"], opc["util_hi"], oph))
print("\nfolds_done=%d/10" % len(rows))
