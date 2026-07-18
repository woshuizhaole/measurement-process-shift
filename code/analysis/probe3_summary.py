import json, numpy as np
P="."
d=json.load(open(f"{P}/results/diag_improve/probe3_backbone_decomp.json"))

def cell(blk, backbone, name):
    per=blk["per"]; pr=blk["paired"]
    vm=per["value_mask"]; vo=per["value_only"]; r4=per.get("R4_ipw",{})
    row=dict(backbone=backbone, cell=name, n_pat=blk["n_pat"],
        vmask_tgt_auroc=vm["tgt_auroc"],
        r4_tgt_auroc=r4.get("tgt_auroc"),
        r4_minus_vmask_auroc=(round(r4["tgt_auroc"]-vm["tgt_auroc"],4) if r4 else None),
        vmask_thrgap=vm["thr_transfer_gap"],
        vonly_thrgap=vo["thr_transfer_gap"],
        r4_thrgap=r4.get("thr_transfer_gap"),
        R4_vs_vmask_srcthr=pr.get("R4_vs_vmask@srcthr"),
        R4_vs_vmask_oracle=pr.get("R4_vs_vmask@oracle"))
    return row

rows=[]
for D in ["A2B","B2A"]:
    rows.append(cell(d["gru"][D],"GRU-D",D))
for f in sorted(d["lgb"]):
    rows.append(cell(d["lgb"][f],"LGB",f))

def verdict(r):
    o=r["R4_vs_vmask_oracle"]
    if o is None: return "NA"
    if o["lo"]>0: return "R4_WIN_oracle"
    if o["hi"]<0: return "R4_LOSE_oracle"
    return "TIE_oracle"
for r in rows: r["oracle_verdict"]=verdict(r)

# aggregate contrasts
def agg(bk):
    sub=[r for r in rows if r["backbone"]==bk]
    win=sum(r["oracle_verdict"]=="R4_WIN_oracle" for r in sub)
    lose=sum(r["oracle_verdict"]=="R4_LOSE_oracle" for r in sub)
    tie=sum(r["oracle_verdict"]=="TIE_oracle" for r in sub)
    aucd=[r["r4_minus_vmask_auroc"] for r in sub if r["r4_minus_vmask_auroc"] is not None]
    vmgap=[r["vmask_thrgap"] for r in sub]
    return dict(backbone=bk,n_cells=len(sub),R4win=win,R4lose=lose,R4tie=tie,
        mean_r4_minus_vmask_auroc=round(float(np.mean(aucd)),4),
        n_r4_improves_ranking=sum(x>0 for x in aucd),
        median_vmask_thrgap=round(float(np.median(vmgap)),4),
        max_vmask_thrgap=round(float(np.max(vmgap)),4))

out=dict(cells=rows, aggregate=[agg("GRU-D"),agg("LGB")],
    notes={
      "candidate_C_survives_oracle_GRU":"A2B +0.046 CI[0.012,0.080]; B2A +0.062 CI[0.032,0.094] — both CI>0 at oracle threshold",
      "candidate_C_LGB_oracle":"net wash: R4 wins 1-2 cells, loses 2 (CI<0), ties rest",
      "threshold_fragility":"value_mask thr-transfer gap median GRU vs LGB; value_only LGB gaps up to 0.517 >> GRU max 0.054",
      "ranking_mechanism":"R4(IPW on measurement process) improves GRU-D tgt AUROC in 2/2 dirs but degrades LGB in several LODO folds",
      "labelfree_transport":"prior-match(u@pm) and probe5 quantile/meanshift transport do NOT close gap; only oracle(target-label) thr does"})
json.dump(out,open(f"{P}/results/diag_improve/probe3_summary.json","w"),indent=1,ensure_ascii=False)
print("AGG:")
for a in out["aggregate"]: print(" ",a)
print("\nCELL oracle verdicts:")
for r in rows:
    ov=r["R4_vs_vmask_oracle"]
    print(f'  {r["backbone"]:6s} {r["cell"]:22s} dAUC={str(r["r4_minus_vmask_auroc"]):>7s} vmGap={r["vmask_thrgap"]:+.3f} oracle_R4vsVM={ov["delta"]:+.4f}[{ov["lo"]:+.3f},{ov["hi"]:+.3f}] -> {r["oracle_verdict"]}')
print("saved -> results/diag_improve/probe3_summary.json")
