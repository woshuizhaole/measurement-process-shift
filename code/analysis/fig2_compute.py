#!/usr/bin/env python
"""Fig2 现算: Panel A (6域观测率矩阵), Panel B (WiDS147逐变量观测率异质), Panel F (value->hospital AUC)."""
import os, sys, json, time, collections
import numpy as np
sys.path.insert(0, "./code/utils")
import features as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
import polars as pl

P = "."
SEED = 20260711
t0 = time.time()
OUT = {}

# ================= load CinC arrays =================
X, M, D, Y, L = F.load_arrays(); meta = F.load_meta()
N, T, C = X.shape
valid = np.arange(T)[None, :] < L[:, None]
obs = (M.sum(1) / np.maximum(valid.sum(1, keepdims=True), 1)).astype(np.float32)  # (N,34) patient obs rate
hosp = meta["hospital"].to_numpy(); dom = meta["domain"].to_numpy()
TV = F.TV
doms = ["A-MICU","A-SICU","A-unknown","B-MICU","B-SICU","B-unknown"]

# ---------- Panel A: 6-domain obs matrix ----------
mat = np.zeros((len(TV), len(doms)))
for di,d in enumerate(doms):
    sel = dom==d
    mat[:,di] = obs[sel].mean(0)
dom_counts = {d:int((dom==d).sum()) for d in doms}
rng_e = mat.max(1) - mat.min(1)  # cross-domain range per var
order = np.argsort(-rng_e)
OUT["panelA"] = dict(
    variables=[TV[i] for i in order],
    domains=doms, domain_counts=dom_counts,
    matrix=[[float(mat[i,dj]) for dj in range(len(doms))] for i in order],
    cross_domain_range=[float(rng_e[i]) for i in order],
)
etco2 = TV.index("EtCO2")
OUT["panelA"]["etco2_row"] = {doms[dj]: float(mat[etco2,dj]) for dj in range(len(doms))}
print(f"[{time.time()-t0:.0f}s] Panel A done. EtCO2:", OUT["panelA"]["etco2_row"])

# ---------- Panel F: value->hospital AUC per single physiological signal ----------
# hospital A vs B; per var, among patients with >=1 obs, value summary feats -> predict hosp
hospAB = (hosp=="B").astype(int)
Xr = X  # X_raw
Mv = (M==1) & valid[:,:,None]  # observed positions
panelF = []
for j,v in enumerate(TV):
    obsmask = Mv[:,:,j]  # (N,T)
    n_obs_per = obsmask.sum(1)
    has = n_obs_per>=1
    # need both hospitals with enough observed patients
    nA = int(((hosp=="A")&has).sum()); nB = int(((hosp=="B")&has).sum())
    if nA<200 or nB<200:
        continue
    vals = Xr[:,:,j]
    feats = np.full((N,5), np.nan)
    for i in np.where(has)[0]:
        vv = vals[i][obsmask[i]]
        feats[i] = [vv.mean(), vv.min(), vv.max(), vv[-1], vv.std() if vv.size>1 else 0.0]
    idx = np.where(has)[0]
    Xf = feats[idx]; yf = hospAB[idx]; g = idx
    # impute col means on train inside pipeline manually: replace nan (none since has) - safe
    gss = GroupShuffleSplit(1, test_size=0.4, random_state=SEED)
    tr,te = next(gss.split(Xf,yf,g))
    if len(np.unique(yf[tr]))<2 or len(np.unique(yf[te]))<2: continue
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(Xf[tr], yf[tr])
    a = float(roc_auc_score(yf[te], clf.predict_proba(Xf[te])[:,1]))
    a = max(a, 1-a)  # identifiability magnitude (direction-agnostic)
    panelF.append(dict(variable=v, value_hosp_auc=a, n_obs_patients=int(has.sum()), nA=nA, nB=nB))
panelF.sort(key=lambda r:-r["value_hosp_auc"])
# mask full hospital-ID AUC (all 34 obs channels -> A vs B) as reference upper anchor
gss = GroupShuffleSplit(1, test_size=0.4, random_state=SEED)
tr,te = next(gss.split(obs, hospAB, np.arange(N)))
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
clf.fit(obs[tr], hospAB[tr])
mask_full_auc = float(roc_auc_score(hospAB[te], clf.predict_proba(obs[te])[:,1]))
OUT["panelF"] = dict(value_signals=panelF, mask_full_AB_auc=mask_full_auc,
                     mask_ref_band=[0.9314,0.99997], note="value_hosp_auc=direction-agnostic max(a,1-a); mask band from six-domain top2..cross-hospital")
print(f"[{time.time()-t0:.0f}s] Panel F done. mask_full_AB={mask_full_auc:.4f}, top value:", panelF[:3])

# ================= Panel B: WiDS 147 per-lab obs-rate heterogeneity =================
df = pl.read_csv(f"{P}/data/external/wids2020/training_v2.csv", infer_schema_length=None, null_values=["","NA"])
hid = df["hospital_id"].to_numpy()
labcols = [c for c in df.columns if c.startswith(("d1_","h1_"))]
Mw = np.stack([df[c].is_null().not_().to_numpy().astype(np.float32) for c in labcols], axis=1)
cc = collections.Counter(hid.tolist())
hosp_ok = [h for h,v in cc.items() if v>=100 and h is not None]  # hospitals w/ >=100 stays
rows=[]
for k,c in enumerate(labcols):
    per=[]
    for h in hosp_ok:
        sel = hid==h
        per.append(Mw[sel,k].mean())
    per=np.array(per)
    p10,p90 = np.percentile(per,10), np.percentile(per,90)
    rows.append(dict(col=c, mean=float(per.mean()), p10=float(p10), p90=float(p90),
                     spread=float(p90-p10), med=float(np.median(per))))
rows.sort(key=lambda r:-r["spread"])
n_gt30 = sum(1 for r in rows if r["spread"]>0.30)
med_spread = float(np.median([r["spread"] for r in rows]))
OUT["panelB"] = dict(n_hospitals=len(hosp_ok), n_labcols=len(labcols),
                     n_spread_gt_0p30=n_gt30, median_spread=med_spread,
                     top=rows[:24])
print(f"[{time.time()-t0:.0f}s] Panel B done. n_gt30={n_gt30} med_spread={med_spread:.4f} nhosp={len(hosp_ok)}")

json.dump(OUT, open(f"{P}/results/stats/fig2_panels_ABF.json","w"), indent=2)
print("SAVED", f"{P}/results/stats/fig2_panels_ABF.json", f"total {time.time()-t0:.0f}s")
