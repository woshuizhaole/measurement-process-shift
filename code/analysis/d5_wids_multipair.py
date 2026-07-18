#!/usr/bin/env python
"""D5 diagnostic (cheap, cross-hospital): fixed SOURCE pool -> many TARGET hospitals on WiDS2020.
Split 147 hospitals into 70% SOURCE / 30% TARGET by hospital id. Train ONE model on pooled
SOURCE rows; evaluate per held-out TARGET hospital. 6 fits total (VM/V/M x lgb/logistic).
Per target hospital t:
  gap        = auc_src_holdout - auc_tgt(VM)
  mask_delta = auc_tgt(V+M) - auc_tgt(V)   [>0 mask helps cross-hosp; <0 backfire]  == pilot cross_delta
  mask_only  = auc_tgt of mask-only model
  fp_auc     = mask -> (t vs source-pool) one-vs-rest fingerprint
  prev_t
Directly tests whether A<->B (frac_M~0.20, lgb mask_delta +0.067) is representative across many hospitals.
"""
import json, time, collections
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

P = "."
SEED = 20260711
rng = np.random.RandomState(SEED)
t0 = time.time()

df = pl.read_csv(f"{P}/data/external/wids2020/training_v2.csv",
                 infer_schema_length=None, null_values=["", "NA"])
y = df["hospital_death"].to_numpy().astype(float)
hid = df["hospital_id"].to_numpy()
ok = (~np.isnan(y)) & np.array([h is not None for h in hid.tolist()])
cols = [c for c in df.columns if c.endswith("_apache") or c.startswith(("d1_", "h1_"))]
V = np.column_stack([df[c].cast(pl.Float64, strict=False).to_numpy() for c in cols]).astype(np.float64)
M = np.isnan(V).astype(np.float64)
y = y[ok]; hid = hid[ok]; V = V[ok]; M = M[ok]

hosps = np.array([h for h in sorted(set(hid.tolist()))])
rng.shuffle(hosps)
n_src = int(0.7 * len(hosps))
src_h = set(hosps[:n_src].tolist()); tgt_h = set(hosps[n_src:].tolist())
is_src = np.array([h in src_h for h in hid.tolist()])
# impute using SOURCE medians only (no target leakage)
med = np.nanmedian(np.where(M[is_src] == 1, np.nan, V[is_src]), axis=0)
med = np.where(np.isnan(med), 0.0, med)
Vimp = np.where(M == 1, med, V)
mu = Vimp[is_src].mean(0); sd = Vimp[is_src].std(0) + 1e-9
Vz = (Vimp - mu) / sd
print(f"n={len(y)} feats={len(cols)} src_hosp={len(src_h)} tgt_hosp={len(tgt_h)} "
      f"src_mort={y[is_src].mean():.4f} ({time.time()-t0:.0f}s)", flush=True)

# source train / holdout
si = np.where(is_src)[0]; rng.shuffle(si)
nh = int(0.2 * len(si)); s_hold = si[:nh]; s_tr = si[nh:]
ytr = y[s_tr]; yhold = y[s_hold]
spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())


def make(model):
    if model == "lgb":
        return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                  min_child_samples=100, n_jobs=8, verbose=-1,
                                  random_state=SEED, scale_pos_weight=spw)
    return LogisticRegression(max_iter=300, C=1.0, class_weight="balanced")


def auc(yt, p):
    return roc_auc_score(yt, p) if len(np.unique(yt)) > 1 else np.nan


# train the 6 models once
FT = {"lgb": (Vimp, np.hstack([Vimp, M])), "logistic": (Vz, np.hstack([Vz, M]))}
fitted = {}
for model in ["lgb", "logistic"]:
    Vx, VMx = FT[model]
    m_vm = make(model).fit(VMx[s_tr], ytr)
    m_v = make(model).fit(Vx[s_tr], ytr)
    m_m = make(model).fit(M[s_tr], ytr)
    fitted[model] = (m_vm, m_v, m_m, Vx, VMx)
    print(f"{model} fitted ({time.time()-t0:.0f}s)", flush=True)

# in-domain source holdout AUCs
src_auc = {}
for model in ["lgb", "logistic"]:
    m_vm, m_v, m_m, Vx, VMx = fitted[model]
    src_auc[model] = auc(yhold, m_vm.predict_proba(VMx[s_hold])[:, 1])

# fingerprint model: mask -> src vs tgt (one model, but we want per-hosp t vs src pool)
big_tgt = [h for h in tgt_h if (hid == h).sum() >= 300]
big_tgt = sorted(big_tgt, key=lambda h: -(hid == h).sum())
print(f"target hospitals >=300: {len(big_tgt)}", flush=True)

recs = []
for t in big_tgt:
    te = hid == t
    yt = y[te]
    if yt.sum() < 5 or (yt == 0).sum() < 5:
        continue
    row = dict(hosp=int(t), n_tgt=int(te.sum()), prev_t=float(yt.mean()))
    for model in ["lgb", "logistic"]:
        m_vm, m_v, m_m, Vx, VMx = fitted[model]
        a_vm = auc(yt, m_vm.predict_proba(VMx[te])[:, 1])
        a_v = auc(yt, m_v.predict_proba(Vx[te])[:, 1])
        a_m = auc(yt, m_m.predict_proba(M[te])[:, 1])
        row[f"{model}_auc_tgt_VM"] = a_vm
        row[f"{model}_gap"] = src_auc[model] - a_vm
        row[f"{model}_mask_delta_tgt"] = a_vm - a_v
        row[f"{model}_mask_only_tgt"] = a_m
    # fingerprint: mask -> (this target hosp vs source-pool sample)
    src_samp = rng.choice(si, min(len(si), 3 * te.sum()), replace=False)
    Xfp = np.vstack([M[te], M[src_samp]])
    yfp = np.r_[np.ones(te.sum()), np.zeros(len(src_samp))]
    pi = rng.permutation(len(yfp)); ntr = int(.6 * len(yfp))
    fp = LogisticRegression(max_iter=200, C=1.0).fit(Xfp[pi[:ntr]], yfp[pi[:ntr]])
    row["fp_auc"] = auc(yfp[pi[ntr:]], fp.predict_proba(Xfp[pi[ntr:]])[:, 1])
    recs.append(row)
    print(f"hosp {t}: n={row['n_tgt']} prev={row['prev_t']:.3f} "
          f"lgb gap={row['lgb_gap']:+.3f} maskΔ={row['lgb_mask_delta_tgt']:+.3f} "
          f"log maskΔ={row['logistic_mask_delta_tgt']:+.3f} fp={row['fp_auc']:.3f} "
          f"maskonly={row['lgb_mask_only_tgt']:.3f}", flush=True)

out = pl.DataFrame(recs)
out.write_csv(f"{P}/results/tables/d5_wids_multipair.csv")


def summ(key):
    a = np.array([r[key] for r in recs], float); a = a[~np.isnan(a)]
    return dict(n=len(a), mean=float(a.mean()), std=float(a.std()),
                p10=float(np.percentile(a, 10)), median=float(np.median(a)),
                p90=float(np.percentile(a, 90)), min=float(a.min()), max=float(a.max()),
                frac_positive=float((a > 0).mean()))


S = {k: summ(k) for k in ["lgb_gap", "lgb_mask_delta_tgt", "lgb_mask_only_tgt", "lgb_auc_tgt_VM",
                          "logistic_gap", "logistic_mask_delta_tgt", "fp_auc", "prev_t"]}
json.dump(dict(n_tgt_hosp=len(recs), src_auc=src_auc, summary=S),
          open(f"{P}/results/stats/d5_wids_multipair.json", "w"), indent=2)
print("\n=== SUMMARY across %d target hospitals ===" % len(recs), flush=True)
for k, v in S.items():
    print(f"{k:26s} mean={v['mean']:+.3f} std={v['std']:.3f} p10={v['p10']:+.3f} "
          f"med={v['median']:+.3f} p90={v['p90']:+.3f} frac>0={v['frac_positive']:.2f}", flush=True)
print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
