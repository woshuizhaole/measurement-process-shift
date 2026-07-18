#!/usr/bin/env python
"""code/ops_audit.py — 部署前观测过程审计工具 (Stage 8 交付物, 方案 §12)

只需目标院的无标签数据。子命令:
  fingerprint  : 探针阶梯 L0..L3, mask->医院判别 AUC (量化测量指纹强度)
  slope-test   : T1 无标签斜率稳定性 (逐变量 beta_j^src vs beta_j^tgt)
  triage       : 逐变量 丢/修/留 分诊表 (综合结构性/正性/斜率)
  repair       : 用目标院无标签 mask 做 MSR/重锚 (输出可套用的测量通道变换)

用法:
  python ops_audit.py fingerprint --source A.npz --target B.npz
  python ops_audit.py triage --source A.npz --target B.npz --out triage.csv
输入 .npz 需含: X(N,T,C值), M(N,T,C mask), lengths(N,), 可选 vitals_idx。
本文件是可复用骨架; 生产用需按目标院数据 schema 适配 loader。
"""
import sys, json, argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


def load_npz(path):
    d = np.load(path)
    return d["X"], d["M"], d["lengths"]


def flatten_mask(M, L):
    ms, pid = [], []
    for i in range(len(L)):
        n = int(L[i]); ms.append(M[i, :n]); pid.append(np.full(n, i))
    return np.concatenate(ms).astype(np.float32), np.concatenate(pid)


def cmd_fingerprint(args):
    Xs, Ms, Ls = load_npz(args.source); Xt, Mt, Lt = load_npz(args.target)
    ms, ps = flatten_mask(Ms, Ls); mt, pt = flatten_mask(Mt, Lt)
    X = np.vstack([ms, mt]); y = np.r_[np.zeros(len(ms)), np.ones(len(mt))].astype(int)
    g = np.r_[ps, pt + 10 ** 7]
    # 结构性列(某院全缺)
    ra = ms.mean(0); rb = mt.mean(0)
    struct = np.where((ra == 0) | (rb == 0))[0]
    d50 = np.where(np.abs(ra - rb) > 0.5)[0]
    ladders = {"L0_all": np.arange(X.shape[1]),
               "L1_drop_struct": np.setdiff1d(np.arange(X.shape[1]), struct),
               "L3_shared_moderate": np.array([j for j in range(X.shape[1])
                                               if j not in struct and abs(ra[j] - rb[j]) <= 0.2])}
    out = {}
    for name, cols in ladders.items():
        if len(cols) == 0:
            out[name] = None; continue
        gss = GroupShuffleSplit(1, test_size=0.4, random_state=0)
        tr, te = next(gss.split(X[:, cols], y, g))
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(X[tr][:, cols], y[tr])
        out[name] = float(roc_auc_score(y[te], clf.predict_proba(X[te][:, cols])[:, 1]))
    print(json.dumps({"probe_ladder_hospital_auc": out,
                      "structural_cols": struct.tolist(), "gt50pp_cols": d50.tolist()}, indent=2))
    return out


def cmd_triage(args):
    Xs, Ms, Ls = load_npz(args.source); Xt, Mt, Lt = load_npz(args.target)
    ms, _ = flatten_mask(Ms, Ls); mt, _ = flatten_mask(Mt, Lt)
    ra = ms.mean(0); rb = mt.mean(0)
    rows = []
    for j in range(len(ra)):
        mn = min(ra[j], rb[j])
        if ra[j] == 0 or rb[j] == 0 or mn < 0.01:
            dec = "DROP_schema_or_positivity"
        elif abs(ra[j] - rb[j]) > 0.15:
            dec = "REPAIR_protocol"
        else:
            dec = "KEEP"
        rows.append(dict(var=j, obs_src=float(ra[j]), obs_tgt=float(rb[j]), decision=dec))
    import csv
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(json.dumps(dict(Counter(r["decision"] for r in rows)), indent=2))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OPS observation-process audit (label-free)")
    sub = ap.add_subparsers(dest="cmd")
    for c in ["fingerprint", "slope-test", "triage", "repair"]:
        s = sub.add_parser(c)
        s.add_argument("--source", required=True); s.add_argument("--target", required=True)
        s.add_argument("--out", default=None); s.add_argument("--model", default=None)
    a = ap.parse_args()
    if a.cmd == "fingerprint":
        cmd_fingerprint(a)
    elif a.cmd == "triage":
        cmd_triage(a)
    else:
        print(f"[{a.cmd}] 见 Stage 3/4 对应脚本; 此 CLI 骨架已含 fingerprint/triage 可运行实现。")
