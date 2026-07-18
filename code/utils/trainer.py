#!/usr/bin/env python
"""code/utils/trainer.py — GRU 训练/评估循环 (Stage 2+ 全程)

- 数据: 从 memmap 取一批病人, 变长 -> pad + pack。
- 值通道: 因果 ffill + 源域归一化 (在 dataset 内做, 用源域 norm)。
- 训练目标: 时间变权的 cost-sensitive BCE (方案 提到把官方 Utility 写成时间变权代价);
  但 Stage2 捷径账本用标准 BCE 即可(公平对比不同特征集)。默认标准 BCE。
- 评估: 输出逐病人 prob 序列 -> normalized_utility + AUROC/AUPRC (病人级 bootstrap 在外层)。
"""
import sys, numpy as np, torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

sys.path.insert(0, "./code/utils")
import features as F
from models import GRUD, SeqTransformer, normalized_utility, best_threshold
from sklearn.metrics import roc_auc_score, average_precision_score


def make_encoder(arch, feat_mode, hidden):
    if arch == "gru":
        return GRUD(feat_mode=feat_mode, hidden=hidden)
    if arch == "transformer":
        return SeqTransformer(feat_mode=feat_mode, d_model=hidden)
    raise ValueError(arch)


class Cache:
    """把一组病人的 (归一化值, mask, delta, static, y, lengths) 预算好放内存/GPU。"""
    def __init__(self, rows, src_norm, device, X=None, M=None, D=None, Y=None, L=None,
                 meta=None, Tcap=200, meas_override=None):
        if X is None:
            X, M, D, Y, L = F.load_arrays()
        rows = np.asarray(rows)
        self.rows = rows
        self.device = device
        Ls = np.minimum(L[rows], Tcap)
        self.L = Ls
        T = int(Ls.max())
        # 值通道: ffill + 源域归一化
        Zc = F.ffill_impute(np.asarray(X[rows][:, :T]), np.asarray(M[rows][:, :T]), src_norm, channels=F.TV)
        self.x = torch.tensor(Zc, dtype=torch.float32)
        self.m = torch.tensor(np.asarray(M[rows][:, :T]), dtype=torch.float32)
        self.d = torch.tensor(np.asarray(D[rows][:, :T]), dtype=torch.float32)
        # delta 归一化(log1p)
        self.d = torch.log1p(self.d)
        if meta is not None:
            age = meta["age"].to_numpy()[rows]
            gen = meta["gender"].to_numpy()[rows]
            age_z = (age - np.nanmean(age)) / (np.nanstd(age) + 1e-8)
            self.static = torch.tensor(np.stack([age_z, gen], 1), dtype=torch.float32)
        else:
            self.static = torch.zeros(len(rows), 2)
        self.y = torch.tensor(np.asarray(Y[rows][:, :T]), dtype=torch.float32)
        self.override = None
        if meas_override is not None:
            self.override = torch.tensor(meas_override[:, :T], dtype=torch.float32)

    def batches(self, bs, shuffle=True, seed=0):
        idx = np.arange(len(self.rows))
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            yield b


def masked_bce(logits, y, lengths, pos_weight=None):
    B, T = logits.shape
    mask = (torch.arange(T, device=logits.device)[None, :] < torch.tensor(lengths, device=logits.device)[:, None])
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, y, reduction="none", pos_weight=pos_weight)
    return (loss * mask).sum() / mask.sum()


def train_eval(train_cache, eval_caches, feat_mode="value_mask", hidden=64, epochs=15,
               lr=1e-3, bs=256, pos_weight=15.0, seed=0, device="cuda:0", verbose=False,
               arch="gru"):
    """训练序列编码器(arch: gru|transformer) 于 train_cache, 在 eval_caches 上评估。
    返回 dict: name -> {probs_by_patient, y_by_patient, auroc, auprc}。阈值/Utility 外层按源域选。"""
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_encoder(arch, feat_mode, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    tc = train_cache
    x, m, d, st, y = tc.x.to(device), tc.m.to(device), tc.d.to(device), tc.static.to(device), tc.y.to(device)
    ov = tc.override.to(device) if tc.override is not None else None
    pw = torch.tensor(pos_weight, device=device)
    for ep in range(epochs):
        model.train()
        for b in tc.batches(bs, seed=seed + ep):
            opt.zero_grad()
            logit = model(x[b], m[b], d[b], st[b], meas_override=ov[b] if ov is not None else None)
            loss = masked_bce(logit, y[b], tc.L[b], pos_weight=pw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if verbose:
            print(f"    ep{ep} loss {loss.item():.4f}")

    model.eval()
    out = {}
    with torch.no_grad():
        for name, ec in eval_caches.items():
            ex, em, ed, es = ec.x.to(device), ec.m.to(device), ec.d.to(device), ec.static.to(device)
            eov = ec.override.to(device) if ec.override is not None else None
            probs, ys, flat_p, flat_y, pid = [], [], [], [], []
            for i in range(0, len(ec.rows), 512):
                sl = slice(i, i + 512)
                lg = model(ex[sl], em[sl], ed[sl], es[sl],
                           meas_override=eov[sl] if eov is not None else None)
                pr = torch.sigmoid(lg).cpu().numpy()
                for k in range(pr.shape[0]):
                    n = int(ec.L[i + k])
                    probs.append(pr[k, :n]); ys.append(ec.y[i + k, :n].numpy().astype(int))
                    flat_p.append(pr[k, :n]); flat_y.append(ec.y[i + k, :n].numpy().astype(int))
                    pid.append(np.full(n, ec.rows[i + k]))
            fp = np.concatenate(flat_p); fy = np.concatenate(flat_y); fg = np.concatenate(pid)
            out[name] = dict(probs=probs, ys=ys, flat_p=fp, flat_y=fy, flat_g=fg,
                             auroc=float(roc_auc_score(fy, fp)),
                             auprc=float(average_precision_score(fy, fp)))
    return model, out
