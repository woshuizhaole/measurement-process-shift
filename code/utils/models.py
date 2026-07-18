#!/usr/bin/env python
"""code/utils/models.py — 共享的序列模型 + 官方 Utility 评分封装 (Stage 2+ 全程)

- GRU-D 风格双输入 (值 + mask/delta)，可开关 mask 通道 -> 支持 value-only / value+mask / mask-only。
- 官方 Utility: 用 vendored evaluate_sepsis_score.compute_prediction_utility，逐病人流式因果。
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "./code/utils")
from evaluate_sepsis_score import compute_prediction_utility


# ----------------------- 官方 Utility -----------------------
def normalized_utility(y_true_list, y_prob_list, thr):
    """逐病人计算官方归一化 Utility（纯内存版，逐行等价于官方 evaluate_sepsis_score）。
    y_true_list/y_prob_list: list of 1D array (每病人一条, 按小时)。thr: 二值化阈值。
    归一化: (U_obs - U_inaction) / (U_best - U_inaction)。
    best predictor 窗口与官方一致: t_sepsis = argmax(labels) - dt_optimal;
      best[max(0,t_sepsis+dt_early) : min(t_sepsis+dt_late+1, n)] = 1。
    已在 tests/test_utility.py 中对拍官方目录接口。"""
    dt_early, dt_optimal, dt_late = -12, -6, 3.0
    max_u_tp, min_u_fn, u_fp, u_tn = 1, -2, -0.05, 0
    u_obs = u_best = u_inact = 0.0
    for y, p in zip(y_true_list, y_prob_list):
        y = np.asarray(y).astype(int)
        pred = (np.asarray(p) >= thr).astype(int)
        n = len(y)
        best = np.zeros(n)
        inact = np.zeros(n)
        if np.any(y):
            t_sepsis = int(np.argmax(y)) - dt_optimal      # 官方: argmax - dt_optimal (= argmax+6)
            best[max(0, t_sepsis + dt_early): min(t_sepsis + int(dt_late) + 1, n)] = 1
        u_obs += compute_prediction_utility(y, pred, dt_early, dt_optimal, dt_late,
                                            max_u_tp, min_u_fn, u_fp, u_tn, check_errors=False)
        u_best += compute_prediction_utility(y, best, dt_early, dt_optimal, dt_late,
                                             max_u_tp, min_u_fn, u_fp, u_tn, check_errors=False)
        u_inact += compute_prediction_utility(y, inact, dt_early, dt_optimal, dt_late,
                                              max_u_tp, min_u_fn, u_fp, u_tn, check_errors=False)
    if u_best - u_inact == 0:
        return 0.0
    return float((u_obs - u_inact) / (u_best - u_inact))


def best_threshold(y_true_list, y_prob_list, grid=None):
    """在源域上选使 Utility 最大的阈值（方案: 阈值只能在源域选）。"""
    if grid is None:
        grid = np.linspace(0.01, 0.6, 60)
    best_u, best_t = -9, 0.5
    for t in grid:
        u = normalized_utility(y_true_list, y_prob_list, t)
        if u > best_u:
            best_u, best_t = u, t
    return float(best_t), float(best_u)


# ----------------------- GRU-D 编码器 -----------------------
class GRUD(nn.Module):
    """序列编码器, 严格区分特征集 (方案 §7.1):
      feat_mode='value'        : 纯 ffill 值 + static。**绝不含 mask/delta/decay** (B0 value-only)。
      feat_mode='value_mask'   : ffill 值(经 GRU-D decay 衰减) + mask + delta + static (B1 value+mask)。
      feat_mode='value_maskchan': 同上但 mask 通道可被 meas_override 替换 (供 MSR/重锚)。
      feat_mode='mask'         : 只有 mask + delta + static, 无值 (B2 mask-only 探针)。
    关键: value 模式下没有 decay 层, 不接触 mask/delta, 保证 value-only 真的纯净。"""
    def __init__(self, n_ch=34, n_static=2, hidden=64, feat_mode="value_mask",
                 mask_input=None, dropout=0.2):
        super().__init__()
        self.feat_mode = feat_mode
        self.n_ch = n_ch
        use_decay = feat_mode in ("value_mask", "value_maskchan")
        self.decay = nn.Linear(n_ch, n_ch) if use_decay else None
        if feat_mode == "value":
            gin = n_ch + n_static                       # 纯值 + static
        elif feat_mode in ("value_mask", "value_maskchan"):
            gin = n_ch + 2 * n_ch + n_static            # decay值 + mask + delta + static
        elif feat_mode == "mask":
            gin = 2 * n_ch + n_static                   # mask + delta + static
        else:
            raise ValueError(feat_mode)
        self.gru = nn.GRU(gin, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x_hat, mask, delta, static, meas_override=None):
        B, T, _ = mask.shape
        parts = []
        if self.feat_mode == "value":
            parts.append(x_hat)                          # 纯 ffill 值, 不经 decay, 不看 mask
        elif self.feat_mode in ("value_mask", "value_maskchan"):
            g = torch.exp(-torch.relu(self.decay(delta)))
            x_dec = mask * x_hat + (1 - mask) * (g * x_hat)
            parts.append(x_dec)
            parts.append(mask if meas_override is None else meas_override)
            parts.append(delta)
        elif self.feat_mode == "mask":
            parts.append(mask if meas_override is None else meas_override)
            parts.append(delta)
        st = static.unsqueeze(1).expand(-1, T, -1)
        parts.append(st)
        z = torch.cat(parts, dim=-1)
        h, _ = self.gru(z)
        return self.head(h).squeeze(-1)               # (B,T) logits


class SeqTransformer(nn.Module):
    """因果 Transformer 编码器, 特征集约定同 GRUD。
    feat_mode: 'value' | 'value_mask' | 'value_maskchan' | 'mask'。因果 mask 保证不看未来。"""
    def __init__(self, n_ch=34, n_static=2, d_model=64, nhead=4, nlayers=2,
                 feat_mode="value_mask", dropout=0.2, max_len=340):
        super().__init__()
        self.feat_mode = feat_mode
        use_decay = feat_mode in ("value_mask", "value_maskchan")
        self.decay = nn.Linear(n_ch, n_ch) if use_decay else None
        if feat_mode == "value":
            din = n_ch + n_static
        elif feat_mode in ("value_mask", "value_maskchan"):
            din = n_ch + 2 * n_ch + n_static
        elif feat_mode == "mask":
            din = 2 * n_ch + n_static
        else:
            raise ValueError(feat_mode)
        self.inp = nn.Linear(din, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2 * d_model,
                                         dropout=dropout, batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, nlayers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x_hat, mask, delta, static, meas_override=None):
        B, T, _ = mask.shape
        parts = []
        if self.feat_mode == "value":
            parts.append(x_hat)
        elif self.feat_mode in ("value_mask", "value_maskchan"):
            g = torch.exp(-torch.relu(self.decay(delta)))
            parts.append(mask * x_hat + (1 - mask) * (g * x_hat))
            parts.append(mask if meas_override is None else meas_override)
            parts.append(delta)
        elif self.feat_mode == "mask":
            parts.append(mask if meas_override is None else meas_override)
            parts.append(delta)
        st = static.unsqueeze(1).expand(-1, T, -1)
        parts.append(st)
        z = self.inp(torch.cat(parts, -1)) + self.pos[:, :T]
        cmask = torch.triu(torch.ones(T, T, device=z.device, dtype=torch.bool), diagonal=1)  # 因果
        h = self.tr(z, mask=cmask)
        return self.head(h).squeeze(-1)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = GRUD(feat_mode="value_mask")
    x = torch.randn(4, 20, 34)
    out = m(x, torch.ones(4, 20, 34), torch.zeros(4, 20, 34), torch.randn(4, 2))
    print("GRUD value_mask out", out.shape, "params", count_params(m))
    # utility sanity: 官方钉死值
    y = [np.array([0, 0, 0, 0, 1, 1])]
    p = [np.array([0, 0, 1, 1, 1, 1], dtype=float)]
    from evaluate_sepsis_score import compute_prediction_utility
    print("pinned raw utility:", compute_prediction_utility(y[0], (p[0] >= 0.5).astype(int)))
