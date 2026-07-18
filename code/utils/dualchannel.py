#!/usr/bin/env python
"""code/utils/dualchannel.py — 双通道编码器 + 仅测量通道的梯度反转 (Stage 4, 方案 §7.2 R3)

value 通道(GRU-D decay值)与 measurement 通道(mask/MSR + delta)分离编码。
对抗器只作用在 measurement 通道表征 h_meas 上, 值通道不加对抗 -> 只清洗测量指纹、保留生理。
损失 = BCE(logit, y)[源标签] + lambda_adv*CE(hosp, GRL(h_meas))[源+目标域标签] (+可选 CLUB/HSIC)。
meas_in 可为 raw mask / MSR残差 / 重锚 mask (由 Cache 的 meas_override 传入)。
"""
import torch
import torch.nn as nn


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)


class DualChannel(nn.Module):
    """value 通道 + measurement 通道; 对抗器仅在 h_meas。
    meas_kind: 'mask'(raw) 决定 measurement 通道输入是 mask 还是外部传入的 meas_override(MSR/重锚)。"""
    def __init__(self, n_ch=34, n_static=2, h_val=96, h_meas=48, n_hosp=2,
                 grl_lambda=1.0, dropout=0.2):
        super().__init__()
        self.decay = nn.Linear(n_ch, n_ch)
        self.gru_val = nn.GRU(n_ch + n_static, h_val, batch_first=True)
        self.gru_meas = nn.GRU(2 * n_ch, h_meas, batch_first=True)     # meas_in + delta
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h_val + h_meas, 64),
                                  nn.ReLU(), nn.Linear(64, 1))
        self.disc = nn.Sequential(nn.Linear(h_meas, 64), nn.ReLU(), nn.Linear(64, n_hosp))
        self.grl_lambda = grl_lambda

    def forward(self, x_hat, mask, delta, static, meas_in=None, hosp=None):
        B, T, _ = mask.shape
        g = torch.exp(-torch.relu(self.decay(delta)))
        x_dec = mask * x_hat + (1 - mask) * (g * x_hat)
        st = static.unsqueeze(1).expand(-1, T, -1)
        hv, _ = self.gru_val(torch.cat([x_dec, st], -1))
        meas = mask if meas_in is None else meas_in
        hm, _ = self.gru_meas(torch.cat([meas, delta], -1))
        # 取每个时间步的输出 (逐小时预测)
        logit = self.head(torch.cat([hv, hm], -1)).squeeze(-1)         # (B,T)
        out = {"logit": logit, "h_meas": hm}
        if hosp is not None:
            # 对抗: 用序列末隐状态(池化)判别医院
            hm_pool = hm.mean(dim=1)                                   # (B,h_meas)
            out["hosp_logit"] = self.disc(grad_reverse(hm_pool, self.grl_lambda))
        return out


if __name__ == "__main__":
    m = DualChannel()
    x = torch.randn(4, 20, 34); mask = (torch.rand(4, 20, 34) > .5).float()
    d = torch.rand(4, 20, 34); st = torch.randn(4, 2)
    o = m(x, mask, d, st, hosp=torch.tensor([0, 1, 0, 1]))
    print("logit", o["logit"].shape, "hosp_logit", o["hosp_logit"].shape,
          "params", sum(p.numel() for p in m.parameters()))
