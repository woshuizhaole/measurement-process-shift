#!/usr/bin/env python
"""code/utils/ops_repair.py — OPS-Repair 方法族 (Stage 4, 方案 §7.2 + §6.4)

- MeasurementPropensity: 逐(医院,变量)交叉拟合 e_j^h(C), C=build_phi(仅永远可得通道)。
- msr(): 测量得分残差 rho_j = (M_j - e)/(e(1-e)), clip。
- reanchor(): 协议重锚 mask -> e_ref(C) + (M - e_h(C))。
- OffsetRecalibrator: King-Zeng/DAMS-rmr/logistic 三模式截距校正 (slope=1, 无需目标标签)。
- GradReverse + DualChannel 在 models.py (Stage 4 训练时组合)。
"""
import sys, numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "./code/utils")
import features as F


class MeasurementPropensity(BaseEstimator):
    """逐变量 e_j(C)=P(M_j=1|C)。C 只含永远可得通道(build_phi)。5 折按病人交叉拟合(out-of-fold)。"""
    def __init__(self, n_splits=5, C=1.0, eps=0.02):
        self.n_splits = n_splits; self.C = C; self.eps = eps

    def fit_transform_oof(self, PH, MM, groups, also_full=True, lag=None):
        """返回 out-of-fold e (行,34)。also_full=True 同时训练全量模型存 self.models_(省一次 fit)。
        PH:(rows,d_phi) MM:(rows,34) groups:病人id。
        lag:(rows,34) 可选 —— 每变量自身的滞后特征(如 log1p(距上次观测)); 若给, 拟合 e_j 时
            把 lag[:,j] 拼进 PH (方案 §6.5 逐变量滞后观测历史)。self.use_lag_ 记录。"""
        n, dj = MM.shape
        E = np.zeros((n, dj), np.float32)
        gkf = GroupKFold(n_splits=self.n_splits)
        folds = list(gkf.split(PH, MM[:, 0], groups))
        self.models_ = []
        self.use_lag_ = lag is not None
        for j in range(dj):
            col = MM[:, j].astype(int); r = col.mean()
            Pj = PH if lag is None else np.hstack([PH, lag[:, j:j + 1]])
            if r < 0.005 or r > 0.995:
                E[:, j] = np.clip(r, self.eps, 1 - self.eps)
                self.models_.append(("const", float(np.clip(r, self.eps, 1 - self.eps))))
                continue
            for tr, te in folds:
                if col[tr].min() == col[tr].max():
                    E[te, j] = np.clip(col[tr].mean(), self.eps, 1 - self.eps); continue
                lr = LogisticRegression(max_iter=200, C=self.C).fit(Pj[tr], col[tr])
                E[te, j] = lr.predict_proba(Pj[te])[:, 1]
            if also_full:
                self.models_.append(("lr", LogisticRegression(max_iter=200, C=self.C).fit(Pj, col)))
        return np.clip(E, self.eps, 1 - self.eps)

    def fit(self, PH, MM):
        """拟合全量模型(用于对新数据预测, 如目标域)。"""
        self.models_ = []
        for j in range(MM.shape[1]):
            col = MM[:, j].astype(int); r = col.mean()
            if r < 0.005 or r > 0.995:
                self.models_.append(("const", float(np.clip(r, self.eps, 1 - self.eps))))
            else:
                self.models_.append(("lr", LogisticRegression(max_iter=500, C=self.C).fit(PH, col)))
        return self

    def predict_e(self, PH, lag=None):
        E = np.zeros((PH.shape[0], len(self.models_)), np.float32)
        for j, (k, m) in enumerate(self.models_):
            if k == "const":
                E[:, j] = m
            else:
                Pj = PH if lag is None else np.hstack([PH, lag[:, j:j + 1]])
                E[:, j] = m.predict_proba(Pj)[:, 1]
        return np.clip(E, self.eps, 1 - self.eps)


def msr(MM, E, kappa=20.0):
    """测量得分残差 (方案 §6.4)。一阶剥离协议 alpha/beta, 保留 gamma*G。"""
    return np.clip((MM - E) / (E * (1 - E)), -kappa, kappa).astype(np.float32)


def reanchor(MM, E_h, E_ref):
    """协议重锚 (part surgery): mask -> e_ref(C) + (M - e_h(C)). logit空间 ComBat 位置校正。"""
    return (E_ref + (MM - E_h)).astype(np.float32)


class OffsetRecalibrator(BaseEstimator, TransformerMixin):
    """冻结源模型 logit, 只做截距平移 (slope=1 -> 不需目标标签)。
    mode: 'prior'(King-Zeng, 目标患病率 pi_t 由 BBSE 无标签估) | 'rmr'(DAMS 相对率) | 'logistic'(需少量目标标签)."""
    def __init__(self, mode="prior"):
        self.mode = mode

    def fit(self, logit_s=None, y_s=None, pi_t=None, r=None):
        if self.mode == "prior":
            pi_s = float(np.mean(y_s))
            self.delta_ = np.log((pi_t / (1 - pi_t)) / (pi_s / (1 - pi_s)))
        elif self.mode == "rmr":
            self.delta_ = float(np.log(np.clip(1 - r, 1e-6, None)))
        elif self.mode == "logistic":
            import statsmodels.api as sm
            X = sm.add_constant(logit_s)
            res = sm.GLM(y_s, X, family=sm.families.Binomial()).fit()
            self.a_, self.b_ = float(res.params[0]), float(res.params[1])
        return self

    def transform(self, logit):
        if self.mode == "logistic":
            return self.a_ + self.b_ * logit
        return logit + self.delta_

    def predict_proba(self, logit):
        p = 1 / (1 + np.exp(-self.transform(logit)))
        return np.c_[1 - p, p]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    PH = rng.normal(size=(2000, 9)); MM = (rng.random((2000, 34)) < 0.3).astype(float)
    g = np.repeat(np.arange(200), 10)
    mp = MeasurementPropensity()
    E = mp.fit_transform_oof(PH, MM, g)
    R = msr(MM, E)
    print("OOF e shape", E.shape, "rho range", R.min(), R.max())
    oc = OffsetRecalibrator("prior").fit(y_s=(rng.random(1000) < 0.1).astype(int), pi_t=0.05)
    print("offset delta", round(oc.delta_, 4))
