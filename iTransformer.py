import time
from functools import partial
from typing import List, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from scipy.special import eval_legendre
from sympy import Poly, legendre, Symbol, chebyshevt
from torch import Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

class LocalMask():
    def __init__(self, B, L,S,device="cpu"):
        mask_shape = [B, 1, L, S]
        with torch.no_grad():
            self.len = math.ceil(np.log2(L))
            self._mask1 = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
            self._mask2 = ~torch.triu(torch.ones(mask_shape,dtype=torch.bool),diagonal=-self.len).to(device)
            self._mask = self._mask1+self._mask2
    @property
    def mask(self):
        return self._mask


def legendreDer(k, x):
    def _legendre(k, x):
        return (2 * k + 1) * eval_legendre(k, x)

    out = 0
    for i in np.arange(k - 1, -1, -2):
        out += _legendre(i, x)
    return out


def phi_(phi_c, x, lb=0, ub=1):
    mask = np.logical_or(x < lb, x > ub) * 1.0
    return np.polynomial.polynomial.Polynomial(phi_c)(x) * (1 - mask)


def get_phi_psi(k, base):
    x = Symbol('x')
    phi_coeff = np.zeros((k, k))
    phi_2x_coeff = np.zeros((k, k))
    if base == 'legendre':
        for ki in range(k):
            coeff_ = Poly(legendre(ki, 2 * x - 1), x).all_coeffs()
            phi_coeff[ki, :ki + 1] = np.flip(np.sqrt(2 * ki + 1) * np.array(coeff_).astype(np.float64))
            coeff_ = Poly(legendre(ki, 4 * x - 1), x).all_coeffs()
            phi_2x_coeff[ki, :ki + 1] = np.flip(np.sqrt(2) * np.sqrt(2 * ki + 1) * np.array(coeff_).astype(np.float64))

        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))
        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            for i in range(k):
                a = phi_2x_coeff[ki, :ki + 1]
                b = phi_coeff[i, :i + 1]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_) < 1e-8] = 0
                proj_ = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()
                psi1_coeff[ki, :] -= proj_ * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj_ * phi_coeff[i, :]
            for j in range(ki):
                a = phi_2x_coeff[ki, :ki + 1]
                b = psi1_coeff[j, :]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_) < 1e-8] = 0
                proj_ = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()
                psi1_coeff[ki, :] -= proj_ * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj_ * psi2_coeff[j, :]

            a = psi1_coeff[ki, :]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_) < 1e-8] = 0
            norm1 = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()

            a = psi2_coeff[ki, :]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_) < 1e-8] = 0
            norm2 = (prod_ * 1 / (np.arange(len(prod_)) + 1) * (1 - np.power(0.5, 1 + np.arange(len(prod_))))).sum()
            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki, :] /= norm_
            psi2_coeff[ki, :] /= norm_
            psi1_coeff[np.abs(psi1_coeff) < 1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff) < 1e-8] = 0

        phi = [np.poly1d(np.flip(phi_coeff[i, :])) for i in range(k)]
        psi1 = [np.poly1d(np.flip(psi1_coeff[i, :])) for i in range(k)]
        psi2 = [np.poly1d(np.flip(psi2_coeff[i, :])) for i in range(k)]

    elif base == 'chebyshev':
        for ki in range(k):
            if ki == 0:
                phi_coeff[ki, :ki + 1] = np.sqrt(2 / np.pi)
                phi_2x_coeff[ki, :ki + 1] = np.sqrt(2 / np.pi) * np.sqrt(2)
            else:
                coeff_ = Poly(chebyshevt(ki, 2 * x - 1), x).all_coeffs()
                phi_coeff[ki, :ki + 1] = np.flip(2 / np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))
                coeff_ = Poly(chebyshevt(ki, 4 * x - 1), x).all_coeffs()
                phi_2x_coeff[ki, :ki + 1] = np.flip(
                    np.sqrt(2) * 2 / np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))

        phi = [partial(phi_, phi_coeff[i, :]) for i in range(k)]

        x = Symbol('x')
        kUse = 2 * k
        roots = Poly(chebyshevt(kUse, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
        # not needed for our purpose here, we use even k always to avoid
        wm = np.pi / kUse / 2

        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))

        psi1 = [[] for _ in range(k)]
        psi2 = [[] for _ in range(k)]

        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            for i in range(k):
                proj_ = (wm * phi[i](x_m) * np.sqrt(2) * phi[ki](2 * x_m)).sum()
                psi1_coeff[ki, :] -= proj_ * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj_ * phi_coeff[i, :]

            for j in range(ki):
                proj_ = (wm * psi1[j](x_m) * np.sqrt(2) * phi[ki](2 * x_m)).sum()
                psi1_coeff[ki, :] -= proj_ * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj_ * psi2_coeff[j, :]

            psi1[ki] = partial(phi_, psi1_coeff[ki, :], lb=0, ub=0.5)
            psi2[ki] = partial(phi_, psi2_coeff[ki, :], lb=0.5, ub=1)

            norm1 = (wm * psi1[ki](x_m) * psi1[ki](x_m)).sum()
            norm2 = (wm * psi2[ki](x_m) * psi2[ki](x_m)).sum()

            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki, :] /= norm_
            psi2_coeff[ki, :] /= norm_
            psi1_coeff[np.abs(psi1_coeff) < 1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff) < 1e-8] = 0

            psi1[ki] = partial(phi_, psi1_coeff[ki, :], lb=0, ub=0.5 + 1e-16)
            psi2[ki] = partial(phi_, psi2_coeff[ki, :], lb=0.5 + 1e-16, ub=1)

    return phi, psi1, psi2

def get_filter(base, k):
    def psi(psi1, psi2, i, inp):
        mask = (inp <= 0.5) * 1.0
        return psi1[i](inp) * mask + psi2[i](inp) * (1 - mask)

    if base not in ['legendre', 'chebyshev']:
        raise Exception('Base not supported')

    x = Symbol('x')
    H0 = np.zeros((k, k))
    H1 = np.zeros((k, k))
    G0 = np.zeros((k, k))
    G1 = np.zeros((k, k))
    PHI0 = np.zeros((k, k))
    PHI1 = np.zeros((k, k))
    phi, psi1, psi2 = get_phi_psi(k, base)
    if base == 'legendre':
        roots = Poly(legendre(k, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        wm = 1 / k / legendreDer(k, 2 * x_m - 1) / eval_legendre(k - 1, 2 * x_m - 1)

        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki](x_m / 2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m / 2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki]((x_m + 1) / 2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m + 1) / 2) * phi[kpi](x_m)).sum()

        PHI0 = np.eye(k)
        PHI1 = np.eye(k)

    elif base == 'chebyshev':
        x = Symbol('x')
        kUse = 2 * k
        roots = Poly(chebyshevt(kUse, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
        # not needed for our purpose here, we use even k always to avoid
        wm = np.pi / kUse / 2

        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki](x_m / 2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m / 2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki]((x_m + 1) / 2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m + 1) / 2) * phi[kpi](x_m)).sum()

                PHI0[ki, kpi] = (wm * phi[ki](2 * x_m) * phi[kpi](2 * x_m)).sum() * 2
                PHI1[ki, kpi] = (wm * phi[ki](2 * x_m - 1) * phi[kpi](2 * x_m - 1)).sum() * 2

        PHI0[np.abs(PHI0) < 1e-8] = 0
        PHI1[np.abs(PHI1) < 1e-8] = 0

    H0[np.abs(H0) < 1e-8] = 0
    H1[np.abs(H1) < 1e-8] = 0
    G0[np.abs(G0) < 1e-8] = 0
    G1[np.abs(G1) < 1e-8] = 0

    return H0, H1, G0, G1, PHI0, PHI1

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class MultiWaveletTransform(nn.Module):
    """
    1D multiwavelet block.
    """

    def __init__(self, ich=1, k=8, alpha=16, c=128,
                 nCZ=1, L=0, base='legendre', attention_dropout=0.1):
        super(MultiWaveletTransform, self).__init__()
        print('base', base)
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ
        self.Lk0 = nn.Linear(ich, c * k)
        self.Lk1 = nn.Linear(c * k, ich)
        self.ich = ich
        self.MWT_CZ = nn.ModuleList(MWT_CZ1d(k, alpha, L, c, base) for i in range(nCZ))

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
        values = values.view(B, L, -1)

        V = self.Lk0(values).view(B, L, self.c, -1)
        for i in range(self.nCZ):
            V = self.MWT_CZ[i](V)
            if i < self.nCZ - 1:
                V = F.relu(V)

        V = self.Lk1(V.view(B, L, -1))
        V = V.view(B, L, -1, D)
        return (V.contiguous(), None)


class MultiWaveletCross(nn.Module):
    """
    1D Multiwavelet Cross Attention layer.
    """

    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes, c=64,
                 k=8, ich=512,
                 L=0,
                 base='legendre',
                 mode_select_method='random',
                 initializer=None, activation='tanh',
                 **kwargs):
        super(MultiWaveletCross, self).__init__()
        print('base', base)

        self.c = c
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        self.attn1 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.attn2 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.attn3 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.attn4 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.T0 = nn.Linear(k, k)
        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((G0.T, G1.T), axis=0)))

        self.register_buffer('rc_e', torch.Tensor(
            np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(
            np.concatenate((H1r, G1r), axis=0)))

        self.Lk = nn.Linear(ich, c * k)
        self.Lq = nn.Linear(ich, c * k)
        self.Lv = nn.Linear(ich, c * k)
        self.out = nn.Linear(c * k, ich)
        self.modes1 = modes

    def forward(self, q, k, v, mask=None):
        B, N, H, E = q.shape  # (B, N, H, E) torch.Size([3, 768, 8, 2])
        _, S, _, _ = k.shape  # (B, S, H, E) torch.Size([3, 96, 8, 2])

        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)
        q = self.Lq(q)
        q = q.view(q.shape[0], q.shape[1], self.c, self.k)
        k = self.Lk(k)
        k = k.view(k.shape[0], k.shape[1], self.c, self.k)
        v = self.Lv(v)
        v = v.view(v.shape[0], v.shape[1], self.c, self.k)

        if N > S:
            zeros = torch.zeros_like(q[:, :(N - S), :]).float()
            v = torch.cat([v, zeros], dim=1)
            k = torch.cat([k, zeros], dim=1)
        else:
            v = v[:, :N, :, :]
            k = k[:, :N, :, :]

        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_q = q[:, 0:nl - N, :, :]
        extra_k = k[:, 0:nl - N, :, :]
        extra_v = v[:, 0:nl - N, :, :]
        q = torch.cat([q, extra_q], 1)
        k = torch.cat([k, extra_k], 1)
        v = torch.cat([v, extra_v], 1)

        Ud_q = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_k = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_v = torch.jit.annotate(List[Tuple[Tensor]], [])

        Us_q = torch.jit.annotate(List[Tensor], [])
        Us_k = torch.jit.annotate(List[Tensor], [])
        Us_v = torch.jit.annotate(List[Tensor], [])

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

        # decompose
        for i in range(ns - self.L):
            # print('q shape',q.shape)
            d, q = self.wavelet_transform(q)
            Ud_q += [tuple([d, q])]
            Us_q += [d]
        for i in range(ns - self.L):
            d, k = self.wavelet_transform(k)
            Ud_k += [tuple([d, k])]
            Us_k += [d]
        for i in range(ns - self.L):
            d, v = self.wavelet_transform(v)
            Ud_v += [tuple([d, v])]
            Us_v += [d]
        for i in range(ns - self.L):
            dk, sk = Ud_k[i], Us_k[i]
            dq, sq = Ud_q[i], Us_q[i]
            dv, sv = Ud_v[i], Us_v[i]
            Ud += [self.attn1(dq[0], dk[0], dv[0], mask)[0] + self.attn2(dq[1], dk[1], dv[1], mask)[0]]
            Us += [self.attn3(sq, sk, sv, mask)[0]]
        v = self.attn4(q, k, v, mask)[0]

        # reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            v = v + Us[i]
            v = torch.cat((v, Ud[i]), -1)
            v = self.evenOdd(v)
        v = self.out(v[:, :N, :, :].contiguous().view(B, N, -1))
        return (v.contiguous(), None)

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :],
                        x[:, 1::2, :, :],
                        ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        B, N, c, ich = x.shape  # (B, N, c, k)
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(B, N * 2, c, self.k,
                        device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x


class FourierCrossAttentionW(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=16, activation='tanh',
                 mode_select_method='random'):
        super(FourierCrossAttentionW, self).__init__()
        print('corss fourier correlation used!')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes
        self.activation = activation

    def forward(self, q, k, v, mask):
        B, L, E, H = q.shape

        xq = q.permute(0, 3, 2, 1)  # size = [B, H, E, L] torch.Size([3, 8, 64, 512])
        xk = k.permute(0, 3, 2, 1)
        xv = v.permute(0, 3, 2, 1)
        self.index_q = list(range(0, min(int(L // 2), self.modes1)))
        self.index_k_v = list(range(0, min(int(xv.shape[3] // 2), self.modes1)))

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]

        xk_ft_ = torch.zeros(B, H, E, len(self.index_k_v), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_k_v):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)

        xqkvw = xqkv_ft
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]

        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1)).permute(0, 3, 2, 1)
        # size = [B, L, H, E]
        return (out, None)


class sparseKernelFT1d(nn.Module):
    def __init__(self,
                 k, alpha, c=1,
                 nl=1,
                 initializer=None,
                 **kwargs):
        super(sparseKernelFT1d, self).__init__()

        self.modes1 = alpha
        self.scale = (1 / (c * k * c * k))
        self.weights1 = nn.Parameter(self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.cfloat))
        self.weights1.requires_grad = True
        self.k = k

    def compl_mul1d(self, x, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", x, weights)

    def forward(self, x):
        B, N, c, k = x.shape  # (B, N, c, k)

        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        x_fft = torch.fft.rfft(x)
        # Multiply relevant Fourier modes
        l = min(self.modes1, N // 2 + 1)
        # l = N//2+1
        out_ft = torch.zeros(B, c * k, N // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :l] = self.compl_mul1d(x_fft[:, :, :l], self.weights1[:, :, :l])
        x = torch.fft.irfft(out_ft, n=N)
        x = x.permute(0, 2, 1).view(B, N, c, k)
        return x


# ##
class MWT_CZ1d(nn.Module):
    def __init__(self,
                 k=3, alpha=64,
                 L=0, c=1,
                 base='legendre',
                 initializer=None,
                 **kwargs):
        super(MWT_CZ1d, self).__init__()

        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        self.A = sparseKernelFT1d(k, alpha, c)
        self.B = sparseKernelFT1d(k, alpha, c)
        self.C = sparseKernelFT1d(k, alpha, c)

        self.T0 = nn.Linear(k, k)

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((G0.T, G1.T), axis=0)))

        self.register_buffer('rc_e', torch.Tensor(
            np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(
            np.concatenate((H1r, G1r), axis=0)))

    def forward(self, x):
        B, N, c, k = x.shape  # (B, N, k)
        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_x = x[:, 0:nl - N, :, :]
        x = torch.cat([x, extra_x], 1)
        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])
        #         decompose
        for i in range(ns - self.L):
            # print('x shape',x.shape)
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x)  # coarsest scale transform

        #        reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)
        x = x[:, :N, :, :]

        return x

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :],
                        x[:, 1::2, :, :],
                        ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):

        B, N, c, ich = x.shape  # (B, N, c, k)
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(B, N * 2, c, self.k,
                        device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


# ########## fourier layer #############
class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        print('fourier enhanced block used!')
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        # get modes on frequency domain
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        print('modes={}, index={}'.format(modes, self.index))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        # Perform Fourier neural operations
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        # Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return (x, None)


# ########## Fourier Cross Former ####################
class FourierCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random',
                 activation='tanh', policy=0):
        super(FourierCrossAttention, self).__init__()
        print(' fourier enhanced cross attention used!')
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        # get modes for queries and keys (& values) on frequency domain
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)

        print('modes_q={}, index_q={}'.format(len(self.index_q), self.index_q))
        print('modes_kv={}, index_kv={}'.format(len(self.index_kv), self.index_kv))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index_q), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        # perform attention mechanism on frequency domain
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        # Return to time domain
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        return (out, None)


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean


class FourierDecomp(nn.Module):
    def __init__(self):
        super(FourierDecomp, self).__init__()
        pass

    def forward(self, x):
        x_ft = torch.fft.rfft(x, dim=-1)


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
            self.decomp3 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)
            self.decomp3 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])

        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend

def decor_time(func):
    def func2(*args, **kw):
        now = time.time()
        y = func(*args, **kw)
        t = time.time() - now
        print('call <{}>, time={}'.format(func.__name__, t))
        return y
    return func2


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False, configs=None):
        super(AutoCorrelation, self).__init__()
        print('Autocorrelation used !')
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.agg = None
        # self.use_wavelet = configs.wavelet

    # @decor_time
    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg  # size=[B, H, d, S]

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )

        out = out.view(B, L, -1)
        return self.out_projection(out), attn

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)

class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # try:
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        # except:
        #     a = 1
        return self.dropout(x)

class FEDformer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, configs, data_feature):
        super(FEDformer, self).__init__()
        self.version = configs.get("version", 'Fourier')
        self.mode_select = configs.get("mode_select", 'random')
        self.modes = configs.get("modes", 64)
        self.seq_len = configs.get("seq_len", 6)
        self.label_len = configs.get("pred_len", 6)
        self.pred_len = configs.get("pred_len", 6)
        self.output_attention = configs.get("output_attention", False)

        # Decomp
        kernel_size = configs.get("kernel_size", [24])
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_in = data_feature.get("num_nodes")
        self.d_model = configs.get("d_model", 512)
        self.embed = configs.get("embed", 'timeF')
        self.freq = configs.get("freq", 'h')
        self.dropout = configs.get("dropout", 0.05)
        self.enc_embedding = DataEmbedding_wo_pos(self.enc_in, self.d_model, self.embed, self.freq,
                                                  self.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(self.dec_in, self.d_model, self.embed, self.freq,
                                                  self.dropout)
        self.L = configs.get("L", 3)
        self.base = configs.get("base", 'legendre')
        self.cross_activation = configs.get("cross_activation", 'tanh')
        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=self.d_model, L=self.L, base=self.base)
            decoder_self_att = MultiWaveletTransform(ich=self.d_model, L=self.L, base=self.base)
            decoder_cross_att = MultiWaveletCross(in_channels=self.d_model,
                                                  out_channels=self.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=self.modes,
                                                  ich=self.d_model,
                                                  base=self.base,
                                                  activation=self.cross_activation)
        else:
            encoder_self_att = FourierBlock(in_channels=self.d_model,
                                            out_channels=self.d_model,
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_self_att = FourierBlock(in_channels=self.d_model,
                                            out_channels=self.d_model,
                                            seq_len=self.seq_len//2+self.pred_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=self.d_model,
                                                      out_channels=self.d_model,
                                                      seq_len_q=self.seq_len//2+self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=self.modes,
                                                      mode_select_method=self.mode_select)
        # Encoder
        enc_modes = int(min(self.modes, self.seq_len//2))
        dec_modes = int(min(self.modes, (self.seq_len//2+self.pred_len)//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.n_heads = configs.get("n_heads", 8)
        self.d_ff = configs.get("d_ff", 2048)
        self.moving_avg = configs.get("moving_avg", [24])
        self.activation = configs.get("activation", 'gelu')
        self.e_layers = configs.get("e_layers", 2)
        self.d_layers = configs.get("d_layers", 1)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        self.d_model, self.n_heads),

                    self.d_model,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model)
        )
        # Decoder
        self.c_out = self.enc_in
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        self.d_model, self.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.c_out,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
            norm_layer=my_Layernorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )

    def forward(self, x_enc, x_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        x_mark_enc = x_enc[:, :, -1, 1:]
        x_enc = x_enc[:, :, :, :, 0]
        x_mark_dec = x_dec[:, :, -1, 1:]
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
