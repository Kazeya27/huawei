from pdb import set_trace

import math
import torch
import torch.nn.functional as F
from einops import rearrange

from torch import nn

from libcity.model.mixer.ST_mixer_block import STMixerBlock


class MidTMixerBlock(nn.Module):
    def __init__(self, config, data_feature,
                 window, t_d_model, hidden, skip_dim, dropout_rate, n_layers, act=nn.ReLU, norm=nn.BatchNorm2d):
        super().__init__()
        self.tmixer0_filter = nn.ModuleList()
        self.tmixer0_gate = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.st_mixers = nn.ModuleList()

        self.act = act
        self.n_layers = n_layers
        self.t_patch_size = config.get("t_patch_size", 2)
        self.num_neighbors = config.get('num_neighbors', 5)
        self.device = config.get('device', torch.device('cpu'))
        self.hop = config.get('hop', 3)
        self.d_model = config.get('d_model', 128)
        self.hidden_dim = config.get('hidden', 32)
        self.bias = config.get('bias', False)
        self.method = config.get('method', 'attn')
        self.t_pool_size = config.get('t_pool_size', 12)
        self.num_patches = config.get('num_patches', [128, 64])
        self.pool_size = config.get('pool_size', [16, 16])
        self.n_slayer = config.get('n_slayer', 2)
        self.num_nodes = data_feature.get('num_nodes', 1)

        self.patch_down = nn.ModuleList([nn.Linear(self.num_nodes, self.num_patches[0])])
        for i in range(self.n_slayer - 1):
            self.patch_down.append(nn.Linear(self.num_patches[i], self.num_patches[i + 1]))
        self.adp_sub_node_emb = nn.ParameterList()
        for i in range(self.n_slayer):
            param = nn.Parameter(torch.empty(self.num_patches[i], hidden, device=self.device))
            nn.init.xavier_uniform_(param)
            self.adp_sub_node_emb.append(param)

        # self.sub_linear = nn.Linear(data_feature.get("num_nodes", 1), self.num_patches)
        # self.adp_sub_node_emb = nn.Parameter(
        #     torch.empty(self.num_patches, hidden)
        # )
        # nn.init.xavier_uniform_(self.adp_sub_node_emb)
        window_num = (window + self.t_patch_size - 1) // self.t_patch_size
        for i in range(n_layers):
            self.tmixer0_filter.append(
                FeatureMixBasicTMixerBlock(window, self.t_patch_size, t_d_model, hidden, dropout_rate, window_num // 2,
                                           window_num, self.method, self.bias, act=self.act, norm=norm))
            self.tmixer0_gate.append(
                FeatureMixBasicTMixerBlock(window, self.t_patch_size, t_d_model, hidden, dropout_rate, window_num // 2,
                                           window_num, self.method, self.bias, act=self.act, norm=norm))
            # self.tmixer0_filter.append(BasicTMixerBlock(window, 2, t_d_model, dropout_rate, act, norm))
            # self.tmixer0_gate.append(BasicTMixerBlock(window, 2, t_d_model, dropout_rate, act, norm))
            window_num = (window + self.t_patch_size - 1) // self.t_patch_size
            self.skip_convs.append(nn.Conv2d(in_channels=hidden,
                                             out_channels=skip_dim,
                                             kernel_size=(1, 1)))
            self.st_mixers.append(STMixerBlock(config, data_feature, window, self.act))

    def forward(self, x):
        # [B, C, N, T]
        skip = 0
        ort_loss = 0  # 2836
        res_list = []
        for i in range(self.n_layers):
            # [B, hidden, N, t]
            residual = x
            res_list.append(residual)
            filter = self.tmixer0_filter[i](x)  # [B, hidden, N, t_d_model]
            gate = self.tmixer0_gate[i](x)
            filter = torch.tanh(filter)
            gate = torch.sigmoid(gate)
            x = filter * gate
            s = x  # 2870
            s = self.skip_convs[i](s)  # [B, C, N, T] 3006
            if i > 0:
                skip = skip[..., -s.size(3):]
            skip = s + skip  # 3142
            sub_x_list = [self.patch_down[0](x.permute(0, 3, 1, 2)).permute(0, 1, 3, 2) + self.adp_sub_node_emb[0]]
            for j in range(1, self.n_slayer):
                sub_x = self.patch_down[j](sub_x_list[j - 1].permute(0, 1, 3, 2)).permute(0, 1, 3, 2) + self.adp_sub_node_emb[j]
                sub_x_list.append(sub_x)

            # sub_x = self.sub_linear(x.permute(0, 3, 1, 2)).permute(0, 1, 3, 2) + self.adp_sub_node_emb
            x = x.permute(0, 3, 2, 1)  # 3176
            x, tmp_loss = self.st_mixers[i](x, sub_x_list)
            # BTSC -> BCST
            x = x.permute(0, 3, 2, 1)  # 5446
            x = x + residual[..., -x.size(3):]
            ort_loss = ort_loss + tmp_loss
        res_list.append(x)
        x = self.act(skip)
        return x, res_list, ort_loss


class FeatureMixBasicTMixerBlock(nn.Module):
    def __init__(self, window, patch_len, t_d_model, hidden, dropout_rate, pool_size, num_slices, method="attn",
                 bias=True, act=nn.ReLU, norm=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm(window)
        self.window = window
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(12 * hidden, t_d_model)
        self.position_embedding = PositionalEncoding(t_d_model)
        self.act = act
        self.dropout = nn.Dropout(dropout_rate)
        # self.weight_pool = WeightPool(t_d_model, hidden, pool_size, num_slices, method, bias)
        self.fc = nn.Linear(t_d_model, 12 * hidden)

    def forward(self, x):
        x = self.norm1(x.permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # if self.window % self.patch_len != 0:
        #     gap = self.window % self.patch_len
        #     x = F.pad(x, (0, gap))
        #
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        B, S, C, n = x.shape
        x = rearrange(x, 'b s c n -> b s (c n)')
        y = self.value_embedding(x)
        # y = y + self.position_embedding(y)
        y = self.act(y)  # [B, S, n, t*C]
        y = self.dropout(y)

        # w1, b1 = self.weight_pool(y)
        # y = torch.einsum('bcst, bsdt->bcsd', y, w1)
        # if b1 is not None:
        #     y = y + b1

        y = self.fc(y)  # [B, S, n, C]
        y = rearrange(y, 'b s (c n) -> b c s n', n=n)
        return y


class BasicTMixerBlock(nn.Module):
    def __init__(self, window, patch_len, t_d_model, dropout_rate, act=nn.ReLU, norm=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm(window)
        self.window = window
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, t_d_model)
        self.position_embedding = PositionalEncoding(t_d_model)
        self.act = act
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(t_d_model, 1)

    def forward(self, x):
        x = self.norm1(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        if self.window % self.patch_len != 0:
            gap = self.window % self.patch_len
            x = F.pad(x, (0, gap))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        y = self.value_embedding(x)
        y = y + self.position_embedding(y)
        y = self.act(y)  # [B, C, S, n, t]
        y = self.dropout(y)
        y = self.fc(y).squeeze(-1)
        return y


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # [B, T, N, t, n]
        if len(x.shape) == 5:
            embedding = self.pe[:, :x.size(3)].unsqueeze(0).unsqueeze(0).expand_as(x).detach()
        else:
            embedding = self.pe[:, :x.size(2)].unsqueeze(0).expand_as(x).detach()
        return embedding
