import numpy as np
import torch

from torch import nn
from pdb import set_trace

from libcity.model.mixer.basic_mixer_block import MixerBlock, PoolMixerBlock, ParamMixerBlock, FeatureMixMixerBlock, \
    FeatureMixPoolMixerBlock


class SFMixerLayer(nn.Module):
    def __init__(self, config, data_feature, seq_len, num_nodes, pool_size, act_fn=nn.GELU):
        super().__init__()
        self.config = config
        self.data_feature = data_feature

        self.input_window = seq_len
        self.num_nodes = num_nodes
        self.pool_size = pool_size

        self.bias = self.config.get('bias', False)
        self.method = self.config.get('method', 'attn')
        self.d_model = self.config.get('d_model', 64)
        self.dropout_rate = self.config.get('dropout_rate', 0.1)
        self.mixer_type = self.config.get('mixer', 'pool')
        self.hidden = self.config.get('hidden', 64)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.feat_mix = self.config.get('feat_mix', False)
        self.t_d_model = self.config.get('t_d_model', 32)
        self.act_fn = act_fn
        self.norm = nn.BatchNorm2d

        if self.mixer_type == 'pool':
            self.time_moe = PoolMixerBlock(self.input_window, self.t_d_model, self.input_window,
                                           self.pool_size, self.num_nodes,
                                           dropout_rate=self.dropout_rate, method=self.method,
                                           bias=self.bias, act=self.act_fn, norm=self.norm)
        else:
            self.time_moe = ParamMixerBlock(self.input_window, self.t_d_model, self.input_window, self.num_nodes,
                                            dropout_rate=self.dropout_rate, method=self.method,
                                            bias=self.bias, act=self.act_fn, norm=self.norm)

        if self.feat_mix:
            self.token_mixer = FeatureMixMixerBlock(self.num_nodes, self.hidden, self.d_model, self.num_nodes,
                                                        self.dropout_rate, self.act_fn, norm=self.norm)
        else:
            self.token_mixer = MixerBlock(self.num_nodes, self.d_model, self.num_nodes,
                                              self.dropout_rate, self.act_fn, norm=self.norm)

        self.st_fusion_mixer = FeatureMixMixerBlock(self.input_window, self.num_nodes, self.d_model,
                                                    self.input_window, self.dropout_rate, self.act_fn,
                                                    norm=self.norm)

        self.feature_mixer = MixerBlock(self.hidden, self.d_model, self.hidden, self.dropout_rate, self.act_fn,
                                        norm=self.norm)

    def ort_loss(self, x):
        if len(x.shape) == 4:
            x = x.permute(2, 0, 1, 3)
        x = torch.reshape(x, (-1, self.num_nodes, self.d_model))

        # 归一化
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-6)

        # 计算所有向量的相似度矩阵
        sim_matrix = torch.matmul(x_norm, x_norm.transpose(1, 2))  # [num_nodes, num_nodes]

        # 去掉对角线元素
        eye = torch.eye(self.num_nodes, device=x.device)
        sim_matrix = sim_matrix * (1 - eye)
        sim_matrix = torch.abs(sim_matrix)
        # 计算损失
        tmp_loss = torch.sum(sim_matrix) / (self.num_nodes * (self.num_nodes - 1))
        return tmp_loss

    def forward(self, x):
        # [B, C, T, S]
        y = self.token_mixer(x).permute(0, 1, 3, 2)  # BTCSd + BTCdS

        # [B, C, S, T]
        y, w1, w2 = self.time_moe(y)  # BCST 5308

        y = y.permute(0, 2, 1, 3)  # BCST -> BSCT

        # BSCT -> BSTC
        y = self.st_fusion_mixer(y).permute(0, 1, 3, 2)  # 5312
        # y = y.permute(0, 1, 3, 2)

        # BSTC -> BCTS
        y = self.feature_mixer(y).permute(0, 3, 2, 1)  # 5344

        if self.mixer_type == 'pool':
            ort_loss2 = self.ort_loss(w2.permute(0, 1, 3, 2))
            # ort_loss4 = self.ort_loss(w4.permute(0, 1, 3, 2))
        else:
            ort_loss2 = self.ort_loss(w2.permute(0, 2, 1))
            # ort_loss4 = self.ort_loss(w4.permute(0, 2, 1))
        ort_loss1 = self.ort_loss(w1)
        return y, ort_loss1 + ort_loss2