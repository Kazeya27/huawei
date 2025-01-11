import numpy as np
import torch

from torch import nn
from pdb import set_trace

from libcity.model.mixer.basic_mixer_block import MixerBlock, PoolMixerBlock, ParamMixerBlock, FeatureMixMixerBlock, \
    FeatureMixPoolMixerBlock
from libcity.model.mixer.sf_mixer_layer import SFMixerLayer


class STMixerBlock(nn.Module):
    def __init__(self, config, data_feature, seq_len, act_fn=nn.GELU):
        super().__init__()
        self.config = config
        self.data_feature = data_feature

        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.num_patches = self.config.get('num_patches', [128, 64])
        self.input_window = seq_len
        self.pool_size = self.config.get('pool_size', [16, 16])
        self.n_slayer = self.config.get('n_slayer', 2)
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

        self.time_mixer = FeatureMixMixerBlock(self.input_window, self.hidden, 2 * self.d_model, self.input_window,
                                               self.dropout_rate, self.act_fn, norm=self.norm)
        # self.time_mixer = MixerBlock(self.input_window, self.d_model, self.input_window,
        #                              self.dropout_rate, self.act_fn, norm=self.norm)  # [B, d_model, N, hidden]
        self.feature_mixer = MixerBlock(self.hidden, self.d_model, self.hidden, self.dropout_rate, self.act_fn,
                                        norm=self.norm)  # [B, T, N, hidden]

        self.st_fusion_mixer = FeatureMixMixerBlock(self.input_window, self.num_nodes, self.d_model,
                                                    self.input_window, self.dropout_rate, self.act_fn, norm=self.norm)

        if self.feat_mix:
            self.token_mixer = FeatureMixMixerBlock(self.num_nodes, self.hidden, self.d_model, self.num_nodes,
                                                    self.dropout_rate, self.act_fn, norm=self.norm)
        else:
            self.token_mixer = MixerBlock(self.num_nodes, 2 * self.d_model, self.num_nodes,
                                          self.dropout_rate, self.act_fn, norm=self.norm)


        self.patch_up = nn.ModuleList([MixerBlock(self.num_patches[0], self.d_model, self.num_nodes,
                                                  self.dropout_rate, self.act_fn, norm=self.norm)])

        self.sf_mixer_layers = nn.ModuleList()
        for i in range(self.n_slayer):
            self.sf_mixer_layers.append(
                SFMixerLayer(config, data_feature, self.input_window, self.num_patches[i], self.pool_size[i], self.act_fn)
            )
            if i != self.n_slayer - 1:
                self.patch_up.append(MixerBlock(self.num_patches[i + 1], self.d_model, self.num_patches[i],
                                                  self.dropout_rate, self.act_fn, norm=self.norm))

    def forward(self, x, sub_x_list):
        #  [B, T, N, C]
        # [B, C, T, N]
        y = self.token_mixer(x.permute(0, 3, 1, 2)).permute(0, 1, 3, 2)  # [B, T, N, hidden] 3302
        y = self.time_mixer(y).permute(0, 2, 1, 3)  # [B, N, d_model, hidden] # 4836

        y = self.st_fusion_mixer(y).permute(0, 3, 1, 2)  # [B, d_model, hidden] # 4906
        # y = y.permute(0, 3, 1, 2)

        y = self.feature_mixer(y)  # [B, T, N, hidden] # 5212

        ort_loss = []
        skip = 0
        for i in range(self.n_slayer - 1, -1, -1):
            sub_x = sub_x_list[i].permute(0, 3, 1, 2)
            sub_y, loss = self.sf_mixer_layers[i](sub_x)
            ort_loss.append(loss)
            skip = sub_y + skip
            skip = self.patch_up[i](skip)
        # BCTS -> BTSC
        skip = skip.permute(0, 2, 3, 1)  # [B, t, S, hidden]
        ort_loss = torch.mean(torch.stack(ort_loss))
        return y + skip, ort_loss
