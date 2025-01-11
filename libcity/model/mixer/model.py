from logging import getLogger

import torch
import torch.nn as nn

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model.mixer.ST_mixer_block import STMixerBlock
from libcity.model.mixer.basic_mixer_block import MixerBlock
from libcity.model.mixer.data_embedding import DataEmbedding
from functools import partial
import torch.nn.functional as F
from pdb import set_trace

from libcity.model.mixer.fusion_block import FusionBlock
from libcity.model.mixer.stfusion_mixer import MidTMixerBlock


class Model(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.config = config
        self.data_feature = data_feature
        self.device = config.get('device', torch.device('cpu'))
        # section 1: data_feature
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.ext_dim = self.data_feature.get("ext_dim", 0)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.timeslots = self.data_feature.get('timeslots', 288)
        self.batch_size = config.get('batch_size', 32)
        self.cl_step = config.get('cl_step', 1)
        self.num_batches = self.data_feature.get('num_batches', 1)

        self.num_neighbors = self.config.get('num_neighbors', 5)
        self.hop = self.config.get('hop', 3)
        self.d_model = self.config.get('d_model', 128)
        self.hidden_dim = self.config.get('hidden', 32)
        self.dropout_rate = self.config.get('dropout_rate', 0.1)
        self.act = self.config.get('act', 'relu')
        node_embedding = self.data_feature.get('node_embedding', None)
        self.node_embedding = torch.tensor(node_embedding, device=self.device)
        self.add_time_in_day = self.config.get('add_time_in_day', True)
        self.add_day_in_week = self.config.get('add_day_in_week', True)
        self.num_patches = self.config.get('num_patches', 32)
        self.input_window = self.config.get('input_window', 12)
        self.output_window = self.config.get('output_window', 12)
        self.mask_val = self.config.get('mask_val', None)
        self.n_layer = self.config.get('n_layer', 3)
        self.n_tlayer = self.config.get('n_tlayer', 4)
        self.t_d_model = self.config.get('t_d_model', 32)
        self.set_loss = self.config.get('set_loss', 'masked_mae')
        self.skip_dim = self.config.get('skip_dim', 256)
        self.end_dim = self.config.get('end_dim', 512)
        self.t_patch_size = config.get("t_patch_size", 2)

        self.norm = nn.BatchNorm2d

        self._logger = getLogger()
        self.use_curriculum_learning = config.get('use_curriculum_learning', True)
        self.step_size = config.get('step_size', 2500)
        self.max_epoch = config.get('max_epoch', 200)
        self.task_level = config.get('task_level', 0)
        if self.max_epoch * self.num_batches < self.step_size * self.output_window:
            self._logger.warning('Parameter `step_size` is too big with {} epochs and '
                                 'the model cannot be trained for all time steps.'.format(self.max_epoch))
        if self.use_curriculum_learning:
            self._logger.info('Use use_curriculum_learning!')

        if self.act == 'relu':
            self.act_fn = nn.ReLU()
        elif self.act == 'gelu':
            self.act_fn = nn.GELU()
        elif self.act == 'leakyrelu':
            self.act_fn = nn.LeakyReLU()

        # section 3: model structure
        self.adp_node_emb = nn.Parameter(
            torch.empty(self.num_nodes, self.hidden_dim)
        )
        nn.init.xavier_uniform_(self.adp_node_emb)

        self.input_proj = DataEmbedding(self.feature_dim - self.ext_dim, self.hidden_dim, self.timeslots,
                                        self.dropout_rate, self.add_time_in_day, self.add_day_in_week)

        self.norm_node = nn.BatchNorm2d(self.num_nodes)

        self.norm_feat = nn.BatchNorm2d(self.hidden_dim)

        self.tmixer = MidTMixerBlock(self.config, self.data_feature,
                                     self.input_window, self.t_d_model, self.hidden_dim, self.skip_dim, self.dropout_rate,
                                     self.n_tlayer, self.act_fn, self.norm)

        self.fusion_block = FusionBlock(self.input_window, self.t_patch_size, self.n_tlayer, self.act_fn)
        self.proj_layer = MixerBlock(self.input_window, self.hidden_dim, self.output_dim, self.dropout_rate,
                                     self.act_fn, self.norm)
        self.skipE = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.skip_dim, kernel_size=1, bias=True),
            self.act_fn
        )

        self.predict_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.skip_dim, out_channels=self.end_dim, kernel_size=1, bias=True),
            self.act_fn,
            nn.Conv2d(in_channels=self.end_dim, out_channels=self.output_window, kernel_size=1, bias=True)
        )

    def forward(self, batch):
        x = batch['X']  # 430
        x = self.input_proj(x)  # 598 [B, T, N, hidden]
        x = x + self.adp_node_emb + self.node_embedding
        skip, x_list, ort_loss = self.tmixer(x.permute(0, 3, 2, 1))
        x = self.fusion_block(x_list)
        x = self.proj_layer(x)
        x = self.skipE(x)
        x = x + skip
        x = self.predict_layer(x)
        return x, ort_loss

    def predict(self, batch):
        x, ort_loss = self.forward(batch)
        return x

    def calculate_loss_without_predict(self, y_true, y_predicted, batches_seen=None):
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        lf = self.get_loss_func()
        if self.training:
            if batches_seen % self.step_size == 0 and self.task_level < self.output_window:
                self.task_level += self.cl_step
                self._logger.info('Training: task_level increase from {} to {}'.format(
                    self.task_level - self.cl_step, self.task_level))
                self._logger.info('Current batches_seen is {}'.format(batches_seen))
            if self.use_curriculum_learning:
                return lf(y_predicted[:, :self.task_level, :, :], y_true[:, :self.task_level, :, :])
            else:
                return lf(y_predicted, y_true)
        else:
            return lf(y_predicted, y_true)

    def calculate_loss(self, batch, batches_seen=None):
        y_true = batch['y']
        y_predicted, ort_loss = self.forward(batch)
        pre_loss = self.calculate_loss_without_predict(y_true, y_predicted, batches_seen)

        if self.training:
            if batches_seen % self.step_size == 0 and self.task_level <= self.output_window:
                self._logger.info('pre_loss={}, ort_loss={}'.format(pre_loss, ort_loss))
        return pre_loss + 0.1 * ort_loss

    def get_loss_func(self):
        if self.set_loss.lower() not in ['mae', 'mse', 'rmse', 'mape', 'logcosh', 'huber', 'quantile', 'masked_mae',
                                         'masked_mse', 'masked_rmse', 'masked_mape', 'masked_huber', 'r2', 'evar']:
            self._logger.warning('Received unrecognized train loss function, set default mae loss func.')
        if self.set_loss.lower() == 'mae':
            lf = loss.masked_mae_torch
        elif self.set_loss.lower() == 'mse':
            lf = loss.masked_mse_torch
        elif self.set_loss.lower() == 'rmse':
            lf = loss.masked_rmse_torch
        elif self.set_loss.lower() == 'mape':
            lf = loss.masked_mape_torch
        elif self.set_loss.lower() == 'logcosh':
            lf = loss.log_cosh_loss
        elif self.set_loss.lower() == 'huber':
            lf = partial(loss.huber_loss, delta=2)
        elif self.set_loss.lower() == 'quantile':
            lf = partial(loss.quantile_loss, delta=self.quan_delta)
        elif self.set_loss.lower() == 'masked_mae':
            lf = partial(loss.masked_mae_torch, null_val=0)
        elif self.set_loss.lower() == 'masked_mse':
            lf = partial(loss.masked_mse_torch, null_val=0)
        elif self.set_loss.lower() == 'masked_rmse':
            lf = partial(loss.masked_rmse_torch, null_val=0)
        elif self.set_loss.lower() == 'masked_mape':
            lf = partial(loss.masked_mape_torch, null_val=0)
        elif self.set_loss.lower() == 'masked_huber':
            lf = partial(loss.masked_huber_loss, delta=2, null_val=0)
        else:
            lf = loss.masked_mae_torch
        return lf