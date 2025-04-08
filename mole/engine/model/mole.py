import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace

class HeadDropout(nn.Module):
    def __init__(self, p=0.5):
        super(HeadDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p

    def forward(self, x):
        # If in evaluation mode, return the input as-is
        if not self.training:
            return x

        # Create a binary mask of the same shape as x
        binary_mask = (torch.rand_like(x) > self.p).float()

        # Set dropped values to negative infinity during training
        return x * binary_mask + (1 - binary_mask) * -1e20


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)

        elif mode == 'denorm':
            x = self._denormalize(x)

        else:
            raise NotImplementedError

        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean

        return x

class MOLE(nn.Module):
    def __init__(self, config):
        super(MOLE, self).__init__()
        self.configs = config

        self.num_predictions = int(config.get("t_dim", 4))
        self.seq_len = int(config.get("seq_len", 12))
        self.pred_len = int(config.get("pred_len", 12))
        self.channels = int(config.get("num_nodes"))
        self.d_model = int(config.get("d_model", 512))
        self.drop = float(config.get("dropout", 0.1))
        self.disable_rev = bool(config.get("disable_rev", False))
        self.head_dropout = float(config.get("head_dropout", 0))
        # set_trace()
        self.Linear = nn.Linear(self.seq_len, self.pred_len * self.num_predictions)

        self.temporal = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.seq_len)
        )

        self.dropout = nn.Dropout(self.drop)
        self.rev = RevIN(self.channels) if not self.disable_rev else None

        self.Linear_Temporal = nn.Sequential(
            nn.Linear(4, self.num_predictions * self.channels),
            nn.ReLU(),
            nn.Linear(self.num_predictions * self.channels, self.num_predictions * self.channels)
        )
        self.head_dropout = HeadDropout(self.head_dropout)

    def forward(self, batch):
        # x: [B, L, D]
        # set_trace()
        x = batch[:, :, :, 0].permute(0, 2, 1)
        x_mark = batch[:, 0, :, 1:]
        x_mark_initial = x_mark[:, 0, :]
        x = self.rev(x, 'norm') if self.rev else x
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        # set_trace()
        temporal_out = self.Linear_Temporal(x_mark_initial).reshape(-1, self.num_predictions, self.channels)
        temporal_out = self.head_dropout(temporal_out)
        temporal_out = nn.Softmax(dim=1)(temporal_out)
        pred_raw = pred.permute(0, 2, 1).reshape(-1, self.channels, self.pred_len, self.num_predictions).permute(0, 3,
                                                                                                                 1, 2)
        pred = pred_raw * temporal_out.unsqueeze(-1)
        pred = pred.sum(dim=1).permute(0, 2, 1)

        pred = self.rev(pred, 'denorm') if self.rev else pred
        pred = pred.permute(0, 2, 1)
        # pred = pred.unsqueeze(-1)
        # set_trace()
        return pred

    def predict(self, batch):
        return self.forward(batch['X'])

    def get_n_param(self):
        n_param = 0
        for param in self.parameters():
            if param.requires_grad:
                n_param += torch.numel(param)
        return n_param