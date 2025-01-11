import torch
import torch.nn.functional as F
import math

from torch import nn
from libcity.model.mixer.weight_pool import WeightPool
from einops import rearrange
from pdb import set_trace


class ParamMixerBlock(nn.Module):
    def __init__(self, input_channel, d_model, output_channel, pool_size, dropout_rate, method='cos', bias=True,
                 act=nn.ReLU, norm=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm(input_channel)
        self.weight1 = nn.Parameter(torch.randn(pool_size, input_channel, d_model), requires_grad=True)
        self.weight2 = nn.Parameter(torch.randn(pool_size, d_model, output_channel), requires_grad=True)
        self.bias = bias
        if bias is True:
            self.b1 = nn.Parameter(torch.randn(pool_size, d_model), requires_grad=True)
            self.b2 = nn.Parameter(torch.randn(pool_size, d_model), requires_grad=True)
        self.act = act
        self.dropout = nn.Dropout(dropout_rate)
        self.projection = (
            nn.Linear(input_channel, output_channel)
            if input_channel != output_channel else nn.Identity()
        )
        self.init_parameter()

    def init_parameter(self):
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        # nn.init.xavier_uniform_(self.weight1)
        # nn.init.xavier_uniform_(self.weight2)
        if self.bias is True:
            nn.init.xavier_uniform_(self.b1)
            nn.init.xavier_uniform_(self.b2)

    def forward(self, x):
        x_proj = self.projection(x)
        y = self.norm1(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        y = torch.einsum('bcst, std->bcsd', y, self.weight1)
        # todo b1的shape可能有问题的
        if self.bias is True:
            y = y + self.b1
        y = self.act(y)
        y = self.dropout(y)
        y = torch.einsum('bcsd, sdt->bcst', y, self.weight2)
        if self.bias is True:
            y = y + self.b2
        y = self.dropout(y)
        return x_proj + y, self.weight1, self.weight2


class FeatureMixPoolMixerBlock(nn.Module):
    def __init__(self, input_channel, hidden, d_model, output_channel, pool_size, num_patches, dropout_rate, method='cos',
                 bias=True, act=nn.ReLU, norm=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm(input_channel)
        self.weight_pool1 = WeightPool(input_channel * hidden, d_model, pool_size, num_patches, method, bias)
        self.weight_pool2 = WeightPool(d_model, output_channel * hidden, pool_size, num_patches, method, bias)
        self.act = act
        self.dropout = nn.Dropout(dropout_rate)
        self.projection = (
            nn.Linear(input_channel, output_channel)
            if input_channel != output_channel else nn.Identity()
        )

    def forward(self, x):
        B, C, S, T = x.shape
        x_proj = self.projection(x)
        y = self.norm1(x.permute(0, 3, 2, 1))
        y = rearrange(y, 'b t s c -> b s (t c)')
        w1, b1 = self.weight_pool1(y)  # [B, C, S, d, T]
        y = torch.einsum('bst, bsdt->bsd', y, w1)
        if b1 is not None:
            y = y + b1
        y = self.act(y)
        y = self.dropout(y)
        w2, b2 = self.weight_pool2(y)
        y = torch.einsum('bsd, bstd->bst', y, w2)
        if b2 is not None:
            y = y + b2
        y = self.dropout(y)
        y = rearrange(y, 'b s (t c) -> b c s t', c=C)
        return x_proj + y, w1, w2


class PoolMixerBlock(nn.Module):
    def __init__(self, input_channel, d_model, output_channel, pool_size, num_patches, dropout_rate, method='cos',
                 bias=True, act=nn.ReLU, norm=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm(input_channel)
        self.weight_pool1 = WeightPool(input_channel, d_model, pool_size, num_patches, method, bias)
        self.weight_pool2 = WeightPool(d_model, output_channel, pool_size, num_patches, method, bias)
        self.act = act
        self.dropout = nn.Dropout(dropout_rate)
        self.projection = (
            nn.Linear(input_channel, output_channel)
            if input_channel != output_channel else nn.Identity()
        )

    def forward(self, x):
        x_proj = self.projection(x)
        y = self.norm1(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        w1, b1 = self.weight_pool1(y)  # [B, C, S, d, T]
        y = torch.einsum('bcst, bsdt->bcsd', y, w1)
        if b1 is not None:
            y = y + b1
        y = self.act(y)
        y = self.dropout(y)
        w2, b2 = self.weight_pool2(y)
        y = torch.einsum('bcsd, bstd->bcst', y, w2)
        if b2 is not None:
            y = y + b2
        y = self.dropout(y)
        return x_proj + y, w1, w2


class FeatureMixMixerBlock(nn.Module):
    def __init__(self, input_channel, hidden, d_model, output_channel, dropout_rate, act=nn.ReLU, norm=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm(input_channel)
        # [b, n, t, h] [h,d] -> [b, n, t, d]
        # [b, n, (t h)] [(t h) d] -> [b, n ,d]
        self.fc1 = nn.Linear(input_channel * hidden, d_model)
        self.fc2 = nn.Linear(d_model, output_channel * hidden)
        self.act = act
        self.dropout = nn.Dropout(dropout_rate)
        self.projection = (
            nn.Linear(input_channel, output_channel)
            if input_channel != output_channel else nn.Identity()
        )

    def forward(self, x):
        B, C, N, T = x.shape
        x_proj = self.projection(x)
        y = self.norm1(x.permute(0, 3, 2, 1))  # [B, N, T, C]
        y = rearrange(y, 'b t n c -> b n (t c)')
        # [t, d]  [t*d, d]
        y = self.act(self.fc1(y))  # [B, T, N*C] @ [N*C,d]
        y = self.dropout(y)
        y = self.fc2(y)  # [B, T, d] @ [d, N*C]
        y = rearrange(y, 'b n (t c) -> b c n t', c=C)
        y = self.dropout(y)
        return x_proj + y


class MixerBlock(nn.Module):
    def __init__(self, input_channel, d_model, output_channel, dropout_rate, act=nn.ReLU, norm=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm(input_channel)
        self.fc1 = nn.Linear(input_channel, d_model)
        self.fc2 = nn.Linear(d_model, output_channel)
        self.act = act
        self.dropout = nn.Dropout(dropout_rate)
        self.projection = (
            nn.Linear(input_channel, output_channel)
            if input_channel != output_channel else nn.Identity()
        )

    def forward(self, x):
        x_proj = self.projection(x)
        y = self.norm1(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        y = self.act(self.fc1(y))  # [B, C, T, N] @ [N,d]
        y = self.dropout(y)
        y = self.fc2(y)  # [B, C, T, d] @ [d, N]
        y = self.dropout(y)
        return x_proj + y

