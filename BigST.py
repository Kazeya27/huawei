import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss

class BigST(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.device = config.get('device', torch.device('cpu'))
        # section 1: data_feature
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._logger = getLogger()

        self.his_length = config.get('input_window', 12)
        self.pred_length = config.get('output_window', 12)

        self.tau = config.get('tau', 0.25)
        self.num_layers = config.get('num_layers', 3)
        self.random_feature_dim = config.get('random_feature_dim', 64)

        self.use_residual = config.get('use_residual', True)
        self.use_bn = config.get('use_bn', True)
        self.use_spatial = config.get('use_spatial', True)
        self.use_long = config.get('use_long', False)

        self.dropout = config.get('dropout', 0.3)
        self.activation = nn.ReLU()
        supports = data_feature.get('supports', None)
        self.supports = [torch.tensor(i).to(self.device) for i in supports]
        self.edge_indices = data_feature.get('edge_indices', None)
        # 转为tensor
        self.edge_indices = torch.tensor(self.edge_indices).to(self.device)

        self.time_num = data_feature.get('time_num', 288)
        self.week_num = 7
        self.node_dim = config.get('node_dim', 32)
        self.time_dim = config.get('time_dim', 32)
        self.output_window = config.get('output_window', 12)
        self.hid_dim = config.get('hid_dim', 32)

        # node embedding layer
        self.node_emb_layer = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
        nn.init.xavier_uniform_(self.node_emb_layer)

        # time embedding layer
        self.time_emb_layer = nn.Parameter(torch.empty(self.time_num, self.time_dim))
        nn.init.xavier_uniform_(self.time_emb_layer)
        self.week_emb_layer = nn.Parameter(torch.empty(self.week_num, self.time_dim))
        nn.init.xavier_uniform_(self.week_emb_layer)

        # embedding layer
        self.input_emb_layer = nn.Conv2d(self.output_window * self.feature_dim, self.hid_dim, kernel_size=(1, 1), bias=True)

        self.W_1 = nn.Conv2d(self.node_dim + self.time_dim * 2, self.hid_dim, kernel_size=(1, 1), bias=True)
        self.W_2 = nn.Conv2d(self.node_dim + self.time_dim * 2, self.hid_dim, kernel_size=(1, 1), bias=True)

        self.linear_conv = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.supports_len = 0
        if self.supports is not None:
            self.supports_len += len(self.supports)

        for i in range(self.num_layers):
            self.linear_conv.append(
                linearized_conv(self.hid_dim * 4, self.hid_dim * 4, self.dropout, self.tau, self.random_feature_dim))
            self.bn.append(nn.LayerNorm(self.hid_dim * 4))

        if self.use_long:
            self.regression_layer = nn.Conv2d(self.hid_dim * 4 * 2 + self.hid_dim + self.output_window,
                                              self.output_window, kernel_size=(1, 1), bias=True)
        else:
            self.regression_layer = nn.Conv2d(self.hid_dim * 4 * 2, self.output_window, kernel_size=(1, 1), bias=True)

    def forward(self, x, feat=None):
        # input: (B, N, T, D)
        x = x.permute(0, 2, 1, 3)
        B, N, T, D = x.size()

        time_emb = self.time_emb_layer[(x[:, :, -1, 1] * self.time_num).type(torch.LongTensor)]
        week_emb = self.week_emb_layer[(x[:, :, -1, 2]).type(torch.LongTensor)]

        # input embedding
        x = x.contiguous().view(B, N, -1).transpose(1, 2).unsqueeze(-1)  # (B, D*T, N, 1)
        input_emb = self.input_emb_layer(x)

        # node embeddings
        node_emb = self.node_emb_layer.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1)  # (B, dim, N, 1)

        # time embeddings
        time_emb = time_emb.transpose(1, 2).unsqueeze(-1)  # (B, dim, N, 1)
        week_emb = week_emb.transpose(1, 2).unsqueeze(-1)  # (B, dim, N, 1)

        x_g = torch.cat([node_emb, time_emb, week_emb], dim=1)  # (B, dim*4, N, 1)
        x = torch.cat([input_emb, node_emb, time_emb, week_emb], dim=1)  # (B, dim*4, N, 1)

        # linearized spatial convolution
        x_pool = [x]  # (B, dim*4, N, 1)
        node_vec1 = self.W_1(x_g)  # (B, dim, N, 1)
        node_vec2 = self.W_2(x_g)  # (B, dim, N, 1)
        node_vec1 = node_vec1.permute(0, 2, 3, 1)  # (B, N, 1, dim)
        node_vec2 = node_vec2.permute(0, 2, 3, 1)  # (B, N, 1, dim)
        for i in range(self.num_layers):
            if self.use_residual:
                residual = x
            x, node_vec1_prime, node_vec2_prime = self.linear_conv[i](x, node_vec1, node_vec2)

            if self.use_residual:
                x = x + residual

            if self.use_bn:
                x = x.permute(0, 2, 3, 1)  # (B, N, 1, dim*4)
                x = self.bn[i](x)
                x = x.permute(0, 3, 1, 2)

        x_pool.append(x)
        x = torch.cat(x_pool, dim=1)  # (B, dim*4, N, 1)

        x = self.activation(x)  # (B, dim*4, N, 1)

        if self.use_long:
            feat = feat.permute(0, 2, 1).unsqueeze(-1)  # (B, F, N, 1)
            x = torch.cat([x, feat], dim=1)
            x = self.regression_layer(x)  # (B, N, T)
        else:
            x = self.regression_layer(x)  # (B, N, T)

        if self.use_spatial:
            s_loss = spatial_loss(node_vec1_prime, node_vec2_prime, self.supports, self.edge_indices)
            return x, s_loss
        else:
            return x, 0

    def predict(self, batch):
        x = batch['X']  # input data
        x, s_loss = self.forward(x)
        return x

    def calculate_loss(self, batch):
        y_true = batch['y']  # ground-truth value
        x = batch['X']  # input data
        y_predicted, s_loss = self.forward(x)  # prediction results
        # denormalization the value
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        if self.use_spatial:
            l = loss.masked_mae_torch(y_predicted, y_true, null_val=0.0)- 0.3 * s_loss
        else:
            l = loss.masked_mae_torch(y_predicted, y_true, null_val=0.0)
        return l

    def calculate_loss_without_predict(self, y_true, y_predicted):
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, null_val=0.0)


def create_products_of_givens_rotations(dim, seed):
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    np.random.seed(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.cos(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return torch.tensor(q, dtype=torch.float32)


def create_random_matrix(m, d, seed=0, scaling=0, struct_mode=False):
    nb_full_blocks = int(m / d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q)
        current_seed += 1
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q[0:remaining_rows])
    final_matrix = torch.vstack(block_list)

    current_seed += 1
    torch.manual_seed(current_seed)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

    return torch.matmul(torch.diag(multiplier), final_matrix)


def random_feature_map(data, is_query, projection_matrix=None, numerical_stabilizer=0.000001):
    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32))
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=len(data.shape) - 1)
    diag_data = diag_data / 2.0
    diag_data = torch.unsqueeze(diag_data, dim=len(data.shape) - 1)
    last_dims_t = len(data_dash.shape) - 1
    attention_dims_t = len(data_dash.shape) - 3
    if is_query:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data - torch.max(data_dash, dim=last_dims_t, keepdim=True)[
                    0]) + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data - torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                                                            dim=attention_dims_t, keepdim=True)[
                    0]) + numerical_stabilizer
        )
    return data_dash


def linear_kernel(x, node_vec1, node_vec2):
    # x: [B, N, 1, nhid] node_vec1: [B, N, 1, r], node_vec2: [B, N, 1, r]
    node_vec1 = node_vec1.permute(1, 0, 2, 3)  # [N, B, 1, r]
    node_vec2 = node_vec2.permute(1, 0, 2, 3)  # [N, B, 1, r]
    x = x.permute(1, 0, 2, 3)  # [N, B, 1, nhid]

    v2x = torch.einsum("nbhm,nbhd->bhmd", node_vec2, x)
    out1 = torch.einsum("nbhm,bhmd->nbhd", node_vec1, v2x)  # [N, B, 1, nhid]

    one_matrix = torch.ones([node_vec2.shape[0]]).to(node_vec1.device)
    node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
    out2 = torch.einsum("nbhm,bhm->nbh", node_vec1, node_vec2_sum)  # [N, 1]

    out1 = out1.permute(1, 0, 2, 3)  # [B, N, 1, nhid]
    out2 = out2.permute(1, 0, 2)
    out2 = torch.unsqueeze(out2, len(out2.shape))
    out = out1 / out2  # [B, N, 1, nhid]

    return out


def spatial_loss(node_vec1, node_vec2, supports, edge_indices):
    B = node_vec1.size(0)
    node_vec1 = node_vec1.permute(1, 0, 2, 3)  # [N, B, 1, r]
    node_vec2 = node_vec2.permute(1, 0, 2, 3)  # [N, B, 1, r]

    node_vec1_end, node_vec2_start = node_vec1[edge_indices[:, 0]], node_vec2[edge_indices[:, 1]]  # [E, B, 1, r]
    attn1 = torch.einsum("ebhm,ebhm->ebh", node_vec1_end, node_vec2_start)  # [E, B, 1]
    attn1 = attn1.permute(1, 0, 2)  # [B, E, 1]

    one_matrix = torch.ones([node_vec2.shape[0]]).to(node_vec1.device)
    node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
    attn_norm = torch.einsum("nbhm,bhm->nbh", node_vec1, node_vec2_sum)

    attn2 = attn_norm[edge_indices[:, 0]]  # [E, B, 1]
    attn2 = attn2.permute(1, 0, 2)  # [B, E, 1]
    attn_score = attn1 / attn2  # [B, E, 1]

    d_norm = supports[0][edge_indices[:, 0], edge_indices[:, 1]]
    d_norm = d_norm.reshape(1, -1, 1).repeat(B, 1, attn_score.shape[-1])
    spatial_loss = torch.mean(attn_score.log() * d_norm)

    return spatial_loss


class conv_approximation(nn.Module):
    def __init__(self, dropout, tau, random_feature_dim):
        super(conv_approximation, self).__init__()
        self.tau = tau
        self.random_feature_dim = random_feature_dim
        self.activation = nn.ReLU()
        self.dropout = dropout

    def forward(self, x, node_vec1, node_vec2):
        B = x.size(0)  # (B, N, 1, nhid)
        dim = node_vec1.shape[-1]  # (N, 1, d)

        random_seed = torch.ceil(torch.abs(torch.sum(node_vec1) * 1e8)).to(torch.int32)
        random_matrix = create_random_matrix(self.random_feature_dim, dim, seed=random_seed).to(
            node_vec1.device)  # (d, r)

        node_vec1 = node_vec1 / math.sqrt(self.tau)
        node_vec2 = node_vec2 / math.sqrt(self.tau)
        node_vec1_prime = random_feature_map(node_vec1, True, random_matrix)  # [B, N, 1, r]
        node_vec2_prime = random_feature_map(node_vec2, False, random_matrix)  # [B, N, 1, r]

        x = linear_kernel(x, node_vec1_prime, node_vec2_prime)

        return x, node_vec1_prime, node_vec2_prime


class linearized_conv(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout, tau=1.0, random_feature_dim=64):
        super(linearized_conv, self).__init__()

        self.dropout = dropout
        self.tau = tau
        self.random_feature_dim = random_feature_dim

        self.input_fc = nn.Conv2d(in_channels=in_dim, out_channels=hid_dim, kernel_size=(1, 1), bias=True)
        self.output_fc = nn.Conv2d(in_channels=in_dim, out_channels=hid_dim, kernel_size=(1, 1), bias=True)
        self.activation = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(p=dropout)

        self.conv_app_layer = conv_approximation(self.dropout, self.tau, self.random_feature_dim)

    def forward(self, input_data, node_vec1, node_vec2):
        x = self.input_fc(input_data)
        x = self.activation(x) * self.output_fc(input_data)
        x = self.dropout_layer(x)

        x = x.permute(0, 2, 3, 1)  # (B, N, 1, dim*4)
        x, node_vec1_prime, node_vec2_prime = self.conv_app_layer(x, node_vec1, node_vec2)
        x = x.permute(0, 3, 1, 2)  # (B, dim*4, N, 1)

        return x, node_vec1_prime, node_vec2_prime
