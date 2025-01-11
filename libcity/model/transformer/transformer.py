import torch
import torch.nn as nn
import math
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model.loss import masked_mae_torch
from pdb import set_trace


# 自定义Transformer层，对N维度做自注意力
class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # 定义前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        """
        src: 输入张量，形状为 [B, T, N, D]，这里我们要对N维度做自注意力
        """
        src = src.permute(0, 2, 1, 3).contiguous()
        B, N, T, D = src.shape
        src = src.view(src.size(0), src.size(1), -1)
        # 自注意力机制
        src2, attn_weights = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # 前馈网络
        src2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # 恢复原始形状 [B, N, T, D]
        src = src.view(B, N, T, D)
        # 再调整回 [B, T, N, D]
        src = src.permute(0, 2, 1, 3).contiguous()
        return src, attn_weights


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x


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
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()


class DataEmbedding(nn.Module):
    def __init__(
            self, feature_dim, embed_dim, minute_size, drop=0.,
            add_time_in_day=False, add_day_in_week=False
    ):
        super().__init__()

        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week

        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)

        if self.add_time_in_day:
            self.minute_size = 1440
            self.daytime_embedding = nn.Embedding(self.minute_size, embed_dim)
        if self.add_day_in_week:
            weekday_size = 7
            self.weekday_embedding = nn.Embedding(weekday_size, embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        origin_x = x  # [B, T, N, C]
        x = self.value_embedding(origin_x[:, :, :, :self.feature_dim])
        x += self.position_encoding(x)
        if self.add_time_in_day:
            x += self.daytime_embedding((origin_x[:, :, :, self.feature_dim] * self.minute_size).round().long())
        if self.add_day_in_week:
            x += self.weekday_embedding(origin_x[:, :, :, self.feature_dim + 1].long())
        x = self.dropout(x)
        return x



class TransformerModel(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super(TransformerModel, self).__init__(config, data_feature)
        self.input_dim = data_feature.get("feature_dim")
        self._scaler = data_feature.get("scaler")
        self.output_dim = data_feature.get("output_dim", 1)
        self.ext_dim = data_feature.get("ext_dim", 1)
        self.d_model = config.get("d_model", 128)
        self.nhead = config.get("nhead", 8)
        self.dim_feedforward = config.get("dim_feedforward", 512)
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get("output_window", 12)
        self.dropout = config.get("dropout", 0.1)
        self.num_layers = config.get("num_layers", 6)
        self.encoder_embedding = DataEmbedding(self.input_dim - self.ext_dim, self.d_model, 1, self.dropout, True, True)
        self.decoder_embedding = nn.Linear(self.d_model, self.d_model)

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(self.input_window * self.d_model, self.nhead, self.dim_feedforward, self.dropout) for _ in range(self.num_layers)
        ])

        self.transformer_layers_decoder = nn.ModuleList([
            TransformerLayer(self.input_window * self.d_model, self.nhead, self.dim_feedforward, self.dropout) for _ in range(self.num_layers)
        ])

        self.regression_layer = nn.Linear(self.d_model, self.output_dim)

    def forward(self, batch):
        x = batch["X"]
        # 假设输入形状为 [B, T, N, D]
        batch_size, seq_len, num_nodes, feat_dim = x.size()
        encoder_output = self.encoder_embedding(x)

        # Encoder部分

        for layer in self.transformer_layers:
            encoder_output, attn_weights = layer(encoder_output)

        # Decoder部分（这里简单示例，可根据实际需求改进）
        decoder_output = self.decoder_embedding(encoder_output)
        for layer in self.transformer_layers_decoder:
            decoder_output, _ = layer(decoder_output)

        # 恢复形状
        decoder_output = decoder_output.view(batch_size, seq_len, num_nodes, self.d_model)

        # 用于回归的输出
        regression_output = self.regression_layer(decoder_output)

        return attn_weights, regression_output

    def calculate_loss(self, batch):
        y_true = batch['y']
        attn_weights, y_predicted = self.forward(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        loss = masked_mae_torch(y_true, y_predicted, 0.0)
        return loss

    def predict(self, batch):
        attn_weights, regression_output = self.forward(batch)
        return regression_output