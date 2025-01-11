import torch

from torch import nn
from pdb import set_trace

class WeightPool(nn.Module):
    def __init__(self, input_dim, d_model, size, num_patches, method='attn', bias=True):
        super().__init__()
        self.size = size
        self.d_model = d_model
        self.method = method

        self.w1 = nn.Parameter(torch.randn(input_dim, self.size), requires_grad=True)
        self.w2 = nn.Parameter(torch.randn(self.size, self.d_model, input_dim), requires_grad=True)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.randn(num_patches, self.d_model), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias)

    def get_weight(self, x):
        if self.method == 'attn':
            sim = torch.softmax(torch.matmul(x, self.w1), dim=-1)  # [B, C, S, T] @ [T, k]  BCSTk
        elif self.method == 'cos':
            q_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
            v_norm = torch.norm(self.w1, p=2, dim=-1, keepdim=True)
            sim = torch.matmul(x, self.w1) / (q_norm * v_norm)
        else:
            raise ValueError('similarity method error')
        if len(sim.shape) == 4:
            weight = torch.einsum('bcsk, kdt->bsdt', sim, self.w2)  # BCSKdT
        else:
            weight = torch.einsum('bsk, kdt->bsdt', sim, self.w2)  # BCSKdT
        return weight

    def forward(self, x):
        return self.get_weight(x), self.bias

