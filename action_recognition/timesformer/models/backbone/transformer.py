from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return F.gelu(input) 

class SelfAttention(nn.Module):
    def __init__(self, config, N_head=6, D=128):
        super().__init__()
        assert D % N_head == 0
        self.config = config
        

        self.N_head = N_head
        
        self.key = nn.Linear(D, D)
        self.query = nn.Linear(D, D)
        self.value = nn.Linear(D, D)
        
        self.attn_drop = nn.Dropout(0.1)
        self.resd_drop = nn.Dropout(0.1)
        
        self.proj = nn.Linear(D, D)
    def forward(self, x, mask=None, query=None):
        # x: B * N * D
        B, N, D = x.size()
        if query is not None:
            q = self.query(query.view(B*N, -1)).view(B, N, self.N_head, D//self.N_head).transpose(1, 2)
        else:
            q = self.query(x.view(B*N, -1)).view(B, N, self.N_head, D//self.N_head).transpose(1, 2)
        k = self.key(x.view(B*N, -1)).view(B, N, self.N_head, D//self.N_head).transpose(1, 2)
        v = self.value(x.view(B*N, -1)).view(B, N, self.N_head, D//self.N_head).transpose(1, 2)
        
        A = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            A = A.masked_fill(mask[:,:,:N,:N] == 0, float('-inf'))
        A = F.softmax(A, dim=-1)
        A_drop = self.attn_drop(A)
        y = (A_drop @ v).transpose(1, 2).contiguous().view(B, N, D)
        y = self.resd_drop(self.proj(y))
        return y, A
    
class SABlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        D = config.D
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.attn = SelfAttention(config, N_head=config.N_head, D=D)
        self.mlp = nn.Sequential(
                nn.Linear(D, 4*D),
                GELU(),
                nn.Linear(4*D, D),
                nn.Dropout(0.1)
            )

        
    def forward(self, x, mask=None, query=None):
        y, att = self.attn(self.ln1(x), mask=mask, query=query)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, att
    

class Transformer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        D = config.D
        self.blocks = nn.ModuleList([SABlock(config) for _ in range(config.n_layer)])
        
        # self.ln_head = nn.LayerNorm(config.D)
        # self.head = nn.Linear(config.D, config.D)
        
        self.apply(self._init_weights)
        
        # print("number of parameters: %d" % sum(p.numel() for p in self.parameters()))
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def forward(self, x, query=None):
        
        for i, blk in enumerate(self.blocks):
            if (i == 0) and (query is not None):
                x, _ = blk(x, query=query)
            else:
                x, _ = blk(x)
        
        # y = self.head(self.ln_head(x))
        return x