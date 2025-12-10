
# AASIST.py — SiLU everywhere + GELU in Dense Layers

import random
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# --------------------------------------------------------------
# GraphAttentionLayer (SELU → SiLU)
# --------------------------------------------------------------
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)

        # Swish
        self.act = nn.SiLU()
        self.temp = kwargs.get("temperature", 1.)

    def forward(self, x):
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        return self.act(x)

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map(self, x):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        return F.softmax(att_map, dim=-2)

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_BN(self, x):
        org = x.size()
        x = x.view(-1, org[-1])
        x = self.bn(x)
        return x.view(org)

    def _init_new_params(self, *size):
        p = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(p)
        return p


# --------------------------------------------------------------
# HtrgGraphAttentionLayer (SELU → SiLU)
# --------------------------------------------------------------
class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)

        # Swish
        self.act = nn.SiLU()
        self.temp = kwargs.get("temperature", 1.)

    def forward(self, x1, x2, master=None):
        n1 = x1.size(1)
        n2 = x2.size(1)

        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        x = self.input_drop(x)
        att_map = self._derive_att_map(x, n1, n2)
        master = self._update_master(x, master)

        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)

        return x[:, :n1], x[:, n1:], master

    def _update_master(self, x, master):
        att_map = self._derive_att_map_master(x, master)
        return self._project_master(x, master, att_map)

    def _pairwise_mul_nodes(self, x):
        nb = x.size(1)
        expanded = x.unsqueeze(2).expand(-1, -1, nb, -1)
        mirror = expanded.transpose(1, 2)
        return expanded * mirror

    def _derive_att_map_master(self, x, master):
        att = x * master
        att = torch.tanh(self.att_projM(att))
        att = torch.matmul(att, self.att_weightM)
        att = att / self.temp
        return F.softmax(att, dim=-2)

    def _derive_att_map(self, x, n1, n2):
        att = self._pairwise_mul_nodes(x)
        att = torch.tanh(self.att_proj(att))

        board = torch.zeros_like(att[:, :, :, 0]).unsqueeze(-1)
        board[:, :n1, :n1] = torch.matmul(att[:, :n1, :n1], self.att_weight11)
        board[:, n1:, n1:] = torch.matmul(att[:, n1:, n1:], self.att_weight22)
        board[:, :n1, n1:] = torch.matmul(att[:, :n1, n1:], self.att_weight12)
        board[:, n1:, :n1] = torch.matmul(att[:, n1:, :n1], self.att_weight12)

        board = board / self.temp
        return F.softmax(board, dim=-2)

    def _project(self, x, att):
        x1 = self.proj_with_att(torch.matmul(att.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(self, x, master, att):
        x1 = self.proj_with_attM(torch.matmul(att.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x):
        org = x.size()
        x = x.view(-1, org[-1])
        x = self.bn(x)
        return x.view(org)

    def _init_new_params(self, *size):
        p = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(p)
        return p


# Residual Blocks + CONV + Encoder (SiLU everywhere)

# --------------------------------------------------------------
# CONV Time Domain Stem (unchanged except SiLU if used)
# --------------------------------------------------------------
class CONV(nn.Module):
    def __init__(self, out_channels, kernel_size):
        super().__init__()
        
        ks = kernel_size if isinstance(kernel_size, tuple) else (1, kernel_size)
        self.conv = nn.Conv2d(1, out_channels, kernel_size=ks)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


# --------------------------------------------------------------
# Residual Block (SELU → SiLU)
# --------------------------------------------------------------
class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(nb_filts[0])
        self.conv1 = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(2, 3), padding=(1, 1))

        self.act = nn.SiLU()

        self.bn2 = nn.BatchNorm2d(nb_filts[1])
        self.conv2 = nn.Conv2d(nb_filts[1], nb_filts[1], kernel_size=(2, 3), padding=(0, 1))

        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(1, 3), padding=(0, 1))

        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.act(out)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out = out + identity
        return self.mp(out)


# --------------------------------------------------------------
# Encoder (6 Residual Blocks)
# --------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, filts):
        super().__init__()
        self.layer1 = Residual_block(filts[1], first=True)
        self.layer2 = Residual_block(filts[2])
        self.layer3 = Residual_block(filts[3])
        self.layer4 = Residual_block(filts[4])
        self.layer5 = Residual_block(filts[4])
        self.layer6 = Residual_block(filts[4])

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x



# ASP + GAT Layers + Fusion + Pre-MLP setup (SiLU everywhere)

# --------------------------------------------------------------
# Attentive Statistics Pooling (from your file, unchanged except SiLU)
# --------------------------------------------------------------
class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, bottleneck_dim)
        self.linear2 = nn.Linear(bottleneck_dim, in_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.eps = 1e-9
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (batch, time, feat)
        h = self.act(self.linear1(x))
        w = self.softmax(self.linear2(h))

        mean = torch.sum(w * x, dim=1)
        var = torch.sum(w * (x - mean.unsqueeze(1)) ** 2, dim=1)

        std = torch.sqrt(var + self.eps)
        return torch.cat([mean, std], dim=1)


# --------------------------------------------------------------
# Model (Main AASIST Fusion + GAT + ASP + Dense Layers (GELU))
# --------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, d_args):
        super().__init__()
        self.d_args = d_args

        filts = d_args["filts"]
        gat_dims = d_args["gat_dims"]
        pool_ratios = d_args["pool_ratios"]
        temperatures = d_args["temperatures"]

        # --------------------------------------------------
        # Convolutional frontend
        # --------------------------------------------------
        self.conv_time = CONV(out_channels=filts[0], kernel_size=d_args["first_conv"])
        self.first_bn = nn.BatchNorm2d(1)
        self.act = nn.SiLU()
        self.encoder = Encoder(filts)

        # --------------------------------------------------
        # Positional parameters for the GAT blocks
        # --------------------------------------------------
        self.pos_S = nn.Parameter(torch.randn(1, 23, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        # --------------------------------------------------
        # Graph Attention Layers
        # --------------------------------------------------
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[1])

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[3])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[2], temperature=temperatures[4])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[2], temperature=temperatures[5])

        # --------------------------------------------------
        # Attentive Statistics Pooling
        # --------------------------------------------------
        fused_feat_dim = 2 * gat_dims[-1]
        self.asp = AttentiveStatsPool(fused_feat_dim, bottleneck_dim=128)

        # ASP output dimension = mean + std → 2 * fused_gat_dim
        asp_out_dim = 2 * fused_feat_dim

        # --------------------------------------------------
        # Dense Layers (Your 5-layer MLP) — NOW USING GELU
        # --------------------------------------------------
        self.fc1 = nn.Linear(asp_out_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)

        self.gelu = nn.GELU()  # GELU ONLY HERE
        self.sigmoid = nn.Sigmoid()





    def forward(self, x, Freq_aug=False):

        if x.dim() == 2:
            # (B, L) -> (B, 1, 1, L)
            x = x.unsqueeze(1).unsqueeze(2)
        elif x.dim() == 3:
            # (B, T, F) -> (B, 1, T, F)
            x = x.unsqueeze(1)
        # --------------------------------------------------
        # FRONTEND
        # --------------------------------------------------
        x = self.first_bn(x)
        x = self.act(x)
        x = self.conv_time(x)
        x = self.encoder(x)

        B, C, T, F = x.size()

        # --------------------------------------------------
        # PREP FOR GAT (S and T graphs)
        # --------------------------------------------------
        # Mean over frequency → Temporal graph
        x_T = torch.mean(x, dim=3)  # (B, C, T)
        x_T = x_T.transpose(1, 2)    # (B, T, C)

        # Mean over time → Spectral graph
        x_S = torch.mean(x, dim=2)  # (B, C, F)
        x_S = x_S.transpose(1, 2)
        x_S = x_S + self.pos_S      # (B, 23, C)

        # --------------------------------------------------
        # FIRST GAT LAYERS
        # --------------------------------------------------
        x_S = self.GAT_layer_S(x_S)
        x_T = self.GAT_layer_T(x_T)

        # --------------------------------------------------
        # HETEROGENEOUS GAT
        # --------------------------------------------------
        x_S, x_T, master1 = self.HtrgGAT_layer_ST11(x_S, x_T, self.master1)
        x_S, x_T, master1 = self.HtrgGAT_layer_ST12(x_S, x_T, master1)
        x_S, x_T, master2 = self.HtrgGAT_layer_ST21(x_S, x_T, self.master2)
        x_S, x_T, master2 = self.HtrgGAT_layer_ST22(x_S, x_T, master2)

        # --------------------------------------------------
        # FUSION
        # --------------------------------------------------
        fused = torch.cat([
            torch.mean(x_S, dim=1),  # (B, dim)
            torch.mean(x_T, dim=1),
        ], dim=1)

        # Fuse again into ASP input dimension
        # fused_gat_dim = gat_dims[-1]
        # → reshape for ASP
        fused = fused.unsqueeze(1)  # (B, 1, feat)

        # --------------------------------------------------
        # ASP POOLING
        # --------------------------------------------------
        pooled = self.asp(fused)  # (B, 2 * feat)

        # --------------------------------------------------
        # DENSE NETWORK (5 layers with GELU)
        # --------------------------------------------------
        y = self.gelu(self.fc1(pooled))
        y = self.gelu(self.fc2(y))
        y = self.gelu(self.fc3(y))
        y = self.gelu(self.fc4(y))
        y = self.gelu(self.fc5(y))

       
        logits = self.fc6(y)
        probs = self.sigmoid(logits)
        return logits, probs


