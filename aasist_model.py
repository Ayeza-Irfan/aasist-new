"""
AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license

MODIFIED: Model.forward now returns last_hidden (the embedding) and output, 
making it usable as a feature extractor in the hybrid model.
"""

import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# --- INTERNAL AASIST COMPONENTS ---

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)
        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)
        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)
        # activate
        self.act = nn.SELU(inplace=True)
        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # attention map
        att_map = torch.matmul(torch.tanh(self.att_proj(x)), self.att_weight)
        att_map = F.softmax(att_map / self.temp, dim=1)  # softmax over node axis

        # project
        x_with_att = self.proj_with_att(x)
        x_without_att = self.proj_without_att(x)

        # aggregation
        x_with_att = torch.sum(x_with_att * att_map, dim=1)  # aggregate by attention

        # output
        x_out = x_with_att + x_without_att[:, 0, :]  # residual with the first node
        x_out = self.bn(x_out)
        return self.act(x_out)


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map for T, S, and M
        self.att_proj_T = nn.Linear(in_dim, out_dim)
        self.att_weight_T = self._init_new_params(out_dim, 1)
        self.att_proj_S = nn.Linear(in_dim, out_dim)
        self.att_weight_S = self._init_new_params(out_dim, 1)
        self.att_proj_M = nn.Linear(in_dim, out_dim)
        self.att_weight_M = self._init_new_params(out_dim, 1)

        # project
        self.proj = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn_T = nn.BatchNorm1d(out_dim)
        self.bn_S = nn.BatchNorm1d(out_dim)
        self.bn_M = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x_T, x_S, master):
        '''
        x_T, x_S   :(#bs, #node, #dim)
        master     :(#bs, 1, #dim)
        '''
        # apply input dropout
        x_T = self.input_drop(x_T)
        x_S = self.input_drop(x_S)
        master = self.input_drop(master)

        # --- T-node update ---
        # 1. T-node self attention
        att_map_T = torch.matmul(torch.tanh(self.att_proj_T(x_T)), self.att_weight_T)
        att_map_T = F.softmax(att_map_T / self.temp, dim=1)
        x_T_att = torch.sum(self.proj(x_T) * att_map_T, dim=1)  # aggregate by attention

        # 2. S-node to T-node cross attention
        att_map_ST = torch.matmul(torch.tanh(self.att_proj_S(x_S)), self.att_weight_S)
        att_map_ST = F.softmax(att_map_ST / self.temp, dim=1)
        x_ST_att = torch.sum(self.proj(x_S) * att_map_ST, dim=1)

        # 3. Master-node to T-node cross attention
        att_map_MT = torch.matmul(torch.tanh(self.att_proj_M(master)), self.att_weight_M)
        att_map_MT = F.softmax(att_map_MT / self.temp, dim=1)
        x_MT_att = torch.sum(self.proj(master) * att_map_MT, dim=1)

        # 4. output
        out_T = x_T_att + x_ST_att + x_MT_att
        out_T = self.bn_T(out_T)
        out_T = self.act(out_T)

        # --- S-node update (Symmetric to T-node update) ---
        # 1. S-node self attention
        att_map_S = torch.matmul(torch.tanh(self.att_proj_S(x_S)), self.att_weight_S)
        att_map_S = F.softmax(att_map_S / self.temp, dim=1)
        x_S_att = torch.sum(self.proj(x_S) * att_map_S, dim=1)

        # 2. T-node to S-node cross attention
        att_map_TS = torch.matmul(torch.tanh(self.att_proj_T(x_T)), self.att_weight_T)
        att_map_TS = F.softmax(att_map_TS / self.temp, dim=1)
        x_TS_att = torch.sum(self.proj(x_T) * att_map_TS, dim=1)

        # 3. Master-node to S-node cross attention
        x_MS_att = torch.sum(self.proj(master) * att_map_MT, dim=1)  # Use same M-proj

        # 4. output
        out_S = x_S_att + x_TS_att + x_MS_att
        out_S = self.bn_S(out_S)
        out_S = self.act(out_S)

        # --- Master-node update ---
        # 1. T-node to M-node cross attention
        att_map_TM = torch.matmul(torch.tanh(self.att_proj_T(x_T)), self.att_weight_T)
        att_map_TM = F.softmax(att_map_TM / self.temp, dim=1)
        x_TM_att = torch.sum(self.proj(x_T) * att_map_TM, dim=1)

        # 2. S-node to M-node cross attention
        att_map_SM = torch.matmul(torch.tanh(self.att_proj_S(x_S)), self.att_weight_S)
        att_map_SM = F.softmax(att_map_SM / self.temp, dim=1)
        x_SM_att = torch.sum(self.proj(x_S) * att_map_SM, dim=1)

        # 3. M-node self attention
        x_M_att = torch.sum(self.proj(master) * att_map_MT, dim=1)

        # 4. output
        out_M = x_TM_att + x_SM_att + x_M_att
        out_M = self.bn_M(out_M)
        out_M = self.act(out_M)

        return out_T.unsqueeze(1), out_S.unsqueeze(1), out_M.unsqueeze(1)


class GraphPool(nn.Module):
    def __init__(self, pool_ratio):
        super().__init__()
        self.pool_ratio = pool_ratio

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        input_len = x.shape[1]
        output_len = int(input_len / self.pool_ratio)
        x = x.transpose(1, 2).contiguous()
        x = F.adaptive_max_pool1d(x, output_size=output_len)
        return x.transpose(1, 2).contiguous()


class CONV(nn.Module):
    def __init__(self,
                 out_channels,
                 kernel_size=5,
                 in_channels=1,
                 stride=1,
                 padding=2):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.selu = nn.SELU(inplace=True)

    def forward(self, x):
        return self.selu(self.bn(self.conv(x)))


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(3, 3),
                               padding=(1, 1),
                               stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(3, 3),
                               padding=(1, 1),
                               stride=1)
        
        if nb_filts[0] == nb_filts[1]:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=nb_filts[0],
                          out_channels=nb_filts[1],
                          padding=(0, 0),
                          kernel_size=(1, 1),
                          stride=1),
                nn.BatchNorm2d(num_features=nb_filts[1]))
        
        self.selu = nn.SELU(inplace=True)
        self.pool = nn.MaxPool2d((2, 2))


    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.pool(out)
        return out


# --- AASIST Model Class (MODIFIED FOR HYBRID FUSION) ---

class Model(nn.Module):
    """
    AASIST Model
    
    NOTE: The forward method is modified to return (last_hidden, output) 
    in all cases, making the feature embedding accessible for the hybrid model.
    """
    def __init__(self, d_args):
        super().__init__()

        self.d_args = d_args
        filts = d_args["filts"]
        gat_dims = d_args["gat_dims"]
        pool_ratios = d_args["pool_ratios"]
        temperatures = d_args["temperatures"]

        self.conv_time = CONV(out_channels=filts[0],
                              kernel_size=d_args["first_conv"],
                              in_channels=1)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts[4])),
            nn.Sequential(Residual_block(nb_filts[4])),
            nn.Sequential(Residual_block(nb_filts[4])))

        self.master = self._init_new_params(1, 1, filts[-1])
        self.master1 = self._init_new_params(1, 1, filts[-1])
        self.master2 = self._init_new_params(1, 1, filts[-1])

        self.GAT_layer_T = GraphAttentionLayer(
            filts[-1], gat_dims[0], temperature=temperatures[0])
        self.GAT_layer_S = GraphAttentionLayer(
            filts[-1], gat_dims[0], temperature=temperatures[0])

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[1])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[1])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[1])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[1])

        self.pool_hS1 = GraphPool(pool_ratios[0])
        self.pool_hT1 = GraphPool(pool_ratios[1])
        self.pool_hS2 = GraphPool(pool_ratios[0])
        self.pool_hT2 = GraphPool(pool_ratios[1])

        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x, Freq_aug=False):
        if Freq_aug:
            if self.training and random.random() < 0.2:
                # Assuming x is Spectrogram-like (Time, Freq), rolling on Freq dimension (2)
                for i in range(x.shape[0]):
                    if x.dim() >= 3:
                        x[i] = x[i].roll(random.randrange(-32, 32), 2)
                    # Note: AASIST is typically Spectrogram-based, but is adapted here 
                    # to accept raw audio/spectrogram input depending on the DataLoader output.
                    # Since this is the AASIST stream, x here represents the features 
                    # derived from the raw audio for AASIST (e.g., Spectrogram).

        x = x.unsqueeze(1)
        x = self.first_bn(x)
        x = self.conv_time(x)

        for i in range(len(self.encoder)):
            x = self.encoder[i](x)

        x = x.flatten(1)
        x = x.unsqueeze(1)
        x = self.selu(x)

        # reshape for graph
        x_S = x  # x: (#bs, #node, #dim)
        x_T = x.transpose(1, 2)  # x: (#bs, #dim, #node)

        # GAT for initial embedding
        out_T = self.GAT_layer_T(x_T)
        out_S = self.GAT_layer_S(x_S)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = out_T1 + out_T2
        out_S = out_S1 + out_S2
        master = master1 + master2

        # final pooling
        out_T, _ = torch.max(out_T, dim=1)
        out_S, _ = torch.max(out_S, dim=1)

        # concat
        out_T = out_T.squeeze(1)
        out_S = out_S.squeeze(1)
        master = master.squeeze(1)

        # final hidden layer
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        # MODIFICATION: Always return the embedding (last_hidden) and the output
        return last_hidden, output