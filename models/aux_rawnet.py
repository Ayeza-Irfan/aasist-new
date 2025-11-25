

"""Improved Auxiliary RawNet-style encoder for AASIST integration.
Features added:
 - SincConv front-end (lightweight implementation)
 - Residual 1D blocks with SE (Squeeze-and-Excitation) modules
 - Attentive statistics pooling (attention-based mean/std)
 - Default embedding dim = 256 (configurable)
 - Designed to accept waveform tensors shaped (B, N) or (B, 1, N)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv1D(nn.Module):
    """Lightweight Sinc-based convolution. Not a full optimized implementation,
    but suitable as a learnable band-pass front end for raw audio.
    Input: (B, 1, T)
    Output: (B, out_channels, T_out)
    """

    def __init__(self, out_channels, kernel_size=251, sample_rate=16000, in_channels=1, min_low_hz=50, min_band_hz=50):
        super().__init__()
        if in_channels != 1:
            raise ValueError("SincConv1D only supports one input channel")
        self.out_channels = out_channels
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # initialize filterbanks in mel scale
        low_hz = 30.0
        high_hz = self.sample_rate / 2 - (min_low_hz + min_band_hz)

        # initialize cut-off frequencies (Hz) linearly on Mel scale
        mel = lambda hz: 2595.0 * math.log10(1.0 + hz / 700.0)
        inv_mel = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        low_mel = mel(low_hz)
        high_mel = mel(high_hz)
        mels = torch.linspace(low_mel, high_mel, out_channels + 1)
        hz = inv_mel(mels)
        # learnable parameters: low and band (as positive values via softplus)
        low = nn.Parameter(hz[:-1])
        high = nn.Parameter(hz[1:])
        self.register_parameter('low_hz_', low)
        self.register_parameter('high_hz_', high)

        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, int((self.kernel_size / 2)))
        self.register_buffer('n', n_lin)

        # Hamming window
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(0, self.kernel_size).float() / (self.kernel_size - 1))
        self.register_buffer('window', window)

    def forward(self, x):
        # x: (B, 1, T)
        device = x.device
        low = torch.abs(self.low_hz_) + 1.0  # ensure positive
        high = torch.abs(self.high_hz_) + low + 1.0  # ensure band > 0
        low = low.to(device)
        high = high.to(device)

        # create filters
        n = self.n.to(device)
        f_times_t_low = torch.matmul(low.unsqueeze(1), (2 * math.pi * n / self.sample_rate).unsqueeze(0))
        f_times_t_high = torch.matmul(high.unsqueeze(1), (2 * math.pi * n / self.sample_rate).unsqueeze(0))

        # ideal band-pass in time domain (sinc difference)
        band_pass = (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (n.unsqueeze(0).to(device) + 1e-8)
        band_pass = band_pass * self.window[:band_pass.shape[1]].to(device)
        # symmetric filters (odd length)
        filters = torch.zeros((self.out_channels, self.kernel_size), device=device)
        mid = (self.kernel_size - 1) // 2
        # fill left part (reversed)
        left = torch.flip(band_pass, dims=[1])
        filters[:, :mid] = left
        filters[:, mid] = 2 * (high - low) / self.sample_rate
        filters[:, mid+1:mid+1+band_pass.shape[1]] = band_pass

        filters = filters.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=1, padding=mid)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for 1D feature maps"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction)),
            nn.SELU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T)
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class _ResBlock1D_SE(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.SELU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        if in_ch != out_ch:
            self.down = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        else:
            self.down = None
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)
        self.se = SEBlock(out_ch, reduction=8)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down is not None:
            identity = self.down(identity)
        out = out + identity
        out = self.se(out)
        out = self.pool(out)
        return out


class AttentiveStatsPooling(nn.Module):
    """Attention-based pooling over time dimension. Returns concatenated mean+std."""
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, x):
        # x: (B, C, T)
        w = self.att(x)  # (B, 1, T)
        w = torch.softmax(w, dim=2)
        mean = torch.sum(x * w, dim=2)  # (B, C)
        # compute std with attention-weighted mean
        mean_exp = mean.unsqueeze(2)
        var = torch.sum(((x - mean_exp) ** 2) * w, dim=2)
        std = torch.sqrt(var + 1e-8)
        return torch.cat([mean, std], dim=1)  # (B, C*2)


class AuxiliaryRawNet(nn.Module):
    def __init__(self, embedding_dim: int = 256, in_channels: int = 1, sample_rate: int = 16000):
        """

        Args:
            embedding_dim: final embedding dimension (default 256)
            in_channels: 1 (mono)
            sample_rate: audio sampling rate
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sample_rate = sample_rate

        # SincConv front-end (learnable band-pass filters)
        self.sinc = SincConv1D(out_channels=32, kernel_size=251, sample_rate=sample_rate, in_channels=in_channels)

        # small conv stack after SincConv
        self.front = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.SELU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SELU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )

        # residual blocks with SE
        self.res1 = _ResBlock1D_SE(64, 128)
        self.res2 = _ResBlock1D_SE(128, 128)
        self.res3 = _ResBlock1D_SE(128, 256)

        # attentive stats pooling
        self.pool = AttentiveStatsPooling(in_dim=256, hidden_dim=128)

        # projection to embedding
        self.fc = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.SELU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        """x can be (B, N) or (B, 1, N)"""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # ensure float32
        x = x.float()

        out = self.sinc(x)          # (B, 32, T)
        out = self.front(out)      # (B, 64, T1)
        out = self.res1(out)       # (B, 128, T2)
        out = self.res2(out)       # (B, 128, T3)
        out = self.res3(out)       # (B, 256, T4)

        pooled = self.pool(out)    # (B, 512) -> mean+std concat where C=256

        emb = self.fc(pooled)      # (B, embedding_dim)
        return emb
