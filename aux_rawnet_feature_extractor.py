import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# --- E_M: Main Encoder (Handcrafted Features: MFCCs) ---
class MFCCEncoder(nn.Module):
    """
    Encoder for Handcrafted Features (MFCCs). 
    Processes MFCCs using a simple 1D-CNN stack.
    """
    def __init__(self, mfcc_dim=40, final_dim=320):
        super().__init__()
        # Simple TDNN/CNN-like stack for MFCCs
        self.conv_stack = nn.Sequential(
            nn.Conv1d(mfcc_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.SELU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.SELU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, final_dim, kernel_size=1)
        )
        self.final_dim = final_dim
    
    def forward(self, x: Tensor) -> Tensor:
        # x is assumed to be (#bs, #time, #mfcc_dim). 
        # Convert to (#bs, #mfcc_dim, #time) for Conv1D.
        x = x.transpose(1, 2)
        x = self.conv_stack(x)
        # Global Max Pooling across time dimension to get fixed-size embedding
        x = torch.max(x, dim=2)[0] 
        return x # E_M: (#bs, final_dim)

# --- E_A: Auxiliary Encoder (Raw Audio) ---
class LightweightRawNetEncoder(nn.Module):
    """
    Lightweight Encoder for Raw Audio Input. 
    Uses a simplified RawNet-like 1D-CNN structure.
    """
    def __init__(self, final_dim=320):
        super().__init__()
        self.conv1d_stack = nn.Sequential(
            # First layer: Large kernel for initial feature extraction
            nn.Conv1d(1, 32, kernel_size=51, stride=2, padding=25),
            nn.BatchNorm1d(32),
            nn.SELU(inplace=True),
            # Subsequent layers
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.SELU(inplace=True),
            nn.Conv1d(64, final_dim, kernel_size=3, padding=1)
        )
        self.final_dim = final_dim

    def forward(self, x: Tensor) -> Tensor:
        # x is assumed to be (#bs, #samples). Convert to (#bs, 1, #samples).
        x = x.unsqueeze(1)
        x = self.conv1d_stack(x)
        # Global Max Pooling across time dimension
        x = torch.max(x, dim=2)[0]
        return x # E_A: (#bs, final_dim)

# --- ARNet Feature Extractor (E_M + E_A -> E_C) ---
class AuxRawNetFeatureExtractor(nn.Module):
    """
    Auxiliary RawNet (ARNet) Feature Extractor (E_M + E_A -> E_C).
    Fuses the handcrafted (MFCC) and raw audio features.
    """
    def __init__(self, d_args):
        super().__init__()
        
        config = d_args["ARNet_config"]
        E_M_dim = config["E_M_dim"]
        E_A_dim = config["E_A_dim"]
        E_ARNet_dim = config["E_ARNet_dim"]
        mfcc_dim = config["mfcc_dim"]

        self.E_M = MFCCEncoder(mfcc_dim=mfcc_dim, final_dim=E_M_dim)
        self.E_A = LightweightRawNetEncoder(final_dim=E_A_dim)
        
        # Concatenate Encoder (E_C) to merge and process the combined features
        self.E_C = nn.Sequential(
            nn.Linear(E_M_dim + E_A_dim, E_ARNet_dim),
            nn.BatchNorm1d(E_ARNet_dim),
            nn.SELU(inplace=True),
            nn.Linear(E_ARNet_dim, E_ARNet_dim) # Final output layer
        )
        self.E_ARNet_dim = E_ARNet_dim

    def forward(self, handcrafted_features: Tensor, raw_audio: Tensor) -> Tensor:
        Fm = self.E_M(handcrafted_features)  # E_M output
        Fa = self.E_A(raw_audio)            # E_A output

        E_Combined = torch.cat((Fm, Fa), dim=1) # Concatenate: (#bs, E_M_dim + E_A_dim)
        E_ARNet = self.E_C(E_Combined)          # E_C output
        return E_ARNet # E_ARNet: (#bs, E_ARNet_dim)