import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# Import the feature extractors from the local files
import aasist_model  
from aux_rawnet_feature_extractor import AuxRawNetFeatureExtractor

class Model(nn.Module):
    """
    AASIST + Auxiliary RawNet (ARNet) Hybrid Fusion Model.
    
    This model runs two parallel streams, concatenates their feature embeddings, 
    and classifies the fused result.
    """
    def __init__(self, d_args):
        super().__init__()
        
        # 1. AASIST Feature Extractor (Raw Audio/Spectrogram Input)
        # We reuse the AASIST model but only for its feature embedding output.
        self.aasist_extractor = aasist_model.Model(d_args)
        
        # Determine sizes from AASIST configuration (5 * gat_dims[1] = 5*128 = 640)
        gat_dim1 = d_args["gat_dims"][1] 
        AASIST_EMBED_DIM = 5 * gat_dim1 
        
        # We remove the final classification layer from the AASIST extractor
        # to ensure we only get the feature embedding.
        del self.aasist_extractor.out_layer
        
        # 2. Auxiliary RawNet Feature Extractor (Dual Input: MFCCs + Raw Audio)
        self.arnet_extractor = AuxRawNetFeatureExtractor(d_args)
        
        # Get ARNet output dimension from its config (e.g., 640)
        ARNET_EMBED_DIM = d_args["ARNet_config"]["E_ARNet_dim"]
        
        # Total combined embedding dimension (e.g., 640 + 640 = 1280)
        COMBINED_DIM = AASIST_EMBED_DIM + ARNET_EMBED_DIM 

        # 3. Final Fusion Classification Layer
        self.fusion_fc = nn.Sequential(
            nn.Linear(COMBINED_DIM, COMBINED_DIM // 2),
            nn.BatchNorm1d(COMBINED_DIM // 2),
            nn.SELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(COMBINED_DIM // 2, 2) # Final binary classification output (0: Spoof, 1: Bonafide)
        )
        
    def forward(self, raw_audio: Tensor, handcrafted_features: Tensor, Freq_aug=False):
        """
        raw_audio: Input 1 (Raw audio data, typically (Batch, 64600)) for AASIST/RawNet Raw stream
        handcrafted_features: Input 2 (MFCC data, typically (Batch, Time, 40)) for RawNet MFCC stream
        """
        
        # 1. AASIST Stream (Raw Audio -> E_AASIST)
        # AASIST takes the raw audio input and internally computes features/spectrograms.
        # It returns (embedding, output). We discard the dummy output.
        E_AASIST, _ = self.aasist_extractor(raw_audio, Freq_aug=Freq_aug) 

        # 2. ARNet Stream (MFCCs + Raw Audio -> E_ARNet)
        E_ARNet = self.arnet_extractor(handcrafted_features, raw_audio) 

        # 3. Concatenation (Final Fusion)
        E_Combined = torch.cat((E_AASIST, E_ARNet), dim=1)

        # 4. Classification
        output = self.fusion_fc(E_Combined)
        
        # We return the combined feature vector and the final classification output
        return E_Combined, output