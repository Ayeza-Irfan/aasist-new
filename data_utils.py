import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchaudio # <-- NEW IMPORT

# Define the sample rate and fixed length used across all data processing
SAMPLE_RATE = 16000
MAX_LEN_SAMPLES = 64600  # ~4 sec audio (64600 samples)

# --- MFCC Feature Calculator ---
# Configuration matching typical RawNet/ASVspoof MFCC setup
MFCC_TRANSFORM = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=40, # As per ARNet_config in .conf file
    melkwargs={
        "n_fft": 512,
        "hop_length": 160,
        "n_mels": 64,
        "f_min": 20,
        "f_max": 8000
    }
)

def calculate_mfcc(audio_tensor: Tensor) -> Tensor:
    """Calculates 40-dim MFCCs from a raw audio tensor."""
    # torchaudio expects shape (num_channels, num_samples). Our input is (num_samples,)
    # We add a channel dimension: (1, num_samples)
    audio_tensor = audio_tensor.unsqueeze(0)
    mfcc_output = MFCC_TRANSFORM(audio_tensor)
    
    # Remove the channel dimension: (40, time) -> Transpose to (time, 40)
    # The AASIST / ARNet MFCC path expects (time, dim)
    return mfcc_output.squeeze(0).transpose(0, 1)


def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad_random(x: np.ndarray, max_len: int) -> np.ndarray:
    x_len = x.shape[0]
    if x_len > max_len:
        # if too long
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = MAX_LEN_SAMPLES  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad_random(X, self.cut)
        
        # --- DUAL INPUT GENERATION (MODIFICATION) ---
        x_inp_raw = Tensor(X_pad) # Input 1: Raw audio for AASIST
        x_inp_mfcc = calculate_mfcc(x_inp_raw) # Input 2: MFCCs for ARNet stream

        y = self.labels[key]
        return x_inp_raw, x_inp_mfcc, y # <-- RETURNS DUAL INPUT
        


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = MAX_LEN_SAMPLES

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad_random(X, self.cut)

        # --- DUAL INPUT GENERATION (MODIFICATION) ---
        x_inp_raw = Tensor(X_pad) # Input 1: Raw audio for AASIST
        x_inp_mfcc = calculate_mfcc(x_inp_raw) # Input 2: MFCCs for ARNet stream
        
        return x_inp_raw, x_inp_mfcc, key # <-- RETURNS DUAL INPUT + UTTERANCE KEY