import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset

LABELS = [
    "zero", "one", "two", "three", "four", 
    "five", "six", "seven", "eight", "nine"
    ]

SAMPLE_RATE = 16000 #16khz is standard sampling rate
N_MFCC = 30 #lower end of standard to reduce computing power

class SpeechCommands(Dataset):
    def __init__(self, subset='training'):
        #uses google speech commands v2 dataset 
        self.data = torchaudio.datasets.SPEECHCOMMANDS(
            root="./data",
            subset = subset,
            download=True
        )
    
        self.data = [
            d for d in self.data if d[2] in LABELS #removing all samples not in defined labels
        ]
        self.mfcc_transform = T.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            melkwargs={
                'n_fft': 400, #25 ms window @ 16khz
                'hop_length': 160, # 10 ms hop step @ 16khz
                'n_mels': 40, # 40 mel filter banks applied 
                'f_max': 8000 # highest frequency in mel filter banks (8000hz)
            }
        )
        
        #augments audio to be more accurate to real audio
        self.freq_mask = T.FrequencyMasking(freq_mask_param=10)
        self.time_mask = T.TimeMasking(time_mask_param=20)
        self.subset = subset
        
        self.label_idx = {label: i for i, label in enumerate(LABELS)}
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform, sample_rate, label, _, _ = self.data[idx]
        
        if waveform.shape[-1] < SAMPLE_RATE:
            #padding samples lower than 16khz sample rate to fit size
            pad = SAMPLE_RATE - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad)) #adds pad # of zeroes to waveform to induce silence
        else:
            waveform = waveform[:, :SAMPLE_RATE] #truncates waveform to fit sample rate
            
        # MFCC transformation is in shape (1, 30, 101)
        # 30 coefficients (N_MFCC), ~101 time frames for 1s @ hop_length=160
        mfcc = self.mfcc_transform(waveform)
        
        if self.subset == 'training':
            mfcc = self.freq_mask(mfcc)
            mfcc = self.time_mask(mfcc)
        
        label_idx = self.label_idx[label]
        return mfcc, label_idx
            