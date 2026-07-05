import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from model import WordClassifierCNN
from preprocessing import LABELS, SAMPLE_RATE, N_MFCC

class DigitRecognizer:
    def __init__(self, model_path='./checkpoints/best_model.pt', confidence_threshold=0.8, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold

        self.model = WordClassifierCNN(n_classes=10).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.mfcc_transform = T.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            melkwargs={
                'n_fft': 400,
                'hop_length': 160,
                'n_mels': 40,
                'f_max': 8000
            }
        )

    def _preprocess_audio(self, waveform):
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if waveform.shape[-1] < SAMPLE_RATE:
            pad = SAMPLE_RATE - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :SAMPLE_RATE]

        mfcc = self.mfcc_transform(waveform)
        return mfcc.unsqueeze(0)

    def recognize_from_file(self, audio_path):
        """
        Load audio from file and recognize spoken digit.
        Returns: (recognized_label, confidence) or ("unrecognized", confidence)
        """
        waveform, sr = torchaudio.load(audio_path)

        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        return self.recognize(waveform)

    def recognize(self, waveform):
        """
        Recognize digit from waveform tensor.
        Args:
            waveform: torch.Tensor of shape (channels, samples)
        Returns:
            (recognized_label: str, confidence: float)
        """
        mfcc = self._preprocess_audio(waveform)
        mfcc = mfcc.to(self.device)

        with torch.no_grad():
            logits = self.model(mfcc)
            probabilities = F.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probabilities, dim=1)

        confidence = confidence.item()
        pred_idx = pred_idx.item()

        if confidence >= self.confidence_threshold:
            return LABELS[pred_idx], confidence
        else:
            return "unrecognized", confidence


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    recognizer = DigitRecognizer()
    label, confidence = recognizer.recognize_from_file(audio_path)

    print(f"Recognized: {label}")
    print(f"Confidence: {confidence:.4f}")
