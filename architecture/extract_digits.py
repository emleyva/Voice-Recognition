import torch
import torchaudio
from inference import DigitRecognizer


def extract_digits_from_audio(audio_path, confidence_threshold=0.8):
    """
    Extract all recognized digits from an audio file.
    Segments audio into 1-second chunks and runs inference on each.
    """
    recognizer = DigitRecognizer(confidence_threshold=confidence_threshold)

    waveform, sr = torchaudio.load(audio_path)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    sample_rate = 16000
    chunk_size = sample_rate
    recognized_digits = []

    num_chunks = (waveform.shape[1] + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, waveform.shape[1])
        chunk = waveform[:, start:end]

        label, confidence = recognizer.recognize(chunk)

        if label != "unrecognized":
            recognized_digits.append((label, confidence, i))
            print(f"Chunk {i}: {label} (confidence: {confidence:.4f})")
        else:
            print(f"Chunk {i}: unrecognized (confidence: {confidence:.4f})")

    return recognized_digits


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_digits.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    digits = extract_digits_from_audio(audio_path)

    print("\n" + "="*50)
    print("Recognized digits:")
    if digits:
        for label, confidence, chunk_idx in digits:
            print(f"  {label} (chunk {chunk_idx}, confidence: {confidence:.4f})")
    else:
        print("  None")
    print("="*50)
