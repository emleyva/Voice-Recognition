import torch
import torchaudio
from pathlib import Path
from inference import DigitRecognizer

# Find checkpoint path (works from any working directory)
SCRIPT_DIR = Path(__file__).parent
CHECKPOINT_PATH = SCRIPT_DIR.parent / 'checkpoints' / 'best_model.pt'


def test_window_strategy(audio_path, window_size, stride=None, label=None):
    """
    Test a specific window size and stride on audio file.
    stride defaults to window_size (non-overlapping)
    """
    if stride is None:
        stride = window_size

    recognizer = DigitRecognizer(model_path=str(CHECKPOINT_PATH), confidence_threshold=0.8)

    waveform, sr = torchaudio.load(audio_path)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    sample_rate = 16000
    window_samples = int(window_size * sample_rate)
    stride_samples = int(stride * sample_rate)

    recognized_digits = []
    chunk_idx = 0
    start = 0

    while start < waveform.shape[1]:
        end = min(start + window_samples, waveform.shape[1])
        chunk = waveform[:, start:end]

        label_result, confidence = recognizer.recognize(chunk)

        if label_result != "unrecognized":
            recognized_digits.append((label_result, confidence, chunk_idx))

        chunk_idx += 1
        start += stride_samples

        if end == waveform.shape[1]:
            break

    return recognized_digits


def compare_strategies(audio_path):
    """
    Test multiple window strategies and compare results.
    """
    strategies = [
        (0.5, 0.5, "0.5s non-overlapping"),
        (0.5, 0.25, "0.5s with 50% overlap (0.25s stride)"),
        (1.0, 1.0, "1.0s non-overlapping (baseline)"),
        (0.75, 0.375, "0.75s with 50% overlap (0.375s stride)"),
        (0.25, 0.25, "0.25s non-overlapping (fine-grained)"),
        (1.0, 0.5, "1.0s with 50% overlap (0.5s stride)"),
    ]

    print("=" * 70)
    print(f"Testing window strategies on: {audio_path}")
    print("=" * 70)
    print()

    results = {}

    for window_size, stride, label in strategies:
        print(f"Testing: {label}")
        print(f"  Window: {window_size}s, Stride: {stride}s")

        digits = test_window_strategy(audio_path, window_size, stride)

        if digits:
            digit_list = [d[0] for d in digits]
            print(f"  Recognized: {', '.join(digit_list)}")
            print(f"  Details: {len(digits)} digits detected")
            for digit, conf, chunk in digits:
                print(f"    - {digit} (confidence: {conf:.4f}, chunk: {chunk})")
        else:
            print(f"  Recognized: None")

        results[label] = digits
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for label, digits in results.items():
        count = len(digits)
        if count > 0:
            digit_str = ", ".join([d[0] for d in digits])
            print(f"{label:45} → {count} digits: {digit_str}")
        else:
            print(f"{label:45} → No digits detected")

    return results


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_windows.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    compare_strategies(audio_path)
