import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from preprocessing import SpeechCommands, LABELS
from model import WordClassifierCNN
from torch.utils.data import DataLoader

def predict_with_confidence(logits, confidence_threshold=0.8):
    """
    Get predictions with confidence filtering.
    Returns predicted class indices and -1 for predictions below threshold.
    """
    probabilities = F.softmax(logits, dim=1)
    confidences, preds = torch.max(probabilities, dim=1)

    preds = preds.cpu()
    confidences = confidences.cpu()

    preds[confidences < confidence_threshold] = -1
    return preds, confidences

def evaluate(confidence_threshold=0.8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = SpeechCommands(subset='testing')
    test_loader  = DataLoader(test_dataset, batch_size=64)

    model = WordClassifierCNN(n_classes=10).to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pt'))
    model.eval()

    all_preds, all_labels, all_confidences = [], [], []

    with torch.no_grad():
        for mfcc, labels in test_loader:
            mfcc = mfcc.to(device)
            logits = model(mfcc)
            preds, confidences = predict_with_confidence(logits, confidence_threshold)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_confidences.extend(confidences.tolist())

    recognized_mask = [p != -1 for p in all_preds]
    recognized_count = sum(recognized_mask)
    total_count = len(all_preds)

    print(f"Recognized: {recognized_count}/{total_count} ({100*recognized_count/total_count:.1f}%)")
    print(f"Confidence threshold: {confidence_threshold}\n")

    if recognized_count > 0:
        filtered_preds = [p for p, m in zip(all_preds, recognized_mask) if m]
        filtered_labels = [l for l, m in zip(all_labels, recognized_mask) if m]

        print("Classification Report (recognized predictions only):")
        print(classification_report(filtered_labels, filtered_preds, target_names=LABELS))

        cm = confusion_matrix(filtered_labels, filtered_preds, labels=list(range(10)))
        disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png')
        plt.show()
    else:
        print("No predictions recognized above threshold!")

if __name__ == '__main__':
    evaluate()