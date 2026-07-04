import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from preprocessing import SpeechCommands, LABELS
from model import WordClassifierCNN
from torch.utils.data import DataLoader

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = SpeechCommands(subset='testing')
    test_loader  = DataLoader(test_dataset, batch_size=64)

    model = WordClassifierCNN(n_classes=10).to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pt'))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for mfcc, labels in test_loader:
            mfcc = mfcc.to(device)
            preds = model(mfcc).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print(classification_report(all_labels, all_preds, target_names=LABELS))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.show()

if __name__ == '__main__':
    evaluate()