import torch
from torch.utils.data import DataLoader
from preprocessing import SpeechCommands
from model import WordClassifierCNN
import torch.nn as nn

def train(n_epochs=20, lr=1e-3, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = SpeechCommands(subset='training')
    val_dataset   = SpeechCommands(subset='validation')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

    model     = WordClassifierCNN(n_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(n_epochs):
        # training
        model.train()
        total_loss, correct, total = 0, 0, 0

        for mfcc, labels in train_loader:
            mfcc, labels = mfcc.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(mfcc)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(dim=1) == labels).sum().item()
            total      += labels.size(0)

        train_acc = correct / total

        # validation
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for mfcc, labels in val_loader:
                mfcc, labels = mfcc.to(device), labels.to(device)
                logits = model(mfcc)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        print(
            f"Epoch {epoch+1:02d} | "
            f"Loss: {total_loss/len(train_loader):.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}")

        # saves best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), './checkpoints/best_model.pt')

if __name__ == '__main__':
    train()