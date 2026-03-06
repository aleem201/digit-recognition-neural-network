import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from src.model import DigitClassifier
from src.train import train
from src.evaluate import evaluate


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


dataset = datasets.MNIST(".", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(".", train=False, download=True, transform=transform)


train_set, val_set = random_split(dataset, [50000, 10000])

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128)
test_loader = DataLoader(test_dataset, batch_size=128)


model = DigitClassifier().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.02)


EPOCHS = 15


for epoch in range(EPOCHS):

    train_loss, train_acc = train(model, train_loader, optimizer, criterion, DEVICE)

    val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")