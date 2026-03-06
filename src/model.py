import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitClassifier(nn.Module):
    """
    Fully Connected Neural Network for MNIST digit classification.
    Architecture: 784 → 512 → 256 → 10
    """

    def __init__(self):
        super(DigitClassifier, self).__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)