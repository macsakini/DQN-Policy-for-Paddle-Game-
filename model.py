import torch
from torch import nn
import torch.optim as optim


class DQNModel(nn.Module):
    def __init__(self):
        super(DQNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 3)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.loss = nn.MSELoss()

    def forward(self, x):
        logits = self.model(x)
        return logits
