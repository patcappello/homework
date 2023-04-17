import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 20)
        self.fc2 = nn.Linear(20, 25)
        self.fc3 = nn.Linear(25, 10)
        self.fc4 = nn.Linear(10, 4)
        self.float()

    def forward(self, x):
        # x = torch.from_numpy(x)
        x = x.to(torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x))
        return x
