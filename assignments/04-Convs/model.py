import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(num_channels, 32, 2)
        self.conv2 = nn.Conv2d(32, 32, 2)
        self.conv3 = nn.Conv2d(32, 24, 2)
        self.conv4 = nn.Conv2d(24, 32, 2)
        self.fc1 = nn.Linear(32, 26)
        self.fc2 = nn.Linear(26, 20)
        self.fc3 = nn.Linear(20, 15)
        self.fc4 = nn.Linear(15, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.softmax(x, dim=0)
        return x
