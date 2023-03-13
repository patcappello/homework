import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    A simple CNN with 4 convolutional layers and 4 fully-connected layers.
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(num_channels, 12, 3)
        self.conv2 = nn.Conv2d(12, 24, 3)
        self.conv3 = nn.Conv2d(24, 32, 2)
        self.conv4 = nn.Conv2d(32, 32, 2)
        self.fc1 = nn.Linear(32, 24)
        self.fc2 = nn.Linear(24, 12)
        # self.fc3 = nn.Linear(20, 15)
        self.fc4 = nn.Linear(12, num_classes)
        self.batchnorm2d = nn.BatchNorm2d(12)
        self.batchnorm1d0 = nn.BatchNorm1d(32)
        self.batchnorm1d1 = nn.BatchNorm1d(24)
        self.batchnorm1d2 = nn.BatchNorm1d(16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        Arguments:
            x: The input data.
        Returns:
            The output of the network.
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (3, 3))
        x = self.batchnorm2d(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # x = self.batchnorm2d(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        # x = self.batchnorm2d(x)
        # x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        x = torch.flatten(x, 1)
        x = self.batchnorm1d0(x)
        x = F.relu(self.fc1(x))
        # x = self.batchnorm1d1(x)
        x = F.relu(self.fc2(x))
        # x = self.batchnorm1d2(x)
        # x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.softmax(x, dim=0)
        return x
