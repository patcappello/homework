from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor


class CONFIG:
    batch_size = 32
    num_epochs = 10
    initial_learning_rate = 0.0005
    initial_weight_decay = initial_learning_rate / num_epochs

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "lr": initial_learning_rate,
        "decay": initial_weight_decay,
        "num_epochs": num_epochs,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [
            ToTensor(),
        ]
    )
