import torch
from model import MLP


def create_model(input_dim: int, output_dim: int) -> MLP:
    """
    Create a multi-layer perceptron model.

    Arguments:
        input_dim (int): The dimension of the input data.
        output_dim (int): The dimension of the output data.
        hidden_dims (list): The dimensions of the hidden layers.

    Returns:
        MLP: The created model.

    """
    return MLP(
        input_dim,
        [128, 64, 32, 32],
        output_dim,
        4,
        torch.nn.functional.relu,
        torch.nn.init.ones_,
    )
