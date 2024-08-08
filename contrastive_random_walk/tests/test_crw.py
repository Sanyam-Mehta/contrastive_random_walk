import numpy as np
import torch

import torch.nn as nn
from contrastive_random_walk.model.crw import ContrastiveRandomWalk
from contrastive_random_walk.model.similarity import get_global_affinity_matrix


def test_contrastive_random_walk():
    # Test case 1: Random input tensor
    model = ContrastiveRandomWalk(
        resnet_type="resnet18", output_dim=128, temperature=1.0, edge_dropout_rate=0.5
    )
    input_tensor = torch.randn(16, 5, 10, 224, 224, 3)  # (B, T, N, H, W, C)
    output = model(input_tensor)  # output shape: (B, N, N)
    assert output.shape == (16, 10, 10)

    # Test case 2: Empty input tensor
    model = ContrastiveRandomWalk(
        resnet_type="resnet18", output_dim=128, temperature=1.0, edge_dropout_rate=0.5
    )
    input_tensor = torch.empty(0, 5, 10, 224, 224, 3)  # (B, T, N, H, W, C)
    output = model(input_tensor)  # output shape: (B, N, N)
    assert output.shape == (0, 10, 10)

    # Test case 3: Input tensor with negative values
    model = ContrastiveRandomWalk(
        resnet_type="resnet18", output_dim=128, temperature=1.0, edge_dropout_rate=0.5
    )
    input_tensor = torch.randn(8, 10, 10, 224, 224, 3)  # (B, T, N, H, W, C)
    input_tensor[:, :, :, :, :, 0] -= 1  # Subtract 1 from the first channel
    output = model(input_tensor)  # output shape: (B, N, N)
    assert torch.all(output >= 0)

    # Test case 4: Input tensor with all zeros
    model = ContrastiveRandomWalk(
        resnet_type="resnet18", output_dim=128, temperature=1.0, edge_dropout_rate=0.5
    )
    input_tensor = torch.zeros(4, 5, 10, 224, 224, 3)  # (B, T, N, H, W, C)
    output = model(input_tensor)  # output shape: (B, N, N)
    assert torch.all(output == 0)

    print("All test cases passed (ContrastiveRandomWalk)!")


test_contrastive_random_walk()
