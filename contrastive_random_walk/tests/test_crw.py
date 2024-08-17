import torch
from contrastive_random_walk.model.crw import ContrastiveRandomWalk


def test_contrastive_random_walk():
    # Test case 1: Random input tensor
    output_dim = 128
    model = ContrastiveRandomWalk(
        resnet_type="resnet18", output_dim=output_dim
    )
    input_tensor = torch.randn(16, 5, 10, 64, 64, 3)  # (B, T, N, H, W, C)
    output = model(input_tensor)  # output shape: (B, T, N, D)

    print(output.shape)
    assert output.shape == (16, 5, 10, output_dim)

    print("All test cases passed (ContrastiveRandomWalk)!")


test_contrastive_random_walk()
