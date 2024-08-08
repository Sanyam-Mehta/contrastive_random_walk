import torch
from contrastive_random_walk.model.crw import ContrastiveRandomWalk


def test_contrastive_random_walk():
    # Test case 1: Random input tensor
    model = ContrastiveRandomWalk(
        resnet_type="resnet18", output_dim=128, temperature=1.0, edge_dropout_rate=0.5
    )
    input_tensor = torch.randn(16, 5, 10, 224, 224, 3)  # (B, T, N, H, W, C)
    output = model(input_tensor)  # output shape: (B, N, N)
    assert output.shape == (16, 10, 10)

    print("All test cases passed (ContrastiveRandomWalk)!")


test_contrastive_random_walk()
