import torch
from contrastive_random_walk.model.crw import ContrastiveRandomWalk

# TODO: Use Pytorch lightning for training

model = ContrastiveRandomWalk(
    resnet_type="resnet18", output_dim=128, temperature=1.0, edge_dropout_rate=0.5
)

# train dataloader
train_dataloader = torch.randn(16, 5, 10, 224, 224, 3)  # (B, T, N, H, W, C)

# validation dataloader
val_dataloader = torch.randn(16, 5, 10, 224, 224, 3)  # (B, T, N, H, W, C)

# test dataloader
test_dataloader = torch.randn(16, 5, 10, 224, 224, 3)  # (B, T, N, H, W, C)
