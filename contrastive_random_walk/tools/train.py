import torch
from contrastive_random_walk.model.crw import ContrastiveRandomWalkLightningWrapper
from contrastive_random_walk.data.video_dataset import VideoList
import lightning as L

# train dataloader
train_dataloader = torch.randn(2, 16, 5, 10, 224, 224, 3)  # (Total Batches, B, T, N, H, W, C)

# validation dataloader
val_dataloader = torch.randn(2, 16, 5, 10, 224, 224, 3)  # (Total Batches, B, T, N, H, W, C)

# test dataloader
test_dataloader = torch.randn(2, 16, 5, 10, 224, 224, 3)  # (Total Batches, B, T, N, H, W, C)

# Initialize the model
model = ContrastiveRandomWalkLightningWrapper(
    resnet_type="resnet18", output_dim=128, temperature=1.0, edge_dropout_rate=0.5, learning_rate=1e-3
)


# TODO: Check if transforms are passed correctly
train_dataset = VideoList(
    filelist="data/kinetics700/trainlist.txt",
    clip_len=16,
    is_train=True,
    frame_gap=1,
    transform=None,
    random_clip=True,
)

# Each element in the dataset is a tensor of size (2*T, NxN, H, W, C), where:
# T: clip length
# N: number of patches
# H: height of each patch (64)
# W: width of each patch (64)
# C: number of channels in the input tensor (3)

# Initialize the trainer
trainer = L.Trainer(max_epochs=1)

# Train the model
trainer.fit(model, train_dataloader, val_dataloader)