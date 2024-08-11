import torch
from contrastive_random_walk.model.crw import ContrastiveRandomWalkLightningWrapper
from contrastive_random_walk.data.kinetics_dataset import KineticsCustom
import lightning as L
from torchvision import transforms as T

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

transforms_video = T.Compose(
    [
        T.Resize(256),
        T.ColorJitter(),
    ]
)

tranformations_frame = T.Compose(
    [
        # RandomResizedCrop is the spatial jitter transofrmation
        T.RandomResizedCrop((64, 64), scale=(0.7, 0.9), ratio=(0.7, 1.3)),
    ]
)

# TODO: Check if transforms are passed correctly
train_dataset = KineticsCustom(
    root="data/kinetics400",
    split="train",
    frames_per_clip=5,
    step_between_clips=1,
    frame_rate=None,
    extensions=("mp4",),
    num_classes=400,
    transform_video=transforms_video,
    tranformations_frame=tranformations_frame,
)

val_dataset = KineticsCustom(
    root="data/kinetics400",
    split="val",
    frames_per_clip=5,
    step_between_clips=1,
    frame_rate=None,
    extensions=("mp4",),
    num_classes=400,
    transform_video=transforms_video,
    tranformations_frame=tranformations_frame,
)   

test_dataset = KineticsCustom(
    root="data/kinetics400",
    split="test",
    frames_per_clip=5,
    step_between_clips=1,
    frame_rate=None,
    extensions=("mp4",),
    num_classes=400,
    transform_video=None,
    tranformations_frame=tranformations_frame,
)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

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