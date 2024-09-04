import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from contrastive_random_walk.data.kinetics_dataset_colab import KineticsCustom
from contrastive_random_walk.model.crw import ContrastiveRandomWalkLightningWrapper
from contrastive_random_walk.viz.visualizer import Visualizer
from torchvision import transforms as T

print("Training Started")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


transforms_video = T.Compose(
    [
        # T.ToPILImage(),
        T.Resize((256, 256)),
    ]
)

tranformations_frame = T.Compose(
    [
        T.ToPILImage(),
        # RandomResizedCrop is the spatial jitter transofrmation
        T.RandomResizedCrop((64, 64), scale=(0.7, 0.9), ratio=(0.7, 1.3)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

# TODO: Check if transforms are passed correctly
# Frame Rate 8 from Allan's Code
train_dataset = KineticsCustom(
    root="/content/drive/MyDrive/data/kinetics/videos",
    split="train",
    frames_per_clip=10,
    step_between_clips=1,
    frame_rate=8,
    extensions=("mp4",),
    num_classes=400,
    transform_video=transforms_video,
    tranformations_frame=tranformations_frame,
)

checkpoint_callback = ModelCheckpoint(
    dirpath='/content/drive/MyDrive/data/checkpoints_rerun_20240904/', 
    save_last=True,
    verbose=True,
    every_n_train_steps=30,
 )

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=8, shuffle=True
)
val_dataloader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=8, shuffle=False
)


# Each element in the dataset is a tensor of size (2*T, NxN, H, W, C), where:
# T: clip length
# N: number of patches
# H: height of each patch (64)
# W: width of each patch (64)
# C: number of channels in the input tensor (3)

# Initialize the visualizer
visualizer = Visualizer(
    tf_log=True,
    use_html=True,
    win_size=256,
    name="contrastive_random_walk_train",
    freq=1,  # every 100 epochs
)

print("Model Initialization")
# Initialize the model
# model = ContrastiveRandomWalkLightningWrapper(
#     resnet_type="resnet18",
#     output_dim=128,
#     temperature=1.0,
#     edge_dropout_rate=0.5,
#     learning_rate=1e-3,
#     visualizer=visualizer,
# ).to(device)
# From Allan's Code
model = ContrastiveRandomWalkLightningWrapper(
    resnet_type="resnet18",
    output_dim=128,
    temperature=0.07,
    edge_dropout_rate=0.1,
    learning_rate=1e-4,
    visualizer=visualizer,
).to(device)


print("Trainer Initialization")
# Initialize the trainer
trainer = L.Trainer(
    max_epochs=10, callbacks=[checkpoint_callback])

print("Starting Training")
# Train the model
trainer.fit(model, train_dataloader, val_dataloader, ckpt_path="")
