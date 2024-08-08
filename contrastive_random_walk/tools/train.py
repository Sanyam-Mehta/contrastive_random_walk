import torch
from contrastive_random_walk.model.crw import ContrastiveRandomWalkLightningWrapper
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

# Initialize the trainer
trainer = L.Trainer(max_epochs=1)

# Train the model
trainer.fit(model, train_dataloader, val_dataloader)