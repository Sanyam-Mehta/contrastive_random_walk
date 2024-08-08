import torch.nn as nn
import torch
from contrastive_random_walk.model.encoders import VideoEncoder
from contrastive_random_walk.model.similarity import get_global_affinity_matrix

import lightning as L


class ContrastiveRandomWalk(nn.Module):
    def __init__(
        self,
        resnet_type="resnet18",
        output_dim=128,
        temperature=1.0,
        edge_dropout_rate=0.5,
    ):
        super(ContrastiveRandomWalk, self).__init__()
        self.video_encoder = VideoEncoder(
            resnet_type=resnet_type, output_dim=output_dim
        )
        self.temperature = temperature
        self.edge_dropout_rate = edge_dropout_rate

    def forward(self, video):
        # video shape: (B, T, N, H, W, C)
        # B: batch size
        # T: number of frames
        # N: number of patches
        # H: height of each patch
        # W: width of each patch
        # C: number of channels in the input tensor

        B, T, N, H, W, C = video.shape

        # Encode the video using the video encoder
        video = self.video_encoder(video)

        # Compute the global affinity matrix (B x N x N)
        global_affinity_matrix = get_global_affinity_matrix(
            video, self.temperature, self.edge_dropout_rate
        )

        return global_affinity_matrix



class ContrastiveRandomWalkLightningWrapper(L.LightningModule):
    def __init__(
        self,
        resnet_type="resnet18",
        output_dim=128,
        temperature=1.0,
        edge_dropout_rate=0.5,
        learning_rate=1e-3,
    ):
        super(ContrastiveRandomWalkLightningWrapper, self).__init__()
        self.model = ContrastiveRandomWalk(
            resnet_type=resnet_type,
            output_dim=output_dim,
            temperature=temperature,
            edge_dropout_rate=edge_dropout_rate,
        )
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        video = batch
        global_affinity_matrix = self.model(video)

        # Compute loss here
        loss = self.compute_contrastive_random_walk_loss(global_affinity_matrix)

        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        video = batch
        global_affinity_matrix = self.model(video)

        # Compute loss here
        loss = self.compute_contrastive_random_walk_loss(global_affinity_matrix)

        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    

    def compute_contrastive_random_walk_loss(self, global_affinity_matrix):
        # Compute the CE loss between:
        # 1. The global affinity matrix (predicted) [B x N x N]
        # 2. The identity matrix (ground truth) [B x N x N]
        B, N, _ = global_affinity_matrix.size()
        
        # Ground truth identity matrix
        ground_truth = torch.eye(N).unsqueeze(0).repeat(B, 1, 1)

        # Flatten the matrices to shape [B*N, N] for CrossEntropyLoss
        predicted_flat = global_affinity_matrix.view(B * N, N)
        ground_truth_flat = ground_truth.view(B * N, N).argmax(dim=1)

        # Compute the CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(predicted_flat, ground_truth_flat)
        return loss


        
