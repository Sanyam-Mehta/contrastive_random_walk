import torch.nn as nn
import torch
from contrastive_random_walk.model.encoders import VideoEncoder
from contrastive_random_walk.model.similarity import get_affinity_matrices

import lightning as L


class ContrastiveRandomWalkLoss(nn.Module):
    def __init__(self):
        super(ContrastiveRandomWalkLoss, self).__init__()

    def forward(self, global_affinity_matrix):
        # # Take the sum of the negative log of the diagonal elements of the global affinity matrix
        # # This is the loss function for the contrastive random walk model

        # # global_affinity_matrix shape: (B, N, N)

        # loss = 0
        
        # B, _, _ = global_affinity_matrix.shape

        # for i in range(B):
        #     single_global_affinity_matrix = global_affinity_matrix[i]

        #     # Take the log of the diagonal elements
        #     log_diag = torch.log(torch.diag(single_global_affinity_matrix))

        #     loss += -torch.sum(log_diag)

        # return loss

        #### Optimized version ####
        # Extract the diagonal elements for all batches
        diag_elements = torch.diagonal(global_affinity_matrix, dim1=-2, dim2=-1)
        
        # Take the log of the diagonal elements
        log_diag = torch.log(diag_elements)
        
        # Sum the negative log values
        loss = -torch.sum(log_diag)

        return loss
            


class ContrastiveRandomWalk(nn.Module):
    def __init__(
        self,
        resnet_type="resnet18",
        output_dim=128
    ):
        super(ContrastiveRandomWalk, self).__init__()
        self.video_encoder = VideoEncoder(
            resnet_type=resnet_type, output_dim=output_dim
        )

    def forward(self, video):
        # video shape: (B, T, N, H, W, C)
        # B: batch size
        # T: number of frames (2*clip_len. This is a palindrome of the original video)
        # N: number of patches (it is actually NxN. So if image is divided in a 7*7 grid, N here will be 49)
        # H: height of each patch (64)
        # W: width of each patch (64)
        # C: number of channels in the input tensor (3)

        B, T, N, H, W, C = video.shape

        # Encode the video using the video encoder
        video = self.video_encoder(video)

        # video shape: (B, T, N, D)
        return video
    

# TODO: Implement the LearnableSimilarity module (Not implemented in the original code)
class LearnableSimilarity(nn.Module):
    def __init__(self, input_dim=128, output_dim=128):
        super(LearnableSimilarity, self).__init__()
        self.self_similarity = nn.Linear(input_dim, output_dim)

    def forward(self, video):
        # video shape: (B, T, N, D)
        # This function computes the global affinity matrix using a learnable similarity function
        # Similarity functions computes the similarity between two adhacent nodes in the graph, i.e.:
        # For each batch item, the similarity function computes the similarity between the N X D embeddings at time t and time t+1
        # The output is a matrix of shape (B, T-1, N, N)
        return video



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
        # self.learnable_self_similarity = LearnableSimilarity()
        self.contrastive_random_walk_loss = ContrastiveRandomWalkLoss()
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        video = batch

        encoded_video = self.model(video)

        # enoded_video shape: (B, T, N, D)

        # Compute the global affinity matrix (B x N x N) [i.e. (B, 49, 49)]
        # These have non-learnable similarity function
        global_affinity_matrix, local_affinity_matrices, edge_dropped_local_affinity_matrices = get_affinity_matrices(
            encoded_video, self.temperature, self.edge_dropout_rate
        )

        # TODO: Could be implemented in the future. Not in the original paper.
        # # Learnable similarity function
        # # video shape: (B, T, N, D)
        # global_affinity_matrix = self.model.self_similarity(video)

        # Compute loss here
        loss = self.contrastive_random_walk_loss(global_affinity_matrix)

        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        video = batch
        global_affinity_matrix = self.model(video)

        # Compute loss here
        loss = self.contrastive_random_walk_loss(global_affinity_matrix)

        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
