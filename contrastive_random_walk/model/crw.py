from collections import OrderedDict

import lightning as L

import numpy as np
import torch
import torch.nn as nn
import einops
from contrastive_random_walk.utils import combine_patches_into_image, divide_image_into_patches

from contrastive_random_walk.model.encoders import VideoEncoder
from contrastive_random_walk.model.similarity import get_affinity_matrices_all_walks
from contrastive_random_walk.viz.visualize_utils import (
    draw_matches,
    pca_feats_top_3K_components,
)


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
        output_dim=128,
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
        palindromic_dataset=False,
        visualizer=None,
        train_viz_freq=10,
        val_viz_freq=10,
    ):
        super(ContrastiveRandomWalkLightningWrapper, self).__init__()
        self.model = ContrastiveRandomWalk(
            resnet_type=resnet_type,
            output_dim=output_dim
        )
        self.temperature = temperature
        self.edge_dropout_rate = edge_dropout_rate
        # self.learnable_self_similarity = LearnableSimilarity()
        self.contrastive_random_walk_loss = ContrastiveRandomWalkLoss()
        self.learning_rate = learning_rate
        self.palindromic_dataset = palindromic_dataset
        self.train_viz_freq = train_viz_freq
        self.val_viz_freq = val_viz_freq
        self.visualizer = visualizer

    def training_step(self, batch, batch_idx):
        
        video_patches, video = batch[0], batch[1]
        

        # print("Encoding Video")
        encoded_video = self.model(video_patches)

        # encoded_video shape: (B, T, N, D)

        # Compute the global affinity matrix (B x N x N) [i.e. (B, 49, 49)]
        # These have non-learnable similarity function
        # global_affinity_matrix, local_affinity_matrices, edge_dropped_local_affinity_matrices = get_affinity_matrices(
        #     encoded_video, self.temperature, self.edge_dropout_rate
        # )
        # TODO: Could be implemented in the future. Not in the original paper.
        # # Learnable similarity function
        # # video shape: (B, T, N, D)
        # global_affinity_matrix = self.model.self_similarity(video)

        # Compute loss here
        # loss = self.contrastive_random_walk_loss(global_affinity_matrix)

        (
            global_affinity_matrix_all_walks_dict,
            local_affinity_matrices,
            edge_dropped_local_affinity_matrices,
        ) = get_affinity_matrices_all_walks(
            encoded_video, self.temperature, self.edge_dropout_rate
        )
        # print("We have affinity matrices")

        # Compute loss here
        loss_all_walks = []

        # TODO: Loss could be weighted by the length of the walk.
        # print("Computing loss")
        for (
            walk_len,
            global_affinity_matrix,
        ) in global_affinity_matrix_all_walks_dict.items():
            loss_all_walks.append(
                self.contrastive_random_walk_loss(global_affinity_matrix)
            )
        # print("Loss computed")

        # Take the mean of the losses
        loss = torch.mean(torch.stack(loss_all_walks))

        if self.current_epoch % self.train_viz_freq == 0:
            # Visualize the video
            print("Visualizing the video")
            visuals = self.get_visuals(video, self.current_epoch)
            print("Displaying the results")
            self.visualizer.display_current_results(visuals, self.current_epoch)

        self.log("train_loss", loss)

        # if epoch is divisible by 10, visualize the video

        return loss

    def validation_step(self, batch, batch_idx):
        video_patches, video = batch[0], batch[1]

        encoded_video = self.model(video_patches)

        (
            global_affinity_matrix_all_walks_dict,
            local_affinity_matrices,
            edge_dropped_local_affinity_matrices,
        ) = get_affinity_matrices_all_walks(
            encoded_video, self.temperature, self.edge_dropout_rate
        )

        # Compute loss here
        loss_all_walks = []

        # TODO: Loss could be weighted by the length of the walk.
        for (
            walk_len,
            global_affinity_matrix,
        ) in global_affinity_matrix_all_walks_dict.items():
            loss_all_walks.append(
                self.contrastive_random_walk_loss(global_affinity_matrix)
            )

        # Take the mean of the losses
        loss = torch.mean(torch.stack(loss_all_walks))

        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_visuals(self, video, step):

        # Video is the original, unpatched video
        # Interpolate the video from 256x256 to 448x448. Input size: (B, T, 1, H, W, C)
        # From the documentation, the interpolate function says that:
        # The input dimensions are interpreted in the form: mini-batch x channels x [optional depth] x [optional height] x width.
        
        B, T, N, H, W, C = video.shape
        # rearrange the dimensions to match the expected input format i.e. (B*T, C, H, W)
        video = einops.rearrange(video, "B T N H W C -> (B T N) C H W")
        video = torch.nn.functional.interpolate(video, size=(448, 448))

        # divide it into 16x16 patches
        grid_size = 16
        patchified_video = divide_image_into_patches(
            einops.rearrange(video, "(B T) C H W -> (B T) H W C", B=B, T=T),
            grid_size=grid_size,
        )

        patchified_video = einops.rearrange(
            patchified_video, "(B T) N H W C -> B T N H W C", B=B, T=T
        )

        print("Interpolated video shape after patchifications: ", patchified_video.shape, video.shape)

        # # rearrange the dimensions back to the original format
        video = einops.rearrange(video, "(B T) C H W -> B T 1 H W C", B=B, T=T)

        # Dimensions: B, T, 1, H, W, C (H==W)
        # assert video.shape[2] == 1, "Video should have only one patch"
        # assert video.shape[3] == video.shape[4], "Height and width should be equal"

        # Encode the video using the video encoder
        encoded_video = self.model(patchified_video)
        # encoded_video shape: (B, T, N, D)

        # randomly select two frames from batch 0
        t1, t2 = np.random.randint(0, encoded_video.shape[1], (2))

        # extract the features for the two frames
        frame1_descriptors = encoded_video[0, t1]  # shape: (N, D)
        frame2_descriptors = encoded_video[0, t2]

        # extract tensor for the two frames
        frame1 = video[0, t1, 0]
        frame2 = video[0, t2, 0]

        # Convert the tensor to numpy array, cv2 expect channels to be last
        image_1 = frame1.cpu().numpy()
        image_2 = frame2.cpu().numpy()

        # draw matches between the two frames using draw_matches function
        drawn_matches = draw_matches(
            image_1=image_1,
            image_2=image_2,
            embeddings_image_1=frame1_descriptors,
            embeddings_image_2=frame2_descriptors,
            grid_size=grid_size,
        )

        # extract one video clip from batch 0 and visualize top 3K components using pca_feats_top_3K_components function
        # video_clip = video[0].squeeze(1)  # shape: (T, H, W, C)

        # Apply PCA to the video clip
        # pca_output = pca_feats_top_3K_components(video_clip)

        # ############## Display results and errors ############
        # if self.opt.isTrain:
        #     self.trainingavgloss = np.mean(self.averageloss)
        #     if self.opt.verbose:
        #         print ('  Iter: %d, Loss: %f' % (step, self.trainingavgloss) )
        #     self.writer.add_scalar(self.opt.name+'/trainingloss/', self.trainingavgloss, step)
        #     self.averageloss = []

        # Label, Image
        print("Creating ordered dict (channel last)")
        print("Shapes: ", image_1.shape, image_2.shape, drawn_matches.shape)
        ordered_dict = OrderedDict(
            [
                (f"frame_1_idx_{t1}", image_1),
                (f"frame_2_idx_{t2}", image_2),
                ("drawn_matches", drawn_matches),
                # ("pca_output", pca_output),
            ]
        )
        return ordered_dict
