import torch.nn as nn

from contrastive_random_walk.model.encoders import VideoEncoder
from contrastive_random_walk.model.similarity import get_global_affinity_matrix


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

        # Compute the global affinity matrix
        global_affinity_matrix = get_global_affinity_matrix(
            video, self.temperature, self.edge_dropout_rate
        )

        return global_affinity_matrix
