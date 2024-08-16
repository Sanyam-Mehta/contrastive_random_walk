import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA


def draw_matches(image_1, image_2, embeddings_image_1, embeddings_image_2):
    # image_1 and image_2 are the original images
    # image_1 dimenisons: (H, W, C)
    # embeddings dimensions: N x D (N is H*W; D is the embedding dimension)

    image_1, image_2 = cv2.resize(image_1, (400, 400)), cv2.resize(image_2, (400, 400))

    # crossCheck=True will return only the best matches if keypoints in both images are best matches of each other
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # height is 20 pixels per row
    height = int(embeddings_image_1.shape[-1] ** 0.5)

    matches = bf.match(
        embeddings_image_1.cpu().detach().numpy(),
        embeddings_image_2.cpu().detach().numpy(),
    )

    scale = image_1.shape[-2] / height

    grid = torch.stack(
        [
            torch.arange(0, height)[None].repeat(height, 1),
            torch.arange(0, height)[:, None].repeat(1, height),
        ]
    )

    grid = grid.view(2, -1)
    grid = grid * scale + scale // 2

    # Extracts keypoints that lie on a 20 x 20 grid
    kps = [cv2.KeyPoint(grid[0][i], grid[1][i], 1) for i in range(grid.shape[-1])]

    # Sort the matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)

    # img1 = img2 = np.zeros((40, 40, 3))
    out = cv2.drawMatches(
        image_1.astype(np.uint8),
        kps,
        image_2.astype(np.uint8),
        kps,
        matches[:],
        None,
        flags=2,
    ).transpose(2, 0, 1)

    return out


def pca_feats_top_3K_components(embeddings_all_images, K=1):
    # embeddings_all_images shape: (N, H, W, C) (C is the embedding dimension, N is the number of images)

    # TODO: From my memory, PCA algo requires zero centered data. Not sure if PCA handles that internally.
    # Also, whiten=True makes data zero-centered and unit variance. This makes component analysis easier as
    # algo could default to hard-wired behaviour. Check this.
    pca = PCA(
        n_components=3 * K,
        svd_solver="full",
        whiten=True,
    )

    pca_output = torch.tensor(
        pca.fit_transform(
            embeddings_all_images.reshape(-1, embeddings_all_images.shape[-1]).numpy()
        )
    )

    pca_output_reshaped = pca_output.reshape(
        embeddings_all_images.shape[0],
        embeddings_all_images.shape[1],
        embeddings_all_images.shape[2],
        3 * K,
    )

    pca_output_all_images = [
        pca_output_reshaped[i] for i in range(pca_output_reshaped.shape[0])
    ]

    # Normalize the PCA output
    pca_output_all_images = [
        (pca_output_all_images[i] - pca_output_all_images[i].min())
        / (pca_output_all_images[i].max() - pca_output_all_images[i].min())
        for i in range(len(pca_output_all_images))
    ]

    return pca_output_all_images
