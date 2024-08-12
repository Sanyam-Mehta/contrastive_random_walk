# Refer notes and video: https://www.youtube.com/watch?v=UaOcjxrPaho

import torch
import torch.nn.functional as F
import numpy as np

def get_label_propagation_matrix(query_frame, target_frame, crw_model, temperature=1.0, top_k=5):    

    """
        query_frame: np.array : Query frame of shape (H, W, C)
        target_frame: np.array : Target frame of shape (H, W, C)
    """

    with torch.no_grad():
        # Get the embeddings for the query and target videos, query and taret shape: (B, T, N, H, W, C)
        # query and target are individual frames
        # Therefore, T = 1

        query = torch.tensor(query_frame).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        target = torch.tensor(target_frame).unsqueeze(0).unsqueeze(0).unsqueeze(0)
       
        features = crw_model(torch.cat((query, target), dim=0))

        # Get the embeddings for the query and target videos
        query_embeddings = features[:query.shape[0]]
        target_embeddings = features[query.shape[0]:]

        assert query_embeddings.shape == target_embeddings.shape
        assert len(query_embeddings.shape) == 4

        # query_embeddings shape: (B, T, N, D) and target_embeddings shape: (B, T, N, D)

        # Calculate the similarity matrix
        correlation_matrix = torch.einsum("btnd,btmd->btnm", query_embeddings, target_embeddings)

        # Compute the row-wise softmax of the correlation matrix
        probability_matrix = F.softmax(correlation_matrix / temperature, dim=-1)

        # probability_matrix shape: (B, T, N, N) [T = 1, i.e. (B, 1, N, N)]

        # Extract top-k weights and indices from each row of the probability matrix
        # Read the probability matrix as:
        # For each batch, for each time frame, for each query node, we have a distribution over N target nodes
        # We want tp extract the top-k target nodes with the highest probability
        # Therefore, the dimension along which we want to extract the top-k elements is the last dimension
        weights, ids = torch.topk(probability_matrix, top_k, dim=-1)


        # weights shape: (B, T, N, K) and ids shape: (B, T, N, K)
        # Reweight the probability matrix
        weights = F.softmax(weights, dim=-1)
        
        # weights: for each batch, for each time frame, for each query node, we have a distribution over top-k target nodes
        # ids: for each batch, for each time frame, for each query node, we have the indexes of the top-k target nodes

    return weights.cpu(), ids.cpu()

        

def propagate_labels(unlabelled_frame, labelled_frame, labels, crw_model):
    """
        unlabelled_frame: np.array : Unlabelled frame of shape (H, W, C)
        labelled_frame: np.array : Labelled frame of shape (H, W, C)
        labels: np.array : Labels of shape (N X C) [C is the number of classes]
        crw_model: torch.nn.Module : ContrastiveRandomWalk model

        The idea here is to extract the weights and indexes of the top-k target nodes, and then calculate a weighted sum of the labels of the top-k target nodes to obtain the label for the query node
    """

    # Get the label propagation matrix
    weights, ids = get_label_propagation_matrix(unlabelled_frame, labelled_frame, crw_model)

    # Weights should have dimensions (N, K) and ids should have dimensions (N, K)
    weights = weights.squeeze(0).squeeze(0)
    ids = ids.squeeze(0).squeeze(0)

    # labels shape: (N, C)
    # Extract the labels of the top-k target nodes
    top_k_labels = labels[ids]

    # top_k_labels shape: (N, K, C)
    # Calculate the weighted sum of the labels of the top-k target nodes
    propagated_labels = torch.einsum("nk, nkc -> nc", weights, top_k_labels)

    # Softmax the propagated labels
    propagated_labels = F.softmax(propagated_labels, dim=-1)

    # TODO -> Write a function to visualize the heatmap (check original code)
    # The idea is to use the propagated labels, and:
    # 1. Upsample the propagated labels to the size of the frame (N X C)
    # 2. For each 

    return propagated_labels.cpu()


def visualize_heatmap(frame, labels, weights, ids):
    """
        frame: np.array : Frame of shape (H, W, C)
        labels: np.array : Labels of shape (N X C) [C is the number of classes]
        weights: np.array : Weights of shape (N, K)
        ids: np.array : Indexes of the top-k target nodes of shape (N, K)
    """

    # Get the coordinates of the query node
    query_node = np.array([frame.shape[0] // 2, frame.shape[1] // 2])

    # Create an empty heatmap
    heatmap = np.zeros(frame.shape[:2])

    # Calculate the coordinates of the top-k target nodes
    target_nodes = np.array([np.unravel_index(i, frame.shape[:2]) for i in ids])

    # Calculate the distance between the query node and the target nodes
    distances = np.linalg.norm(target_nodes - query_node, axis=1)

    # Calculate the weights for each target node
    node_weights = weights * (1 / distances)

    # Normalize the weights
    node_weights = node_weights / np.sum(node_weights)

    # Create the heatmap
    for i, node in enumerate(target_nodes):
        heatmap[node[0], node[1]] = node_weights[i]

    # Plot the heatmap
    plt.imshow(frame)
    plt.imshow(heatmap, alpha=0.5)
    plt.show()

 
        