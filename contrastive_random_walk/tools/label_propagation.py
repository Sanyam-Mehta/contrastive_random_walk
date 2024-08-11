# Refer notes and video: https://www.youtube.com/watch?v=UaOcjxrPaho

import torch
import torch.nn.functional as F

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

        

        


        