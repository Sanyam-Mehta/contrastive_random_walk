import torch
from torch import nn

import torch.nn.functional as F

def softmax_similarity_function_local(input, temperature=1.0):
    """
    Compute softmax similarity function between consecutive frames of input tensor.

    Args:
        input (torch.Tensor): Input tensor of dimensions B x T x N x D.
        temperature (float): Temperature parameter for softmax function.

    Returns:
        torch.Tensor: Output tensor of dimensions B x (T-1) x N x N.
    """
    B, T, N, D = input.size()

    # Compute correlation matrix between nodes of two consecutive frames using einsum
    correlation_matrix = torch.einsum('btnd,btmd->btnm', input[:, :-1], input[:, 1:])

    # Compute the row-wise softmax of the correlation matrix
    probability_matrix = F.softmax(correlation_matrix / temperature, dim=-1)

    return probability_matrix


def edge_dropout(probability_matrix, edge_dropout_rate=0.5, epsilon=1e-10):
    """
    Apply edge dropout to the probability matrix. Some edges are randomly set to a very small value epsilon.

    Args:
        probability_matrix (torch.Tensor): Probability matrix of dimensions B x T x N x N.
        edge_dropout_rate (float): Edge dropout rate.

    Returns:
        torch.Tensor: Output tensor of dimensions B x T x N x N.
    """
    mask = (torch.rand_like(probability_matrix) > edge_dropout_rate).float()

    # mask is 1 when the edge is not dropped, 0 otherwise
    return probability_matrix * mask + epsilon * (1 - mask)

def get_local_affinity_matrices(input, temperature=1.0):
    """
    Compute local affinity matrices for each pair of consecutive frames in the input tensor.

    Args:
        input (torch.Tensor): Input tensor of dimensions B x T x N x D.
        temperature (float): Temperature parameter for softmax function.

    Returns:
        torch.Tensor: Output tensor of dimensions B x (T-1) x N x N.
    """
    return softmax_similarity_function_local(input, temperature)


def get_global_affinity_matrix(input, temperature=1.0, edge_dropout_rate=0.5):
    """
        Multiplies the local affinity matrices to get the global affinity matrix.
    """
    local_affinity_matrices = get_local_affinity_matrices(input, temperature)

    edge_dropped_local_affinity_matrices = edge_dropout(local_affinity_matrices, edge_dropout_rate)

    # Renormalize the edge-dropped local affinity matrices
    edge_dropped_local_affinity_matrices = edge_dropped_local_affinity_matrices / edge_dropped_local_affinity_matrices.sum(dim=-1, keepdim=True)

    # Assert that the sum of each row is equal to 1
    assert torch.isclose(edge_dropped_local_affinity_matrices.sum(dim=-1), torch.ones_like(edge_dropped_local_affinity_matrices.sum(dim=-1))).all()

    global_affinity_matrix = torch.prod(local_affinity_matrices, dim=1)

    return global_affinity_matrix
    


# Example usage:
input_tensor = torch.randn(16, 5, 10, 128) # (B, T, N, D)
output = softmax_similarity_function_local(input_tensor, temperature=1.0) # output shape: (B, T-1, N, N)

print(output.shape)

# assert the sum of each row is equal to 1
assert torch.isclose(output.sum(dim=-1), torch.ones_like(output.sum(dim=-1))).all()

# Example usage:
input_tensor = torch.randn(16, 5, 10, 128) # (B, T, N, D)
output = get_local_affinity_matrices(input_tensor, temperature=1.0) # output shape: (B, T-1, N, N)

print(output.shape)

# assert the sum of each row is equal to 1
assert torch.isclose(output.sum(dim=-1), torch.ones_like(output.sum(dim=-1))).all()

# Example usage:
input_tensor = torch.randn(16, 5, 10, 128) # (B, T, N, D)
output = get_global_affinity_matrix(input_tensor, temperature=1.0) # output shape: (B, N, N)

print(output.shape)

# asset the shape of the output tensor is correct
assert output.shape == (16, 10, 10)
