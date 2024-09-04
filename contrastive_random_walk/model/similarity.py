import torch

import torch.nn.functional as F

import numpy as np


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
    correlation_matrix = torch.einsum("btnd,btmd->btnm", input[:, :-1], input[:, 1:])

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

    edge_dropped_local_affinity_matrices = edge_dropout(
        local_affinity_matrices, edge_dropout_rate
    )

    # Renormalize the edge-dropped local affinity matrices
    edge_dropped_local_affinity_matrices = (
        edge_dropped_local_affinity_matrices
        / edge_dropped_local_affinity_matrices.sum(dim=-1, keepdim=True)
    )

    # Assert that the sum of each row is equal to 1
    # assert torch.isclose(
    #     edge_dropped_local_affinity_matrices.sum(dim=-1),
    #     torch.ones_like(edge_dropped_local_affinity_matrices.sum(dim=-1)),
    # ).all()

    try:
        assert torch.isclose(
            edge_dropped_local_affinity_matrices.sum(dim=-1),
            torch.ones_like(edge_dropped_local_affinity_matrices.sum(dim=-1)),
        ).all()
    except AssertionError as e:
        print(edge_dropped_local_affinity_matrices.sum(dim=-1))
        print(np.any(edge_dropped_local_affinity_matrices.sum(dim=-1) < 0))
        print(e.__traceback__)
        raise AssertionError

    global_affinity_matrix = torch.prod(edge_dropped_local_affinity_matrices, dim=1)

    return global_affinity_matrix


def get_affinity_matrices(input, temperature=1.0, edge_dropout_rate=0.5):
    """
    Multiplies the local affinity matrices to get the global affinity matrix.
    """
    local_affinity_matrices = get_local_affinity_matrices(input, temperature)

    edge_dropped_local_affinity_matrices = edge_dropout(
        local_affinity_matrices, edge_dropout_rate
    )

    # Renormalize the edge-dropped local affinity matrices
    edge_dropped_local_affinity_matrices = (
        edge_dropped_local_affinity_matrices
        / edge_dropped_local_affinity_matrices.sum(dim=-1, keepdim=True)
    )

    # Assert that the sum of each row is equal to 1
    try:
        assert torch.isclose(
            edge_dropped_local_affinity_matrices.sum(dim=-1),
            torch.ones_like(edge_dropped_local_affinity_matrices.sum(dim=-1)),
        ).all()
    except AssertionError as e:
        print(edge_dropped_local_affinity_matrices.sum(dim=-1))
        print(np.any(edge_dropped_local_affinity_matrices.sum(dim=-1) < 0))
        print(e.__traceback__)
        raise AssertionError

    global_affinity_matrix = torch.prod(edge_dropped_local_affinity_matrices, dim=1)

    return global_affinity_matrix, local_affinity_matrices, edge_dropped_local_affinity_matrices


def get_affinity_matrices_all_walks(input, temperature=1.0, edge_dropout_rate=0.5):
    """
    Multiplies the local affinity matrices to get the global affinity matrix.
    """
    local_affinity_matrices = get_local_affinity_matrices(input, temperature)

    edge_dropped_local_affinity_matrices = edge_dropout(
        local_affinity_matrices, edge_dropout_rate
    )

    # edge_dropped_local_affinity_matrices shape: (B, T-1, N, N)
    

    # Renormalize the edge-dropped local affinity matrices
    edge_dropped_local_affinity_matrices = (
        edge_dropped_local_affinity_matrices
        / edge_dropped_local_affinity_matrices.sum(dim=-1, keepdim=True)
    )

    # Assert that the sum of each row is equal to 1
    # assert torch.isclose(
    #     edge_dropped_local_affinity_matrices.sum(dim=-1),
    #     torch.ones_like(edge_dropped_local_affinity_matrices.sum(dim=-1)),
    # ).all()

    try:
        assert torch.isclose(
            edge_dropped_local_affinity_matrices.sum(dim=-1),
            torch.ones_like(edge_dropped_local_affinity_matrices.sum(dim=-1)),
        ).all()

    except AssertionError as e:
        print(edge_dropped_local_affinity_matrices.sum(dim=-1))
        print(np.any(edge_dropped_local_affinity_matrices.sum(dim=-1) < 0))
        print(e.__traceback__)
        raise AssertionError
    


    # Extract global affinity matrix for each walk. Walks are defined as the palindromic sequence of frames starting from frame 0.
    # Walk 1: 0-1-0
    # Walk 2: 0-1-2-1-0
    # Walk 3: 0-1-2-3-2-1-0
    #...
    # Walk T: 0-1-2-3-...-T-...-3-2-1-0

    global_affinity_matrix_all_walks_dict = dict()
    #print("Going inside affinity matrices multiplcation loop")
    for walk_len in range(2, input.size(1)+1):
        # walk_front contains the product of the local affinity matrices from frame at idx 0 to frame at idx walk_len-1
        walk_front = torch.prod(edge_dropped_local_affinity_matrices[:, :walk_len], dim=1)

        # walk_back contains the product of the local affinity matrices from frame at idx walk_len-2 to frame at idx 0
        walk_back = torch.prod(torch.flip(edge_dropped_local_affinity_matrices[:, :walk_len], dims=[1]), dim=1)

        walk_palindrome =  walk_palindrome = torch.einsum('bij,bjk->bik', walk_front, walk_back)

        # walk_palindrome shape: (B, N, N)
        assert walk_palindrome.shape == (input.size(0), input.size(2), input.size(2))


        global_affinity_matrix_all_walks_dict[walk_len] = walk_palindrome

    #("Multiplcation loop done")
    return global_affinity_matrix_all_walks_dict, local_affinity_matrices, edge_dropped_local_affinity_matrices