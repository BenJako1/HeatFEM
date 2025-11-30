import numpy as np

def edge_length(nodes):
    """
    nodes: array of shape (..., 2, 3)
       nodes[...,0,:] = first node of edge
       nodes[...,1,:] = second node of edge

    Returns:
        lengths: array of shape (...)
    """

    p0 = nodes[..., 0, :]
    p1 = nodes[..., 1, :]

    diff = p1 - p0

    return np.linalg.norm(diff, axis=-1)

def tri_area(nodes):
    """
    nodes: shape (..., 3, 3)
        For each triangle:
        nodes[i, j] = [x_j, y_j, z_j] for j = 0,1,2

    Returns:
        areas: shape (...,)
        Area of each triangle.
    """

    p0 = nodes[..., 0, :]
    p1 = nodes[..., 1, :]
    p2 = nodes[..., 2, :]

    # Edge vectors
    v1 = p1 - p0
    v2 = p2 - p0

    # Cross product of edges
    cross_prod = np.cross(v1, v2)

    # Triangle area = 0.5 * norm of cross product
    area = 0.5 * np.linalg.norm(cross_prod, axis=-1)

    return area