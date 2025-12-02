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

def tet_volume(nodes):
    """
    Compute tetrahedron volumes (vectorised).

    Parameters
    ----------
    nodes : array_like
        Shape (4,3) for one tet or (N,4,3) for many tets.
        nodes[:, :] gives coordinates of the 4 vertices.

    Returns
    -------
    volume : float or ndarray
        Volume(s) of the tetrahedron(s).
    """
    nodes = np.asarray(nodes)

    # Ensure array shape is (N,4,3)
    if nodes.ndim == 2:
        nodes = nodes[None, ...]   # convert (4,3) → (1,4,3)

    # Compute the edge vectors for all elements
    v1 = nodes[:, 1] - nodes[:, 0]
    v2 = nodes[:, 2] - nodes[:, 0]
    v3 = nodes[:, 3] - nodes[:, 0]

    # Compute determinant for each tetrahedron
    dets = np.einsum('ij,ij->i', np.cross(v1, v2), v3)

    # Volume = |det| / 6
    vol = np.abs(dets) / 6.0

    return vol if vol.size > 1 else vol[0]