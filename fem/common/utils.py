import numpy as np

def edge_length(nodes):
    """
    Compute length of edge connecting 2 nodes.

    Parameters:
        nodes: array of shape (..., 2, 3)
            nodes[...,0,:] = first node of edge
            nodes[...,1,:] = second node of edge

    Returns:
        lengths: array of shape (...)
    """

    A = nodes[..., 0, :]
    B = nodes[..., 1, :]

    diff = B - A

    length = np.linalg.norm(diff, axis=-1)

    return length

def tri_area(nodes):
    """
    Compute the area of a triangle defined by 3 points.

    Parameters:
        nodes: shape (..., 3, 3)
            For each triangle:
            nodes[i, j] = [x_j, y_j, z_j] for j = 0,1,2

    Returns:
        areas: shape (...,)
        Area of each triangle.
    """

    A = nodes[..., 0, :]
    B = nodes[..., 1, :]
    C = nodes[..., 2, :]

    AB = B - A
    AC = C - A

    cross = np.cross(AB, AC)

    area = 0.5 * np.linalg.norm(cross, axis=-1)

    return area

def quad_area(nodes):
    """
    Compute the area of a quadrilateral defined by 4 points.

    Parameters:
        nodes: shape (..., 3, 4)
            For each quad:
            nodes[i, j] = [x_j, y_j, z_j] for j = 0,1,2

    Returns:
        areas: shape (...,)
        Area
    """

    A = nodes[..., 0, :]
    B = nodes[..., 1, :]
    C = nodes[..., 2, :]
    D = nodes[..., 3, :]

    AC = C - A
    BD = D - B

    cross = np.cross(AC, BD)

    area = 0.5 * np.linalg.norm(cross, axis=-1)

    return area

def tet_volume(nodes):
    """
    Compute volume of a tetrahedron defined by 4 nodes.

    Parameters:
        nodes : array_like
            Shape (4,3) for one tet or (N,4,3) for many tets.
            nodes[:, :] gives coordinates of the 4 vertices.

    Returns:
        volume : float or ndarray
            Volume(s) of the tetrahedron(s).
    """
    nodes = np.asarray(nodes)

    if nodes.ndim == 2:
        nodes = nodes[None, ...]

    A = nodes[:, 0]
    B = nodes[:, 1]
    C = nodes[:, 2]
    D = nodes[:, 3]
    
    AB = B - A
    AC = C - A
    AD = D - A

    dets = np.sum(np.cross(AB, AC) * AD, axis=1)

    vol = np.abs(dets) / 6.0

    return vol

if __name__ == "__main__":
    nodes = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [2, 1, 0],
                       [0, 1, 0]])

    A = quad_area(nodes)
    print(A)