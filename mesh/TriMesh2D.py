import numpy as np
from fem.common.utils import tri_area

class TriMesh2D:
    def __init__(self, Lx, Ly, nx, ny):
        self.type = "2D"

        self.N = nx*ny

        self.x = np.zeros(self.N)
        self.y = np.zeros(self.N)
        self.z = np.zeros(self.N)

        for j in range(ny):
            for i in range(nx):
                idx = j*nx + i
                x = Lx * i / (nx - 1)
                y = Ly * j / (ny - 1)
                self.x[idx] = x
                self.y[idx] = y

        self.nodes = np.vstack((self.x, self.y, self.z)).T

        # Generate elements from quads
        self.elements = np.zeros(((nx - 1)*(ny - 1)*2, 3), dtype=int)

        k = 0
        for j in range(ny - 1):
            for i in range(nx - 1):

                # local quad nodes
                A = j*nx + i
                B = A + 1
                D = A + nx
                C = D + 1     # (j+1)*nx + (i+1)

                # Pattern 2 (upper-left → lower-right diagonal)
                # tri1 = (A, B, C)
                # tri2 = (A, C, D)

                self.elements[k, :] = [A, B, C]
                k += 1
                self.elements[k, :] = [A, D, C] #Altered from ACD for testing
                k += 1
        
        self.A = tri_area(self.nodes[self.elements])
    
    def boundaryEdge(self, x_in=None, y_in=None, z_in=None, tol=1e-8):
        """
        Vectorised detection of triangle edges lying on a given axis.
        Only edges with both nodes on the axis are returned.

        Parameters
        ----------
        mesh : object
            Must have mesh.elements (Ne x 4) and mesh.x, mesh.y, mesh.z arrays.
        x_in, y_in, z_in : float or None
            Axis coordinates.
        tol : float
            Tolerance for floating point comparison.

        Returns
        -------
        boundary_faces : np.ndarray
            Array of shape (Nfaces, 3) containing node indices of boundary faces.
        """

        # 3 edges per triangle
        tri_edges = np.array([
            [0, 1],
            [0, 2],
            [1, 2]
        ])

        # Expand element nodes to faces
        # shape (Ne, 4 faces, 3 nodes per face)
        all_edges = self.elements[:, tri_edges]

        # Flatten to (Ne*4, 3)
        flat_edges = all_edges.reshape(-1, 2)

        # Gather coordinates of each face
        x = self.x[flat_edges]  # shape (Ne*4, 3)
        y = self.y[flat_edges]
        z = self.z[flat_edges]

        # Start with all True mask
        mask = np.ones(len(flat_edges), dtype=bool)

        if x_in is not None:
            mask &= np.all(np.abs(x - x_in) < tol, axis=1)
        if y_in is not None:
            mask &= np.all(np.abs(y - y_in) < tol, axis=1)
        if z_in is not None:
            mask &= np.all(np.abs(z - z_in) < tol, axis=1)

        # Select faces that satisfy plane condition
        boundary_faces = flat_edges[mask]

        return boundary_faces