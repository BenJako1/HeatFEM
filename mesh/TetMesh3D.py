import numpy as np
from fem.common.utils import tet_volume

class TetMesh3D:
    def __init__(self, Lx, Ly, Lz, nx, ny, nz):
        """
        Generation of a box mesh with tetrahedral elements.

        Parameters
        ----------
        Lx, Ly,, Lz : int, float
            Dimensions of box in x, y, and z directions.
        nx, ny, nz : int
            Number of nodes in each direction

        Returns
        -------
        mesh : object
        """

        self.type = "3D"

        self.N = nx*ny*nz

        self.x = np.zeros(self.N)
        self.y = np.zeros(self.N)
        self.z = np.zeros(self.N)

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    idx = k*(nx*ny) + j*nx + i
                    x = Lx * i / (nx - 1)
                    y = Ly * j / (ny - 1)
                    z = Lz * k / (nz - 1)
                    self.x[idx] = x
                    self.y[idx] = y
                    self.z[idx] = z
        
        self.nodes = np.vstack((self.x, self.y, self.z)).T

        # Generate elements from quads
        self.elements = np.zeros(((nx - 1)*(ny - 1)*(nz - 1)*5, 4), dtype=int)

        count = 0
        for k in range(nz - 1):
            for j in range(ny - 1):
                for i in range(nx - 1):

                    # local quad nodes
                    A = k*(nx*ny) + j*nx + i
                    B = A + 1
                    D = A + nx
                    C = D + 1     # (j+1)*nx + (i+1)
                    E = A + (nx*ny)
                    F = E + 1
                    H = E + nx
                    G = H + 1

                    self.elements[count, :] = [A, B, D, E]
                    count += 1
                    self.elements[count, :] = [B, E, F, G]
                    count += 1
                    self.elements[count, :] = [B, C, D, G]
                    count += 1
                    self.elements[count, :] = [D, E, G, H]
                    count += 1
                    self.elements[count, :] = [B, D, E, G]
                    count += 1
        
        self.V = tet_volume(self.nodes[self.elements])
    
    def boundarySurface(self, x_in=None, y_in=None, z_in=None, tol=1e-8):
        """
        Vectorised detection of tetrahedron faces lying in a given plane.
        Only faces with all 3 nodes in the plane are returned.

        Parameters
        ----------
        mesh : object
            Must have mesh.elements (Ne x 4) and mesh.x, mesh.y, mesh.z arrays.
        x_in, y_in, z_in : float or None
            Plane coordinates.
        tol : float
            Tolerance for floating point comparison.

        Returns
        -------
        boundary_faces : np.ndarray
            Array of shape (Nfaces, 3) containing node indices of boundary faces.
        """

        # 4 faces per tetrahedron
        tet_faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ])

        # Expand element nodes to faces
        # shape (Ne, 4 faces, 3 nodes per face)
        all_faces = self.elements[:, tet_faces]

        # Flatten to (Ne*4, 3)
        flat_faces = all_faces.reshape(-1, 3)

        # Gather coordinates of each face
        x = self.x[flat_faces]  # shape (Ne*4, 3)
        y = self.y[flat_faces]
        z = self.z[flat_faces]

        # Start with all True mask
        mask = np.ones(len(flat_faces), dtype=bool)

        if x_in is not None:
            mask &= np.all(np.abs(x - x_in) < tol, axis=1)
        if y_in is not None:
            mask &= np.all(np.abs(y - y_in) < tol, axis=1)
        if z_in is not None:
            mask &= np.all(np.abs(z - z_in) < tol, axis=1)

        # Select faces that satisfy plane condition
        boundary_faces = flat_faces[mask]

        return boundary_faces