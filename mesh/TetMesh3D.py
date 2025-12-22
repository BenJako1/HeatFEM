from .mesh_base import BaseMesh
import numpy as np
from fem.common.utils import tet_volume

class TetMesh3D(BaseMesh):
    def __init__(self, Lx, Ly, Lz, nx, ny, nz):
        """
        Generation of a box mesh with tetrahedral elements.

        Parameters:
            Lx, Ly, Lz : int, float
                Dimensions of box in x, y, and z directions.
            nx, ny, nz : int
                Number of nodes in each direction

        Returns:
            mesh : object
        """

        self.type = "TH4"

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