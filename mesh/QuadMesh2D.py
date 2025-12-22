from .mesh_base import BaseMesh
import numpy as np
from fem.common.utils import quad_area

class QuadMesh2D(BaseMesh):
    def __init__(self, Lx, Ly, nx, ny):
        """
        Generation of a rectangular mesh in z=0 with quadrilateral elements.

        Parameters:
            Lx, Ly : int, float
                Dimensions of domain in x and y directions.
            nx, ny : int
                Number of nodes in each direction.

        Returns:
            mesh : object
        """

        self.type = "Q4"

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
        self.elements = np.zeros(((nx - 1)*(ny - 1), 4), dtype=int)

        k = 0
        for j in range(ny - 1):
            for i in range(nx - 1):

                # local quad nodes
                A = j*nx + i
                B = A + 1
                D = A + nx
                C = D + 1

                self.elements[k, :] = [A, B, C, D]
                k += 1
        
        self.A = quad_area(self.nodes[self.elements])