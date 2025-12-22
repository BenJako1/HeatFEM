import numpy as np
from fem.common.utils import edge_length

class L2:
    def B_matrix(self, coords):
        L = edge_length(coords)
        B = 1 / L * np.array([[-1, 1]])
        return B

    def get_K(self, sim, coords, e):
        k = sim.k[e]
        A = sim.A[e]
        L = sim.mesh.L[e]
        B = self.B_matrix(coords)
        return k * A * L * (B.T @ B)
