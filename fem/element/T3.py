import numpy as np
from fem.common.utils import tri_area

class T3:
    def B_matrix(self, coords):
        A = tri_area(coords)
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        return 1/(2*A)*np.array([[y[1]-y[2], y[2]-y[0], y[0]-y[1]],
                                   [x[2]-x[1], x[0]-x[2], x[1]-x[0]]])

    def get_K(self, sim, coords, e):
        k = sim.k[e]
        A = sim.mesh.A[e]
        t = sim.t[e]
        B = self.B_matrix(coords)
        return k * t * A * (B.T @ B)