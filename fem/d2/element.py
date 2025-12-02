import numpy as np
from fem.common.utils import tri_area

def element_B_matrix(x, y, z):
    A = tri_area(np.column_stack((x, y, z)))
    return 1/(2*A)*np.array([[y[1]-y[2], y[2]-y[0], y[0]-y[1]],
                             [x[2]-x[1], x[0]-x[2], x[1]-x[0]]])

def get_element_geometry(sim, e):
    return {
            "k": sim.k[e],
            "t": sim.t[e],
            "A": sim.mesh.A[e]
            }

def elemental_conductance(B, geom):
    k = geom["k"]
    t = geom["t"]
    A = geom["A"]
    return k * t * A * (B.T @ B)