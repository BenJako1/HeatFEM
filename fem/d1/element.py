import numpy as np

def element_B_matrix(x, y, z):
    return 1

def get_element_geometry(sim, e):
    return {
            "A": sim.A,
            "L": sim.mesh.L[e]
            }

def elemental_conductance(B, k, geom):
    A = geom["A"]
    L = geom["L"]
    return k * A / L * np.array([[1, -1],
                                 [-1, 1]])