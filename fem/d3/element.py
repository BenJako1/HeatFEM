import numpy as np

def element_B_matrix(x, y, z):
    A = np.stack([
        np.ones_like(x),
        x, y, z
    ], axis=-1)

    # Invert A (vectorised)
    Ainv = np.linalg.inv(A)

    # B consists of rows 1:, i.e. β_i, γ_i, δ_i
    # Ainv shape = (...,4,4)
    # We need rows 1..3 (i.e. columns of Ainv correspond to shape functions)
    grads = Ainv[..., 1:, :]  # shape (...,3,4)

    # This already gives the B matrix
    return grads

def get_element_geometry(sim, e):
    return {
            "k": sim.k[e],
            "V": sim.mesh.V[e]
           }

def elemental_conductance(B, geom):
    k = geom["k"]
    V = geom["V"]
    return k * V * (B.T @ B)