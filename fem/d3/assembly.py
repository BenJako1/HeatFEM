import numpy as np
from .element import element_B_matrix, elemental_conductance

def assemble_3d(mesh, k):
    K = np.zeros((mesh.N, mesh.N))
    F = np.zeros(mesh.N)

    for e, nodes in enumerate(mesh.elements):
        x = mesh.x[nodes]
        y = mesh.y[nodes]
        z = mesh.z[nodes]
        B = element_B_matrix(x, y, z)
        Ke = elemental_conductance(k, mesh.V[e], B)

        K[np.ix_(nodes, nodes)] += Ke
    
    return K, F