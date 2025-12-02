import numpy as np

def assemble(sim):
    K = np.zeros((sim.mesh.N, sim.mesh.N))
    Q = np.zeros(sim.mesh.N)
    T = np.zeros(sim.mesh.N)

    element = sim.element

    for e, nodes in enumerate(sim.mesh.elements):
        x = sim.mesh.x[nodes]
        y = sim.mesh.y[nodes]
        z = sim.mesh.z[nodes]

        B = element.element_B_matrix(x, y, z)
        geom = element.get_element_geometry(sim, e)
        Ke = element.elemental_conductance(B, geom)

        K[np.ix_(nodes, nodes)] += Ke

    return K, Q, T, K.copy(), Q.copy()