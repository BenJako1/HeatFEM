import numpy as np

def assemble(sim):
    K = np.zeros((sim.mesh.N, sim.mesh.N))
    Q = np.zeros(sim.mesh.N)
    T = np.zeros(sim.mesh.N)

    element = sim.element
    
    for e, nodes in enumerate(sim.mesh.elements):
        Ke = element.get_K(sim, sim.mesh.nodes[nodes], e)

        K[np.ix_(nodes, nodes)] += Ke

    return K, Q, T, K.copy(), Q.copy()