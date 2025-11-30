import numpy as np
from .utils import edge_length, tri_area

class Boundary:
    def __init__(self, sim):
        self.sim = sim
        self.sim.boundNodes = []
    
    def apply_temp0d(self, nodes, temp):
        if np.isscalar(nodes):
            nodes = np.array([nodes])
        else:
            nodes = np.array(nodes).flatten()
    
        self.sim.T[nodes] = temp
        self.sim.boundNodes.extend(nodes.tolist())
    
    def apply_temp1d(self, edges, temp):
        nodes = edges.flatten()
        self.apply_temp0d(nodes, temp)

    def apply_temp2d(self, faces, temp):
        nodes = faces.flatten()
        self.apply_temp0d(nodes, temp)
    
    def apply_gen1d(self, edges, A, Q_gen):
        coords = self.sim.mesh.nodes[edges]

        V = A * edge_length(coords)

        for i, edge in enumerate(edges):
            f = V[i] * Q_gen / 2 * np.ones((2))
        
            self.sim.Q_sol[edge] += f
    
    def apply_gen2d(self, faces, t, Q_gen):
        coords = self.sim.mesh.nodes[faces]

        V = t * tri_area(coords)

        for i, face in enumerate(faces):
            f = V[i] * Q_gen / 3 * np.ones((3))

            self.sim.Q_sol[face] += f

    def apply_conv0d(self, nodes, h, A, T_inf):
        k = h * A
        f = h * A * T_inf

        self.sim.K_sol[nodes, nodes] += k
        self.sim.Q_sol[nodes] += f
    
    def apply_conv1d(self, edges, h, t, T_inf):
        coords = self.sim.mesh.nodes[edges]

        A = t * edge_length(coords)

        for i, edge in enumerate(edges):
            k = (h * A[i] / 6) * np.array([[2, 1],
                                           [1, 2]])
            f = (h * T_inf * A[i] / 2) * np.ones((2))

            self.sim.K_sol[np.ix_(edge, edge)] += k
            self.sim.Q_sol[edge] += f
    
    def apply_conv2d(self, faces, h, T_inf):
        coords = self.sim.mesh.nodes[faces]

        A = tri_area(coords)

        for i, face in enumerate(faces):
            k = (h * A[i] / 12) * np.array([[2, 1, 1],
                                            [1, 2, 1],
                                            [1, 1, 2]])
            f = (h * T_inf * A[i] / 3) * np.ones((3))

            self.sim.K_sol[np.ix_(face, face)] += k
            self.sim.Q_sol[face] += f