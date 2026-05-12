import numpy as np
from fem.common.utils import edge_length, tri_area, quad_area


class Boundary:
    def __init__(self, sim):
        self.sim = sim
        self.sim.boundNodes = []
    
    def apply_temp(self, nodes, temp):
        if np.isscalar(nodes):
            nodes = np.array([nodes])
        else:
            nodes = np.array(nodes).flatten()
    
        self.sim.T[nodes] = temp
        self.sim.boundNodes.extend(nodes.tolist())
    
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
    
    # Write function for generation in 3D

    def apply_conv0d(self, nodes, h, T_inf):
        nodes = np.atleast_1d(nodes)
        A = self.sim.cA

        for i, node in enumerate(nodes):
            k = h * A[i]
            f = h * A[i] * T_inf

            self.sim.K_sol[node, node] += k
            self.sim.Q_sol[node] += f
    
    def apply_conv1d(self, edges, h, T_inf):
        coords = self.sim.mesh.nodes[edges]

        if self.sim.mesh.type == "L2":
            A = self.sim.P * edge_length(coords)
        elif self.sim.mesh.type == "T3":
            print(self.sim.t[edges][:, 0])
            print(edge_length(coords))
            A = self.sim.t[edges][:, 0] * edge_length(coords)

        for i, edge in enumerate(edges):
            k = (h * A[i] / 6) * np.array([[2, 1],
                                           [1, 2]])
            f = (h * T_inf * A[i] / 2) * np.ones((2))

            self.sim.K_sol[np.ix_(edge, edge)] += k
            self.sim.Q_sol[edge] += f
    
    def apply_conv2d(self, faces, h, T_inf):
        coords = self.sim.mesh.nodes[faces]

        A = tri_area(coords)
        base_arr = np.array([[2, 1, 1],
                             [1, 2, 1],
                             [1, 1, 2]])

        for i, face in enumerate(faces):
            k = (h * A[i] / 12) * base_arr
            f = (h * T_inf * A[i] / 3) * np.ones((3))

            self.sim.K_sol[np.ix_(face, face)] += k
            self.sim.Q_sol[face] += f

if __name__ == "__main__":
    pass