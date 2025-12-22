import numpy as np

class TH4:
    def B_matrix(self, coords):
        A = np.concatenate((np.ones((4,1)), coords), axis=1)
        Ainv = np.linalg.inv(A)
        B = Ainv[..., 1:, :]
        return B

    def get_K(self, sim, coords, e):
        k = sim.k[e]
        V = sim.mesh.V[e]
        B = self.B_matrix(coords)
        return k * V * (B.T @ B)
    
if __name__ == "__main__":
    nodes = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    e = TH4()
    e.B_matrix(nodes)