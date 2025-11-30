import numpy as np

class LineMesh1D:
    def __init__(self, L, N):
        self.type = "1D"

        self.L = L
        self.N = N

        # Generate nodal indices and x-coordinates lists
        self.x = np.linspace(0, L, N)
        self.y = np.zeros(N)
        self.z = np.zeros(N)

        self.nodes = np.vstack((self.x, self.y, self.z)).T

        # Generate element node lists
        self.elements = np.zeros([N-1, 2], dtype=int)
        for i in range(len(self.elements)):
            self.elements[i] = [i, i+1]

        # Calculate element lengths
        self.L = np.diff(self.x)