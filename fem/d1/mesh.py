import numpy as np

class LineMesh1D:
    def __init__(self, L, N):
        self.L = L
        self.N = N

        # Generate nodal indices and x-coordinates lists
        self.nodes = np.arange(N)
        self.x = np.linspace(0, L, N)

        # Generate element node lists
        self.elements = np.zeros([N-1, 2], dtype=int)
        for i in range(len(self.elements)):
            self.elements[i] = [i, i+1]

        # Calculate element lengths
        self.element_len = np.diff(self.x)