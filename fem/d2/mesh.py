import numpy as np

class TriMesh2D:
    def __init__(self, L, H, nx, ny):
        self.N = nx*ny

        self.x = np.zeros(self.N)
        self.y = np.zeros(self.N)
        self.z = np.zeros(self.N)

        for j in range(ny):
            for i in range(nx):
                idx = j*nx + i
                x = L * i / (nx - 1)
                y = H * j / (ny - 1)
                self.x[idx] = x
                self.y[idx] = y

        self.nodes = np.vstack((self.x, self.y, self.z)).T

        # Generate elements from quads
        self.elements = np.zeros(((nx - 1)*(ny - 1)*2, 3), dtype=int)

        k = 0
        for j in range(ny - 1):
            for i in range(nx - 1):

                # local quad nodes
                A = j*nx + i
                B = A + 1
                D = A + nx
                C = D + 1     # (j+1)*nx + (i+1)

                # Pattern 2 (upper-left → lower-right diagonal)
                # tri1 = (A, B, C)
                # tri2 = (A, C, D)

                self.elements[k, :] = [A, B, C]
                k += 1
                self.elements[k, :] = [A, D, C] #Altered from ACD for testing
                k += 1
        
        tris_x, tris_y = self.x[self.elements], self.y[self.elements]
        x1, y1 = tris_x[:,0], tris_y[:,0]
        x2, y2 = tris_x[:,1], tris_y[:,1]
        x3, y3 = tris_x[:,2], tris_y[:,2]

        self.A = 0.5 * np.abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

class gmsh2D:
    def __init__(self, nodes, elements):
        self.N = len(nodes)
        self.x = nodes[:,0]
        self.y = nodes[:,1]
        self.nodes = nodes
        self.elements = elements

        tris_x, tris_y = self.x[self.elements], self.y[self.elements]
        x1, y1 = tris_x[:,0], tris_y[:,0]
        x2, y2 = tris_x[:,1], tris_y[:,1]
        x3, y3 = tris_x[:,2], tris_y[:,2]

        self.A = 0.5 * np.abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))