import numpy as np

class TriMesh2D:
    def __init__(self, L, H, nx, ny):
        self.N = nx*ny
        # Generate nodes
        self.nodes = np.zeros((self.N, 2))
        self.x = np.zeros(self.N)
        self.y = np.zeros(self.N)
        for j in range(ny):
            for i in range(nx):
                idx = j*nx + i
                x = L * i / (nx - 1)
                y = H * j / (ny - 1)
                self.nodes[idx] = [x, y]
                self.x[idx] = x
                self.y[idx] = y

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
        
        tris = self.nodes[self.elements]
        x1, y1 = tris[:,0,0], tris[:,0,1]
        x2, y2 = tris[:,1,0], tris[:,1,1]
        x3, y3 = tris[:,2,0], tris[:,2,1]

        self.A = 0.5 * np.abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

class gmsh2D:
    def __init__(self, nodes, elements):
        self.N = len(nodes)
        self.nodes = nodes
        self.x = nodes[:,0]
        self.y = nodes[:,1]
        self.elements = elements

        tris = self.nodes[self.elements]
        x1, y1 = tris[:,0,0], tris[:,0,1]
        x2, y2 = tris[:,1,0], tris[:,1,1]
        x3, y3 = tris[:,2,0], tris[:,2,1]

        self.A = 0.5 * np.abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))