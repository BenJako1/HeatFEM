import numpy as np
from fem.common.utils import edge_length, tri_area, tet_volume

class GenericMesh:
    def __init__(self, nodes, elements, type):
        self.type = type

        self.N = len(nodes)

        self.nodes = np.array(nodes)
        self.elements = np.array(elements)

        self.x = self.nodes[:][0]
        self.y = self.nodes[:][1]
        self.z = self.nodes[:][2]

        if self.type == "1D":
            self.L = edge_length(self.nodes[self.elements])
        if self.type == "2D":
            self.A = tri_area(self.nodes[self.elements])
        if self.type == "3D":
            self.V = tet_volume(self.nodes[self.elements])