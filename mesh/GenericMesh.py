import numpy as np
from fem.common.utils import edge_length, tri_area, tet_volume

class GenericMesh:
    def __init__(self, nodes, elements, type):
        self.type = type

        # Generate number of nodes list
        self.N = len(nodes)

        # Generate node, edge and element lists
        self.nodes = np.array(nodes)
        self.elements = np.array(elements)

        # Generate coordinate lists
        self.x = self.nodes[:, 0]
        self.y = self.nodes[:, 1]
        self.z = self.nodes[:, 2]

        # Define characteristic unit lists
        if self.type == "L2":
            self.L = edge_length(self.nodes[self.elements])
        if self.type == "T3":
            self.A = tri_area(self.nodes[self.elements])
        if self.type == "TH4":
            self.V = tet_volume(self.nodes[self.elements])