import numpy as np
from fem.common.utils import unpack_dict, edge_length

def bound(sim, elementDict, edgeDict, nodeDict):
    sim.T = np.zeros(sim.mesh.N)
    sim.F = np.zeros(sim.mesh.N)
    sim.boundNodes = []

    sim.conv_1d = []
    sim.conv_2d = []

    ed = unpack_dict(elementDict)
    bd = unpack_dict(edgeDict)
    nd = unpack_dict(nodeDict)

    for element, type, value in zip(ed["element"], ed["type"], ed["value"]):
        if type == "gen":
            nodes = sim.mesh.elements[element]
            Qv = sim.mesh.A[element] * value * sim.t
            sim.F[nodes] += Qv / 3
        
        if type == "conv":
            h, T_inf = value
            sim.conv_2d.append((sim.mesh.elements[element], h, sim.mesh.A[element], T_inf))
    
    for edge, type, value in zip(bd["edge"], bd["type"], bd["value"]):
        if type == "conv":
            h, T_inf = value
            length = edge_length(sim.mesh.nodes[edge[0]], sim.mesh.nodes[edge[1]])
            A = length * sim.t
            sim.conv_1d.append((edge, h, A, T_inf))
        
        if type == "flux":
            sim.F[edge] = value
        
        if type == "temp":
            sim.T[edge] = value
            for node in edge:
                sim.boundNodes.append(node)

    sim.boundNodes = np.sort(np.unique(sim.boundNodes))