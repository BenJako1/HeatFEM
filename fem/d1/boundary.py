import numpy as np
from fem.common.utils import unpack_dict

def bound(sim, elementDict, nodeDict):
    sim.T = np.zeros(sim.mesh.N)
    sim.F = np.zeros(sim.mesh.N)
    sim.boundNodes = []

    sim.conv_0d = []
    sim.conv_1d = []

    ed = unpack_dict(elementDict)
    nd = unpack_dict(nodeDict)

    for element, typ, value in zip(ed["element"], ed["type"], ed["value"]):
        if typ == "gen":
            n1, n2 = sim.mesh.elements[element]
            Qv = sim.A * value * sim.mesh.element_len[element]
            sim.F[n1] += Qv / 2
            sim.F[n2] += Qv / 2

        elif typ == "conv":
            h, Wc, T_inf = value
            L = sim.mesh.element_len[element]
            area = Wc * L
            sim.conv_1d.append((sim.mesh.elements[element], h, area, T_inf))
    
    for node, typ, value in zip(nd["node"], nd["type"], nd["value"]):
        if typ == "temp":
            sim.T[node] = value
            sim.boundNodes.append(node)
        
        elif typ == "flux":
            sim.F[node] += value
        
        elif typ == "conv":
            h, T_inf = value
            area = sim.A
            sim.conv_0d.append((node, h, area, T_inf))
            
    sim.boundNodes = np.sort(np.unique(sim.boundNodes))