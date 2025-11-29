import numpy as np
from .element import elemental_conductance
from .element import *

def assemble_1d(mesh, k, A, F_known=None):
    K = np.zeros((mesh.N, mesh.N))
    F = np.zeros(mesh.N)

    if F_known is not None:
        F = F_known

    for e, nodes in enumerate(mesh.elements):
        Le = mesh.element_len[e]

        Ke = elemental_conductance(k, A, Le)
        K[np.ix_(nodes, nodes)] += Ke
    
    return K, F

def assemble_nonphys_0d(K, F, convBC=None):
    if convBC is None:
        convBC = {}
    
    for nodes, h, area, T_inf in convBC:
        K_conv = convection_stiffness_0d(h, area)
        F_conv = convection_load_0d(h, area, T_inf)
        
        K[nodes,nodes] += K_conv
        F[nodes] += F_conv
    
    return K_conv, F_conv

def assemble_nonphys_1d(K, F, convBC=None):
    if convBC is None:
        convBC = {}
    
    for nodes, h, area, T_inf in convBC:
        K_conv = convection_stiffness_1d(h, area)
        F_conv = convection_load_1d(h, area, T_inf)
        
        K[np.ix_(nodes, nodes)] += K_conv
        F[nodes] += F_conv
    
    return K_conv, F_conv