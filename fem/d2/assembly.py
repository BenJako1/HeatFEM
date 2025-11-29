import numpy as np
from .element import *

def assemble_2d(mesh, k, t, F_known=None):
    K = np.zeros((mesh.N, mesh.N))
    F = np.zeros(mesh.N)

    if F_known is not None:
        F = F_known

    for e, nodes in enumerate(mesh.elements):
        x = mesh.x[nodes]
        y = mesh.y[nodes]
        B = element_B_matrix(x, y, mesh.A[e])
        Ke = elemental_conductance(k, t, mesh.A[e], B)

        K[np.ix_(nodes, nodes)] += Ke
    
    return K, F

def assemble_nonphys_1d(K, F, convBC=None):
    if convBC is None:
        convBC = {}

    for nodes, h, area, T_inf in convBC:
        K_conv = convection_stiffness_1d(h, area)
        F_conv = convection_load_1d(h, area, T_inf)
        
        K[np.ix_(nodes, nodes)] += K_conv
        F[nodes] += F_conv
    
    return K, F

def assemble_nonphys_2d(K, F, convBC=None):

    # If there is not convection, create empty set
    if convBC is None:
        convBC = {}

    for nodes, h, area, T_inf in convBC:
        K_conv = convection_stiffness_2d(h, area)
        F_conv = convection_load_2d(h, area, T_inf)
        
        K[np.ix_(nodes, nodes)] += K_conv
        F[nodes] += F_conv
    
    return K, F