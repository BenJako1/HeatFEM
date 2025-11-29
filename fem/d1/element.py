import numpy as np

def elemental_conductance(k, A, Le):
    return k * A / Le * np.array([[1, -1], [-1, 1]])

def convection_stiffness_0d(h, A):
    return h * A

def convection_load_0d(h, A, Tinf):
    return h * Tinf * A 

def convection_stiffness_1d(h, Le):
    return h * Le / 6 * np.array([[2, 1],
                                  [1, 2]])

def convection_load_1d(h, A, Tinf):
    return h * Tinf * A / 2 * np.array([1, 1])