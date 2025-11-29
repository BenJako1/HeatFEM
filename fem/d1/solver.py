import numpy as np
from fem.common.solver_base import SolverBase
from .assembly import assemble_1d, assemble_nonphys_0d, assemble_nonphys_1d
from .mesh import LineMesh1D
from .boundary import bound

class HeatSolver1D(SolverBase):
    def __init__(self, k, A):
        self.k = k
        self.A = A
    
    def apply_boundary_conditions(self, elementDict, nodeDict):
        bound(self, elementDict, nodeDict)
    
    def assemble(self):
        self.K, self.Q = assemble_1d(self.mesh, self.k, self.A, self.F)
        
        self.K_sol = np.copy(self.K)
        self.Q_sol = np.copy(self.Q)
        assemble_nonphys_0d(self.K_sol,
                            self.Q_sol,
                            self.conv_0d)
        assemble_nonphys_1d(self.K_sol,
                            self.Q_sol,
                            self.conv_1d)

    def solve(self):
        return SolverBase.solve(self)