from fem.common.solver_base import SolverBase
from fem.common.boundary import Boundary
from .assembly import assemble_3d
import numpy as np

class HeatSolver3D(SolverBase):
    def __init__(self, mesh, k):
        self.k = k
        self.mesh = mesh
        self.boundary = Boundary(self)
    
    def assemble(self):
        self.K, self.Q = assemble_3d(self.mesh, self.k)
        self.T = np.zeros(self.mesh.N)

        self.K_sol = np.copy(self.K)
        self.Q_sol = np.copy(self.Q)

    def solve(self):
        return SolverBase.solve(self)