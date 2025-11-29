import numpy as np
from fem.common.solver_base import SolverBase
from .assembly import assemble_2d, assemble_nonphys_1d, assemble_nonphys_2d
from .boundary import bound

class HeatSolver2D(SolverBase):
    def __init__(self, k, t):
        self.k = k
        self.t = t

    def apply_boundary_conditions(self, elementDict, edgeDict, nodeDict):
        bound(self, elementDict, edgeDict, nodeDict)

    def assemble(self):
        self.K, self.Q = assemble_2d(self.mesh, self.k, self.t, self.F)

        self.K_sol = np.copy(self.K)
        self.Q_sol = np.copy(self.Q)
        assemble_nonphys_1d(self.K_sol,
                            self.Q_sol,
                            convBC=self.conv_1d)
        assemble_nonphys_2d(self.K_sol,
                            self.Q_sol,
                            convBC=self.conv_2d)

    def solve(self):
        return SolverBase.solve(self)