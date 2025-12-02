from .common.solver_base import SolverBase
from .boundary import Boundary
from .assemble import assemble
from .property import Property

class HeatSolver(SolverBase):
    def __init__(self, mesh, **params):
        self.mesh = mesh
        self.property = Property(self)
        self.boundary = Boundary(self)

        self.__dict__.update(params)

        if self.mesh.type == "1D":
            from fem.d1 import element
        elif self.mesh.type == "2D":
            from fem.d2 import element
        elif self.mesh.type == "3D":
            from fem.d3 import element
        else:
            raise ValueError("Invalid element type.")

        self.element = element
    
    def assemble(self):
        self.K, self.Q, self.T, self.K_sol, self.Q_sol = assemble(self)

    def solve(self):
        return SolverBase.solve(self)