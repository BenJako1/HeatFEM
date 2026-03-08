from ..common.solver_base import SolverBase
from ..boundary import Boundary
from ..assembly import assemble
from ..property import Property

class steadySolver(SolverBase):
    def __init__(self, mesh, **params):
        self.mesh = mesh
        self.property = Property(self)
        self.boundary = Boundary(self)

        #self.__dict__.update(params)

        if self.mesh.type == "L2":
            from fem.element.L2 import L2 as element
        elif self.mesh.type == "T3":
            from fem.element.T3 import T3 as element
        #elif self.mesh.type == "Q4":
            #from fem.element.Q4 import Q4 as element
        elif self.mesh.type == "TH4":
            from fem.element.TH4 import TH4 as element
        else:
            raise ValueError("Invalid element type.")

        self.element = element()

    def assemble(self):
        self.K, self.Q, self.T, self.K_sol, self.Q_sol = assemble(self)

    def solve(self):
        return SolverBase.solve(self)