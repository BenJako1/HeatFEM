from fem.d2.solver import HeatSolver2D
from fem.d2.mesh import TriMesh2D
import numpy as np

sim = HeatSolver2D(k=0.2, t=1.25)

sim.mesh = TriMesh2D(120, 160, 100, 10)

wall = [[0, 3]]

free = [[2, 5]]

wall = [[0, 100],
        [100, 200],
        [200, 300],
        [300, 400],
        [400, 500],
        [500, 600],
        [600, 700],
        [700, 800],
        [800, 900]]

free = [[99, 199],
        [199, 299],
        [299, 399],
        [399, 499],
        [499, 599],
        [599, 699],
        [699, 799],
        [799, 899],
        [899, 999]]

elementDict = {
    "element": [f"0:{len(sim.mesh.elements)}"],
    "type": ["conv"],
    "value": [np.array([4e-4, 30])]
}

edgeDict = {
    "edge": [wall, free],
    "type": ["temp", "conv"],
    "value": [330, np.array([4e-4, 30])]
}

nodeDict = {
    "node": [],
    "type": [],
    "value": []
}

sim.apply_boundary_conditions(elementDict, edgeDict, nodeDict)
sim.assemble()
T, Q = sim.solve()

print(T)

from fem.d2.postprocess import plot_surface
plot_surface(sim.mesh, T)