from fem.d2.solver import HeatSolver2D
from fem.d2.mesh import gmsh2D
import numpy as np
import pygmsh

sim = HeatSolver2D(k=600, t=0.1)

ms = 0.2

with pygmsh.geo.Geometry() as geom:
    p1 = geom.add_point([0, 0, 0], mesh_size = ms)
    p2 = geom.add_point([2, 0, 0], mesh_size = ms)
    p3 = geom.add_point([2, 1, 0], mesh_size = ms)
    p4 = geom.add_point([1, 1, 0], mesh_size = ms/10)
    p5 = geom.add_point([1, 2, 0], mesh_size = ms)
    p6 = geom.add_point([0, 2, 0], mesh_size = ms)

    l1 = geom.add_line(p1, p2)   
    l2 = geom.add_line(p2, p3)
    l3 = geom.add_line(p3, p4)
    l4 = geom.add_line(p4, p5)
    l5 = geom.add_line(p5, p6)
    l6 = geom.add_line(p6, p1)

    loop = geom.add_curve_loop([l1, l2, l3, l4, l5, l6])
    surf = geom.add_plane_surface(loop)

    geom.add_physical(l2, "rightBoundary")
    geom.add_physical(l6, "leftBoundary")

    mesh = geom.generate_mesh()

edge_indices = mesh.cell_sets_dict["rightBoundary"]["line"]
rightBoundary = mesh.cells_dict["line"][edge_indices].tolist()

edge_indices = mesh.cell_sets_dict["leftBoundary"]["line"]
leftBoundary = mesh.cells_dict["line"][edge_indices].tolist()

nodes = mesh.points
elements = mesh.cells[1].data

sim.mesh = gmsh2D(nodes, elements)

elementDict = {
    "element": [f"0:{len(elements)}"],
    "type": ["conv"],
    "value": [np.array([0.1, 20])]
}

edgeDict = {
    "edge": [leftBoundary, rightBoundary],
    "type": ["temp", "conv"],
    "value": [330, np.array([0.1, 20])]
}

nodeDict = {
    "node": [],
    "type": [],
    "value": []
}

sim.apply_boundary_conditions(elementDict, edgeDict, nodeDict)
sim.assemble()
T, Q = sim.solve()

from fem.d2.postprocess import plot_surface
plot_surface(sim.mesh, T)