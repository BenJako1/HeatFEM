import pygmsh
import matplotlib.pyplot as plt
import numpy as np

with pygmsh.geo.Geometry() as geom:
    p1 = geom.add_point([0, 0, 0])
    p2 = geom.add_point([2, 0, 0])
    p3 = geom.add_point([2, 1, 0])
    p4 = geom.add_point([1, 1, 0])
    p5 = geom.add_point([1, 2, 0])
    p6 = geom.add_point([0, 2, 0])

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
edges_on_boundary = mesh.cells_dict["line"][edge_indices]
#nodes_on_boundary = np.unique(edges_on_boundary)

edge_indices = mesh.cell_sets_dict["leftBoundary"]["line"]
edges_on_boundary1 = mesh.cells_dict["line"][edge_indices]
#nodes_on_boundary2 = np.unique(edges_on_boundary)

print(mesh.cells[1].data)

plt.scatter(mesh.points[:,0], mesh.points[:,1])
plt.grid()
#plt.show()
