import pygmsh
import time
import numpy as np
from fem.solver.steadySolver import steadySolver
from mesh.GenericMesh import GenericMesh

# Start timer
start_time = time.perf_counter()

# Define geometry with pygmsh
with pygmsh.geo.Geometry() as geom:
    ms = 0.05 # Set mesh refinement

    p0 = geom.add_point([0, 0, 0], mesh_size = ms)
    p1 = geom.add_point([2, 0, 0], mesh_size = ms)
    p2 = geom.add_point([2, 1, 0], mesh_size = ms)
    p3 = geom.add_point([5, 1, 0], mesh_size = ms)
    p4 = geom.add_point([5, 3, 0], mesh_size = ms)
    p5 = geom.add_point([0, 3, 0], mesh_size = ms)

    l0 = geom.add_line(p0, p1)
    l1 = geom.add_line(p1, p2)
    l2 = geom.add_line(p2, p3)
    l3 = geom.add_line(p3, p4)
    l4 = geom.add_line(p4, p5)
    l5 = geom.add_line(p5, p0)

    loop = geom.add_curve_loop([l0, l1, l2, l3, l4, l5])
    surf = geom.add_plane_surface(loop)

    geom.add_physical(surf, label="Domain")
    geom.add_physical([l1, l2], label="conv_channel")
    geom.add_physical([l4], label="conv_face")

    gmsh = geom.generate_mesh()

# Create mesh
mesh = GenericMesh(gmsh.points, gmsh.cells_dict["triangle"], "T3")
# Define boundaries
conv_cold = gmsh.cells_dict["line"][gmsh.cell_sets["conv_channel"][0]]
conv_hot = gmsh.cells_dict["line"][gmsh.cell_sets["conv_face"][0]]

# Create simulation object
sim = steadySolver(mesh)

# Define properties
sim.property.k(25)
sim.property.t(1)

# Assemble
sim.assemble()

# Apply boundary conditions
sim.boundary.apply_conv1d(conv_cold, 200, 20)
sim.boundary.apply_conv1d(conv_hot, 200, 300)

# Solve
T, Q = sim.solve()

# Determine runtime
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")

# Visualise
from postprocess.plot_contour import plot_contour
plot_contour(mesh, T, cmap='viridis', levels=50, isolines=True)