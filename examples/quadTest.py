from fem.solver.steadySolver import steadySolver
from mesh.QuadMesh2D import QuadMesh2D

mesh = QuadMesh2D(2, 1, 3, 2)
wall = mesh.boundaryEdge(x_in=0)
free = mesh.boundaryEdge(x_in=2)

# Create simulation object
sim = steadySolver(mesh)

# Assign properties
sim.property.k(1)
sim.property.t(1)

# Assemble matrices
sim.assemble()

# Apply boundary conditions
sim.boundary.apply_temp1d(wall, 100)
sim.boundary.apply_conv2d(sim.mesh.elements, 2, 10)
sim.boundary.apply_conv1d(free, 1, sim.t[0], 10)

# Solve
T, Q = sim.solve()

# Visualise
from postprocess.plot_contour import plot_contour
plot_contour(mesh, T, isolines=False)