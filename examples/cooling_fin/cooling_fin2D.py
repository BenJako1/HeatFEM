from fem.solver import HeatSolver
from mesh.TriMesh2D import TriMesh2D

# Create mesh object
mesh = TriMesh2D(120, 160, 20, 30)
# Define boundaries
wall = mesh.boundaryEdge(x_in=0)
free = mesh.boundaryEdge(x_in=120)

# Create simulation object
sim = HeatSolver(mesh, k=0.2, t=1.25)

# Assign properties
sim.property.k(0.2)
sim.property.t(1.25)

# Assemble matrices
sim.assemble()

# Apply boundary conditions
sim.boundary.apply_temp1d(wall, 330)
sim.boundary.apply_conv2d(sim.mesh.elements, 2e-4*2, 30)
sim.boundary.apply_conv1d(free, 2e-4, sim.t[0], 30)

# Solve
T, Q = sim.solve()
# Visualise
from postprocess.plot_temp2D import plot_temp2D
plot_temp2D(mesh, T)