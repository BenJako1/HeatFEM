from fem.solver import HeatSolver
from mesh.LineMesh1D import LineMesh1D

# Create mesh object
mesh = LineMesh1D(120, 20)
# Define boundaries
wall = 0
free = mesh.N - 1

# Create simulation object
sim = HeatSolver(mesh, k=0.2, A=200)
# Assemble matrices
sim.assemble()

# Apply boundary conditions
sim.boundary.apply_temp0d(wall, 330)
sim.boundary.apply_conv1d(sim.mesh.elements, 2e-4, 320, 30)
sim.boundary.apply_conv0d(free, 2e-4, sim.A, 30)

# Solve
T, Q = sim.solve()

# Visualise
from postprocess.plot_temp1D import plot_temp1D
plot_temp1D(mesh, T)