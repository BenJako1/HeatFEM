from fem.solver import HeatSolver
from mesh.TetMesh3D import TetMesh3D

# Create mesh object
mesh = TetMesh3D(120, 160, 1.25, 20, 30, 3)
# Define boundaries
wall = mesh.boundarySurface(x_in=0)
free = mesh.boundarySurface(x_in=120)
top = mesh.boundarySurface(z_in=1.25)
bottom = mesh.boundarySurface(z_in=0)

# Create simulation object
sim = HeatSolver(mesh, k=0.2)

# Assign properties
sim.property.k(0.2)
sim.property.t(1.25)

# Assemble matrices
sim.assemble()

# Apply boundary conditions
sim.boundary.apply_temp2d(wall, 330)
sim.boundary.apply_conv2d(free, 2e-4, 30)
sim.boundary.apply_conv2d(top, 2e-4, 30)
sim.boundary.apply_conv2d(bottom, 2e-4, 30)

# Solve
T, Q = sim.solve()

# Visualise
from postprocess.visualise_temp3D import visualise_temp3D
visualise_temp3D(mesh, T)