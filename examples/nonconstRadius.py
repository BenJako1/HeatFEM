from fem.solver.steadySolver import steadySolver
from mesh.LineMesh1D import LineMesh1D

# Create mesh object
mesh = LineMesh1D(1, 50)
# Define boundaries
x0 = 0
x1 = mesh.N - 1

# Create simulation object
sim = steadySolver(mesh)

# Assign properties
sim.property.k(20)
area_func = lambda x: x**0.5 + 1
print(len(mesh.x))
sim.property.A(area_func(mesh.x))

# Assemble matrices
sim.assemble()

# Apply boundary conditions
sim.boundary.apply_temp(x0, 100)
sim.boundary.apply_conv0d(x1, 1e3, 20)

# Solve
T, Q = sim.solve()

# Visualise
from postprocess.plot_temp1D import plot_temp1D
plot_temp1D(mesh, T)