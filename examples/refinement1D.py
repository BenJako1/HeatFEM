from fem.solver.steadySolver import steadySolver
from mesh.LineMesh1D import LineMesh1D

refinements = [3, 5, 8, 10, 20]

for N in refinements:
    # Create mesh object
    mesh = LineMesh1D(1, N)
    # Define boundaries
    wall = 0
    free = mesh.N - 1

    # Create simulation object
    sim = steadySolver(mesh)
    sim.property.k(600)
    sim.property.cA(0.01)
    sim.property.P(200)

    # Assemble matrices
    sim.assemble()

    # Apply boundary conditions
    sim.boundary.apply_temp(wall, 330)
    sim.boundary.apply_conv1d(sim.mesh.elements, 0.4, 30)
    sim.boundary.apply_conv0d(free, 0.01, 30)

    # Solve
    T, Q = sim.solve()

    # Visualise
    import matplotlib.pyplot as plt
    plt.plot(mesh.x, T, label=N)

plt.legend(title="Nodes")
plt.xlim(min(mesh.x), max(mesh.x))
plt.xlabel("x")
plt.ylabel("Temperature")
plt.grid()
plt.show()