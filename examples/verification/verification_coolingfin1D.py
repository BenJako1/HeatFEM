from fem.solver import HeatSolver
from mesh.LineMesh1D import LineMesh1D

k = 0.2
A = 200
h = 2e-4
t = 320
T_inf = 30
T_wall = 330

mesh = LineMesh1D(120, 4)

wall = 0
free = mesh.N - 1

sim = HeatSolver(mesh, k=k, A=A)
sim.assemble()

print(sim.K)

sim.boundary.apply_temp0d(wall, T_wall)
sim.boundary.apply_conv0d(free, h, sim.A, T_inf)
sim.boundary.apply_conv1d(sim.mesh.elements, h, t, T_inf)

print(sim.K_sol)
print(sim.Q_sol)

T, Q = sim.solve()

print(T)
print(Q)