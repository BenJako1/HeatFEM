from fem.solver import HeatSolver
from mesh.GenericMesh import GenericMesh

k = [1.2, 0.2]
A = 1
h_inside = 12
h_outside = 2
T_inf_inside = 1500
T_inf_outside = 20

nodes = [[0, 0, 0],
         [0.25, 0, 0],
         [0.37, 0, 0]]

elements = [[0, 1],
            [1, 2]]

mesh = GenericMesh(nodes, elements, "1D")

inside = 0
outside = mesh.N - 1

sim = HeatSolver(mesh)
sim.property.k(k, elements)
sim.property.A(A)

sim.assemble()

print(sim.K)

sim.boundary.apply_conv0d(inside, h_inside, A, T_inf_inside)
sim.boundary.apply_conv0d(outside, h_outside, A, T_inf_outside)

print(sim.K_sol)
print(sim.Q_sol)

T, Q = sim.solve()

print(T)
print(Q)