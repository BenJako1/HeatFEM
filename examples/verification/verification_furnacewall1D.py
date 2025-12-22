from fem.solver.steadySolver import steadySolver
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

sim = steadySolver(mesh)
sim.property.k(k, elements)
sim.property.A(A)

sim.assemble()

print("done assembling")
sim.boundary.apply_conv0d(inside, h_inside, A, T_inf_inside)
sim.boundary.apply_conv0d(outside, h_outside, A, T_inf_outside)

T, Q = sim.solve()

print(T)
print(Q)

from postprocess.plot_temp1D import plot_temp1D

plot_temp1D(mesh, T)