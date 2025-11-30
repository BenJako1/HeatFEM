from fem.d3.solver import HeatSolver3D
from fem.d3.mesh import TetMesh3D

mesh = TetMesh3D(120, 160, 50, 10, 10, 5)

wall = mesh.boundarySurface(x_in=0)
free = mesh.boundarySurface(x_in=120)
top = mesh.boundarySurface(z_in=1.25)
bottom = mesh.boundarySurface(z_in=0)

sim = HeatSolver3D(mesh, k=0.2)
sim.assemble()

sim.boundary.apply_temp2d(wall, 330)
sim.boundary.apply_conv2d(free, 2e-4, 30)
sim.boundary.apply_conv2d(top, 2e-4, 30)
sim.boundary.apply_conv2d(bottom, 2e-4, 30)

print("done assembling")
T, Q = sim.solve()
print("done solving")

from fem.d3.postprocess import visualise_tet_mesh

visualise_tet_mesh(mesh, T)