# HeatFEM

HeatFEM is a lightweight finite element framework for solving steady-state heat
transfer problems in 1D, 2D, and 3D. It supports tetrahedral meshes, triangular
meshes, and line elements, and includes visualisation utilities.

## Installation

```bash
pip install git+https://github.com/BenJako1/HeatFEM.git
```

## Usage

Create a mesh object using one of the mesh classes (LineMesh1D, TriMesh2D or TetMesh3D)

A solver object is created using the HeatSolver class. Element type does not need to be specified as this is an ettribute of the mesh class.

```
from fem.solver import HeatSolver
sim = HeatSolver(mesh, ...)
```

Assemble the matrices with ```HeatSolver.assemble()``` and with the ```HeatSolver.boundary``` class, apply the boundary conditions.

Finally, the matrices can be solved using ```HeatSolver.solve()``` and the output can be visualised.

### Example: 1D Cooling Fin

```
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
```

