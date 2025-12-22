import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

def plot_contour(mesh, T, levels=21, cmap='inferno', isolines=False):
    levels = np.linspace(T.min(), T.max(), levels)
    tri = mtri.Triangulation(mesh.x, mesh.y, mesh.elements)

    fig = plt.figure()
    ax = fig.add_subplot()

    c = ax.tricontourf(tri, T, levels=levels, cmap=cmap)
    if isolines:
        ax.tricontour(tri, T, colors='black', levels=20)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.colorbar(c)

    plt.show()