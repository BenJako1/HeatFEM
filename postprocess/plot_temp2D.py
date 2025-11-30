import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def plot_temp2D(mesh, T, cmap='inferno'):
    tri = mtri.Triangulation(mesh.x, mesh.y, mesh.elements)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(tri, T, cmap=cmap)
    plt.show()