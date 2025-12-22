import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def plot_temperature_along_line(mesh, T, p0, p1, npts=200):
    nodes = mesh.nodes[:, :2]
    elements = mesh.elements

    # ----------------------------------
    # Build triangulation (once!)
    # ----------------------------------
    tri = mtri.Triangulation(
        nodes[:, 0],
        nodes[:, 1],
        elements
    )

    interp = mtri.LinearTriInterpolator(tri, T)

    # ----------------------------------
    # Sample points on line
    # ----------------------------------
    s = np.linspace(0.0, 1.0, npts)
    x = p0[0] + s * (p1[0] - p0[0])
    y = p0[1] + s * (p1[1] - p0[1])

    T_line = interp(x, y)

    # Distance along line
    dist = np.sqrt((x - p0[0])**2 + (y - p0[1])**2)

    # ----------------------------------
    # Plot
    # ----------------------------------
    plt.figure()
    plt.plot(dist, T_line, lw=2)
    plt.xlabel("Distance along line")
    plt.ylabel("Temperature")
    plt.title("Temperature along line")
    plt.grid(True)
    plt.show()

    return dist, T_line