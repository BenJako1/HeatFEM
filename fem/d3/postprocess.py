import numpy as np
import pyvista as pv

def visualise_tet_mesh(mesh, T, show_edges=True, clim=None):
    """
    Visualise tetrahedral mesh with temperature data.
    
    mesh.x, mesh.y, mesh.z : (Nnodes,)
    mesh.elements : (Ne, 4) tetra elementsectivity
    T : nodal temperature vector (Nnodes,)
    """

    # Build (Nnodes, 3) coordinate array
    points = np.column_stack((mesh.x, mesh.y, mesh.z))

    # PyVista requires a "cell array": [npts, id0, id1, id2, id3, npts, ...]
    ne = mesh.elements.shape[0]
    cells = np.hstack(
        np.column_stack([np.full(ne, 4), mesh.elements]).astype(np.int32)
    )

    # VTK cell type for tetrahedra = 10
    celltypes = np.full(ne, 10, dtype=np.uint8)

    # Build the unstructured grid
    grid = pv.UnstructuredGrid(cells, celltypes, points)

    # Attach nodal temperature data
    grid.point_data["Temperature"] = T

    # Create plotter
    plotter = pv.Plotter()
    actor = plotter.add_mesh(
        grid,
        scalars="Temperature",
        show_edges=show_edges,
        opacity=1.0,
        clim=clim,          # colour limits (optional)
        cmap="inferno",     # nice for temperature
    )

    plotter.add_scalar_bar(
        title="Temperature",
        n_labels=5,
        italic=False,
        bold=True,
    )

    plotter.add_axes()

    plotter.show()


# ---------------------------------------------------------------------------
# Example usage:
# visualise_tet_mesh(mesh, T)
# ---------------------------------------------------------------------------