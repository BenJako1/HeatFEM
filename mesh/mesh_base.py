import numpy as np

class BaseMesh:
    def boundaryEdge(self, x_in=None, y_in=None, z_in=None, tol=1e-8):
        """
        Vectorised detection of triangle edges lying on a given axis.
        Only edges with both nodes on the axis are returned.

        Parameters:
            mesh : object
                Must have mesh.elements (Ne x 4) and mesh.x, mesh.y, mesh.z arrays.
            x_in, y_in, z_in : float or None
                Axis coordinates.
            tol : float
                Tolerance for floating point comparison.

        Returns:
            boundary_edges : np.ndarray
                Array of shape (Nfaces, 3) containing node indices of boundary faces.
        """

        if self.type == "T3":
            elem_edges = np.array([
                [0, 1],
                [0, 2],
                [1, 2]
            ])
        elif self.type == "Q4":
            elem_edges = np.array([
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0]
            ])

        # Expand element nodes to faces
        # shape (Ne, 4 faces, 3 nodes per face)
        all_edges = self.elements[:, elem_edges]

        # Flatten to (Ne*4, 3)
        flat_edges = all_edges.reshape(-1, 2)

        # Gather coordinates of each face
        x = self.x[flat_edges]  # shape (Ne*4, 3)
        y = self.y[flat_edges]
        z = self.z[flat_edges]

        # Start with all True mask
        mask = np.ones(len(flat_edges), dtype=bool)

        if x_in is not None:
            mask &= np.all(np.abs(x - x_in) < tol, axis=1)
        if y_in is not None:
            mask &= np.all(np.abs(y - y_in) < tol, axis=1)
        if z_in is not None:
            mask &= np.all(np.abs(z - z_in) < tol, axis=1)

        # Select faces that satisfy plane condition
        boundary_edges = flat_edges[mask]

        return boundary_edges
    
    def boundarySurface(self, x_in=None, y_in=None, z_in=None, tol=1e-8):
        """
        Detection of surfaces lying on a user-specified plane.

        Parameters:
            mesh : object
                Must have mesh.elements (Ne x 4) and mesh.x, mesh.y, mesh.z arrays.
            x_in, y_in, z_in : float or None
                Plane coordinates.
            tol : float
                Tolerance for floating point comparison.

        Returns:
            boundary_faces : np.ndarray
                Array of shape (Nfaces, 3) containing node indices of boundary faces.
        """

        # 4 faces per tetrahedron
        tet_faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ])

        # Expand element nodes to faces
        # shape (Ne, 4 faces, 3 nodes per face)
        all_faces = self.elements[:, tet_faces]

        # Flatten to (Ne*4, 3)
        flat_faces = all_faces.reshape(-1, 3)

        # Gather coordinates of each face
        x = self.x[flat_faces]  # shape (Ne*4, 3)
        y = self.y[flat_faces]
        z = self.z[flat_faces]

        # Start with all True mask
        mask = np.ones(len(flat_faces), dtype=bool)

        if x_in is not None:
            mask &= np.all(np.abs(x - x_in) < tol, axis=1)
        if y_in is not None:
            mask &= np.all(np.abs(y - y_in) < tol, axis=1)
        if z_in is not None:
            mask &= np.all(np.abs(z - z_in) < tol, axis=1)

        # Select faces that satisfy plane condition
        boundary_faces = flat_faces[mask]

        return boundary_faces