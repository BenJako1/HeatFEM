import numpy as np

class Property():
    def __init__(self, sim):
        self.sim = sim
    
    def k(self, values, elements=None):
        num_elems = len(self.sim.mesh.elements)
        self.sim.k = self._assign_property(values, elements, num_elems)
    
    def A(self, values, elements=None):
        num_elems = len(self.sim.mesh.elements)
        self.sim.A = self._assign_property(values, elements, num_elems)

    def t(self, values, elements=None):
        num_elems = len(self.sim.mesh.elements)
        self.sim.t = self._assign_property(values, elements, num_elems)

    def _assign_property(self, values, elements, num_elems):
        """
        Assign property values to a global per-element array.

        Parameters
        ----------
        values : scalar, list, or array
            Property values to assign.
            Can be scalar or match len(elements).

        elements : list or array or list-of-lists
            Element indices, or node-lists defining elements.

        num_elems : int
            Total number of elements in the mesh.

        Returns
        -------
        prop : ndarray
            Array of length num_elems with assigned values.
        """

        # convert values to array
        k = np.asarray(values)

        # Interpret the meaning of elements
        if elements is None:
            # global assignment
            if k.size == 1:
                return np.full(num_elems, k.item())
            elif k.size == num_elems:
                return k.copy()
            else:
                raise ValueError(
                    f"Global property must be scalar or length {num_elems}, not {k.size}"
                )

        # convert to python list first
        elements = list(elements)
        first = elements[0]

        # case: elements=[[0,1,2], [3,4,5]]
        if hasattr(first, "__len__") and not isinstance(first, (int, np.integer)):
            # treat each entry as defining an element implicitly → assign indices 0..N-1
            elements = np.arange(len(elements), dtype=int)

        else:
            # elements=[0,3,5]
            elements = np.asarray(elements, dtype=int)

        # Assign values
        prop = np.zeros(num_elems)

        if k.size == 1:
            # scalar assignment
            prop[elements] = k.item()

        elif k.size == len(elements):
            # match assignment
            prop[elements] = k

        else:
            raise ValueError(
                f"Property length {k.size} does not match number of selected elements {len(elements)}."
            )

        return prop