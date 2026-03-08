import numpy as np

class Property():
    def __init__(self, sim):
        self.sim = sim
    
    # k doesnt make sense being a nodal value because then we cant have defined border between neighboring elements
    # A and t doent make sense being elemental since that would lead to disctete geometry
    
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

        Parameters:
            values : scalar, list, or array
                Property values to assign.
                Can be scalar or match len(elements).

            elements : list or array or list-of-lists
                Element indices, or node-lists defining elements.

            num_elems : int
                Total number of elements in the mesh.

        Returns:
            prop : ndarray
                Array of length num_elems with assigned values.
        """

        values = np.asarray(values)

        # Default to global assignment, then direct assignment
        # If neither is met, elemt-wise assignment
        if elements is None:
            if values.size == 1:
                return np.full(num_elems, values.item())
            elif values.size == num_elems:
                return values.copy()
            else:
                raise ValueError(
                    f"Global property must be scalar or length {num_elems}, not {values.size}"
                )
            
        elements = list(elements)
        first = elements[0]

        # If first element is array_like and not an integer
        # Allows for input of elements as list of nodes and list of element indices
        if hasattr(first, "__len__") and not isinstance(first, (int, np.integer)):
            elements = np.arange(len(elements), dtype=int)

        else:
            elements = np.asarray(elements, dtype=int)

        prop = np.zeros(num_elems)

        if values.size == 1:
            prop[elements] = values.item()

        elif values.size == len(elements):
            prop[elements] = values

        else:
            raise ValueError(
                f"Property length {values.size} does not match number of selected elements {len(elements)}."
            )

        return prop