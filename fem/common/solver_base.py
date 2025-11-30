import numpy as np

class SolverBase:
    def apply_boundary_conditions(self, *args, **kwargs):
        raise NotImplementedError

    def solve(self):
        K = self.K_sol.copy()
        Q = self.Q_sol.copy()

        self.boundNodes = np.sort(np.unique(self.boundNodes))
        self.freeNodes = np.array([int(i) for i in range(self.mesh.N) if i not in self.boundNodes])

        # remove rows
        for count, bn in enumerate(self.boundNodes):
            K = np.delete(K, bn - count, axis=0)
            Q = np.delete(Q, bn - count, axis=0)

        # subtract known boundary temperatures
        for bn in self.boundNodes:
            Q -= K[:, bn] * self.T[bn]

        # remove columns
        for count, bn in enumerate(self.boundNodes):
            K = np.delete(K, bn - count, axis=1)

        T_unknown = np.linalg.solve(K, Q)

        Tsol = np.zeros(self.mesh.N)
        for i, fn in enumerate(self.freeNodes):
            Tsol[fn] = T_unknown[i]
        for bn in self.boundNodes:
            Tsol[bn] = self.T[bn]

        Qsol = self.K @ Tsol

        if round(np.sum(Qsol), 5) != 0:
            raise ValueError("Flux is non-conservative!")
    
        return Tsol, Qsol