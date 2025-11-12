import numpy as np
import matplotlib.pyplot as plt
import utils

class oneD:
    def __init__(self, k, A):
        self.x = None
        self.N = None
        self.nodes = None
        self.elements = None

        self.k = k
        self.A = A
    
    def geometry(self, L, N, verbose=False):
        # Generate nodal indices and x-coordinates lists
        self.N = N
        self.nodes = np.array(list(range(N)))
        self.x = np.linspace(0, L, N)

        # Generate element node lists
        self.elements = np.zeros([N-1, 2])
        for i in range(len(self.elements)):
            self.elements[i, 0] = i
            self.elements[i, 1] = i+1

        # Calculate element lenghts
        self.element_len = np.zeros([N-1])
        for i in range(self.N-1):
            self.element_len[i] = ((self.x[i+1] - self.x[i]))

        if verbose:
            print(f'Nodes at: x={self.x}')
            print(f'Elements with nodes: \n {self.elements}')
            print(f'Elements lengths: \n {self.element_len}')

    def assemble_conductance(self, verbose=False):
        # Generate conductance matrix
        self.Conductance = np.zeros([self.N, self.N])
        for i in range(self.N-1):
            coefficient = self.k * self.A / self.element_len[i]
            self.Conductance[i, i] += coefficient
            self.Conductance[i, i+1] += -coefficient
            self.Conductance[i+1, i] += -coefficient
            self.Conductance[i+1, i+1] += coefficient
        
        if verbose:
            print(f'Conductivity matrix: \n {self.Conductance}')
    
    def bound(self, boundDict, verbose=False):
        # Unpack BC dictionary
        boundDict_unpacked = utils.unpack_dict(boundDict)
        nodes_unpacked = boundDict_unpacked["nodes"]
        types_unpacked = boundDict_unpacked["type"]
        values_unpacked = boundDict_unpacked["value"]

        # Initialize temperature and heat flux vectors
        self.T = np.full(shape=[self.N], fill_value=0, dtype=object)
        self.Q = np.full(shape=[self.N], fill_value=0, dtype=object)
        # Convection and boundNodes are used in matrix manipulation in solution
        self.Convection = np.full(shape=[self.N], fill_value=0, dtype=object)
        self.boundNodes = []
        for i in range(len(nodes_unpacked)):
            if types_unpacked[i] == "temp":
                self.T[nodes_unpacked[i]] = values_unpacked[i]
                self.boundNodes.append(nodes_unpacked[i])
            elif types_unpacked[i] == "flux":
                self.Q[nodes_unpacked[i]] += values_unpacked[i]
            elif types_unpacked[i] == "gen":
                Q_value = self.A * values_unpacked[i] * self.element_len[nodes_unpacked[i]]
                self.Q[nodes_unpacked[i]] += Q_value / 2
                self.Q[nodes_unpacked[i]+1] += Q_value / 2
            elif types_unpacked[i] == "convFace":   # Requires [h, T_inf] as input
                Q_value = float(values_unpacked[i][0]) * float(values_unpacked[i][1]) * self.A
                self.Q[nodes_unpacked[i]] += Q_value
                self.Convection[nodes_unpacked[i]] += float(values_unpacked[i][0]) * self.A
            elif types_unpacked[i] == "convSurf":   # Requires [h, W_c, T_inf]  as input
                Q_value = float(values_unpacked[i][0]) * float(values_unpacked[i][1]) * float(values_unpacked[i][2]) * self.element_len[nodes_unpacked[i]]
                conv_value = float(values_unpacked[i][0]) * float(values_unpacked[i][1]) * self.element_len[nodes_unpacked[i]]           
                self.Convection[nodes_unpacked[i]] += conv_value / 2
                self.Convection[nodes_unpacked[i]+1] += conv_value / 2
                self.Q[nodes_unpacked[i]] += Q_value / 2
                self.Q[nodes_unpacked[i]+1] += Q_value / 2
    
        # Create list of free nodes from boundNodes
        self.freeNodes = [int(i) for i in range(self.N) if i not in self.boundNodes]

        if verbose:
            print(f'Temperature vector: {self.T}')
            print(f'Flux vector: {self.Q}')
            print(f'Convection vector: {self.Convection}')

    def solve(self):
        K_calc = np.copy(self.Conductance)
        Q_calc = np.copy(self.Q)

        # Incorporate convection values in conduction matrix
        if any(self.Convection) != 0:
            for i in range(len(self.Convection)):
                K_calc[i,i] += self.Convection[i]

        # Delete rows with known temperature values
        for i in range(len(self.boundNodes)):
            K_calc = np.delete(K_calc, self.boundNodes[i] - i, axis=0)
            Q_calc = np.delete(Q_calc, self.boundNodes[i] - i, axis=0)
        
        # Move columns from conductance matrix to heat flux vector
        for i in range(len(self.boundNodes)):
            Q_calc = Q_calc - K_calc[:,self.boundNodes[i]] * self.T[self.boundNodes[i]]
        for i in range(len(self.boundNodes)):
            K_calc = np.delete(K_calc, self.boundNodes[i] - i, axis=1)

        # Solve with linalg.solve or arithmetic if matrix is one element
        if len(K_calc[0]) >= 2:
            T_unknown = np.linalg.solve(np.float64(K_calc), np.float64(Q_calc))
        elif len(K_calc[0]) == 1:
            T_unknown = np.array([Q_calc[0] / K_calc[0,0]])

        # Reassemble temperature vector with BC and calculated values
        T_sol = np.zeros([self.N])
        for i in range(len(self.freeNodes)):
            T_sol[self.freeNodes[i]] = T_unknown[i]
        for i in range(len(self.boundNodes)):
            T_sol[self.boundNodes[i]] = self.T[self.boundNodes[i]]
        
        # Back-calculate heat flux vector
        Q_sol = np.float64(self.Conductance) @ np.float64(T_sol)

        # Ensure sum of Q is zero
        if round(np.sum(Q_sol), 5) != 0:
            raise ValueError("Heat flux is non-conservative. Somethings gone wrong :(")

        return T_sol, Q_sol

if __name__ == "__main__":
    sim = oneD(k=0.1, A=1)
    sim.geometry(L=0.4, N=20, verbose=False)
    sim.assemble_conductance(verbose=False)
    convSurfProp = np.array([20, 0.1, 20])
    convFaceProp = np.array([2, 20])
    boundDict = {"nodes": [0,"0:19"], "type": ["temp","convSurf"], "value": [300,convSurfProp]}
    sim.bound(boundDict, verbose=False)
    T, Q = sim.solve()

    plt.plot(sim.x, T)
    plt.xlim(0,max(sim.x))
    plt.grid()
    plt.show()

"""
TO DO:
- Add "properties" dict to allow for non-uniform h and k and T_inf
- Edge case check especially on back calculating Q
- Implement "None" datatype for visualisation
- Figure out if the behaviours when convFace input < 0 is legit
"""