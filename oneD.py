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
    
    def geometry(self, L=1, N=5, verbose=False):
        self.N = N
        self.nodes = np.array(list(range(N)))
        self.x = np.linspace(0, L, N)

        self.elements = np.zeros([N-1, 2])
        for i in range(len(self.elements)):
            self.elements[i, 0] = i
            self.elements[i, 1] = i+1

        self.element_len = np.zeros([N-1, 1])
        for i in range(self.N-1):
            self.element_len[i] = ((self.x[i+1] - self.x[i]))

        if verbose:
            print(f'Nodes at: x={self.x}')
            print(f'Elements with nodes: \n {self.elements}')
            print(f'Elements lengths: \n {self.element_len}')

    def assemble_conductance(self, verbose=False):
        element_conductance = np.array([[1, -1],
                                       [-1, 1]])
        self.Conductance = np.zeros([self.N, self.N])
        for i in range(self.N-1):
            coefficient = self.k * self.A / self.element_len[i]
            self.Conductance[i, i] += element_conductance[0, 0] * coefficient
            self.Conductance[i, i+1] += element_conductance[0, 1] * coefficient
            self.Conductance[i+1, i] += element_conductance[1, 0] * coefficient
            self.Conductance[i+1, i+1] += element_conductance[1, 1] * coefficient
        
        if verbose:
            print(f'Conductivity matrix: \n {self.Conductance}')
    
    def bound(self, boundDict, verbose=False):
        boundDict_unpacked = utils.unpack_dict(boundDict)
        nodes_unpacked = boundDict_unpacked["nodes"]
        types_unpacked = boundDict_unpacked["type"]
        values_unpacked = boundDict_unpacked["value"]

        self.T = np.full(shape=[self.N], fill_value=0, dtype=object)
        self.Q = np.full(shape=[self.N], fill_value=0, dtype=object)
        self.Convection = np.full(shape=[self.N], fill_value=0, dtype=object)
        self.boundNodes = []
        for i in range(len(nodes_unpacked)):
            if types_unpacked[i] == "temp":
                self.T[nodes_unpacked[i]] = values_unpacked[i]
                if self.Q[nodes_unpacked[i]] == 0:
                    self.Q[nodes_unpacked[i]] = None
                self.boundNodes.append(nodes_unpacked[i])
            elif types_unpacked[i] == "flux":
                if self.Q[nodes_unpacked[i]] == None:
                    self.Q[nodes_unpacked[i]] = values_unpacked[i]
                else:
                    self.Q[nodes_unpacked[i]] += values_unpacked[i]
            elif types_unpacked[i] == "gen":
                Q_value = self.A * values_unpacked[i] * self.element_len[nodes_unpacked[i]]
                if self.Q[nodes_unpacked[i]] == None:
                    self.Q[nodes_unpacked[i]] = Q_value / 2
                else:
                    self.Q[nodes_unpacked[i]] += Q_value / 2
                if self.Q[nodes_unpacked[i]+1] == None:
                    self.Q[nodes_unpacked[i]+1] = Q_value / 2
                else:
                    self.Q[nodes_unpacked[i]+1] += Q_value / 2
            elif types_unpacked[i] == "convFace":   # Requires [h, T_inf] as input
                Q_value = float(values_unpacked[i][0]) * float(values_unpacked[i][1]) * self.A
                self.Convection[nodes_unpacked[i]] += float(values_unpacked[i][0]) * self.A
                if self.Q[nodes_unpacked[i]] == None:
                    self.Q[nodes_unpacked[i]] = Q_value
                else:
                    self.Q[nodes_unpacked[i]] += Q_value
            elif types_unpacked[i] == "convSurf":   # Requires [h, W_c, T_inf]  as input
                Q_value = float(values_unpacked[i][0]) * float(values_unpacked[i][1]) * float(values_unpacked[i][2]) * self.element_len[nodes_unpacked[i]]
                self.Convection[nodes_unpacked[i]] += float(values_unpacked[i][0]) * float(values_unpacked[i][1]) * self.element_len[nodes_unpacked[i]]
                if self.Q[nodes_unpacked[i]] == None:
                    self.Q[nodes_unpacked[i]] = Q_value / 2
                else:
                    self.Q[nodes_unpacked[i]] += Q_value / 2
                if self.Q[nodes_unpacked[i]+1] == None:
                    self.Q[nodes_unpacked[i]+1] = Q_value / 2
                else:
                    self.Q[nodes_unpacked[i]+1] += Q_value / 2
    
        self.freeNodes = [int(i) for i in range(self.N) if i not in self.boundNodes]

        if verbose:
            print(f'Temperature vector: {self.T}')
            print(f'Flux vector: {self.Q}')
            print(f'Convection vector: {self.Convection}')

    def solve(self):
        solutionArray = self.Conductance
        Q = self.Q

        if any(self.Convection) != 0:
            for i in range(len(self.Convection)):
                solutionArray[i,i] += self.Convection[i]
        
        print(self.Convection)
        print(solutionArray)

        for i in range(len(self.boundNodes)):
            solutionArray = np.delete(solutionArray, self.boundNodes[i] - i, axis=0)
            Q = np.delete(Q, self.boundNodes[i] - i, axis=0)
        
        for i in range(len(self.boundNodes)):
            Q = Q - solutionArray[:,self.boundNodes[i]] * self.T[self.boundNodes[i]]

        for i in range(len(self.boundNodes)):
            solutionArray = np.delete(solutionArray, self.boundNodes[i] - i, axis=1)

        if len(solutionArray[0]) >= 2:
            T_unknown = np.linalg.solve(np.float64(solutionArray), np.float64(Q))
        elif len(solutionArray[0]) == 1:
            T_unknown = np.array([Q[0] / solutionArray[0,0]])

        T_sol = np.zeros([self.N])
        for i in range(len(self.freeNodes)):
            T_sol[self.freeNodes[i]] = T_unknown[i]
        for i in range(len(self.boundNodes)):
            T_sol[self.boundNodes[i]] = self.T[self.boundNodes[i]]
        
        Q_sol = np.float64(self.Conductance) @ np.float64(T_sol)

        return T_sol, Q_sol

if __name__ == "__main__":
    sim = oneD(k=0.2, A=200)
    sim.geometry(L=120, N=4, verbose=False)
    sim.assemble_conductance(verbose=False)
    convSurfProp = np.array([2e-4, 160, 30])
    convFaceProp = np.array([2e-4, 30])
    boundDict = {"nodes": [0,"1:3",3], "type": ["temp","flux","flux"], "value": [330,0,-200]}
    sim.bound(boundDict, verbose=True)
    T, Q = sim.solve()

    print(T)
    print(Q)

    plt.plot(sim.x, T)
    plt.xlim(0,max(sim.x))
    plt.grid()
    plt.show()

"""
TO DO:
- Fix convection parameter input
- Add "properties" dict to allow for non-uniform h and k and T_inf
- Edge case check especially on back calculating Q
- Allow for separate h, T_inf and A_s inputs
- Evaluate whether datatype "None" should realy be used
- Figure out if the behaviours when convFace input < 0 is legit
"""