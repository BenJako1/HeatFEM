import numpy as np
import matplotlib.pyplot as plt

class Bar:
    def __init__(self, k, A):
        self.nodes = None
        self.N = None
        self.elements = None

        self.k = k
        self.A = A
    
    def geometry(self, L=1, N=5, verbose=False):
        self.N = N
        self.nodes = np.linspace(0, L, N)

        self.elements = np.zeros([N-1, 2])
        for i in range(len(self.elements)):
            self.elements[i, 0] = i
            self.elements[i, 1] = i+1

        if verbose:
            print(f'Nodes at: x={self.nodes}')
            print(f'Elements with nodes: \n {self.elements}')

    def assemble_conductance(self, verbose=False):
        element_conductance = np.array([[1, -1],
                                       [-1, 1]])
        self.Conductance = np.zeros([self.N, self.N])
        for i in range(self.N-1):
            coefficient = self.k * self.A / (self.nodes[i+1] - self.nodes[i])
            self.Conductance[i, i] += element_conductance[0, 0] * coefficient
            self.Conductance[i, i+1] += element_conductance[0, 1] * coefficient
            self.Conductance[i+1, i] += element_conductance[1, 0] * coefficient
            self.Conductance[i+1, i+1] += element_conductance[1, 1] * coefficient
        
        if verbose:
            print(f'Conductivity matrix: \n {self.Conductance}')
    
    def bound(self, boundDict, verbose=False):
        if len(boundDict["nodes"]) == len(boundDict["type"]) == len(boundDict["value"]):
            # MAKE BOUNDARY INPUT SIMPLER
            self.T = np.full(shape=[self.N], fill_value=None, dtype=object)
            self.Q = np.full(shape=[self.N], fill_value=0, dtype=object)
            self.boundNodes = []
            for i in range(len(boundDict["nodes"])):
                if boundDict["type"][i] == "temp":
                    self.T[boundDict["nodes"][i]] = boundDict["value"][i]
                    self.Q[boundDict["nodes"][i]] = None
                    self.boundNodes.append(boundDict["nodes"][i])
                elif boundDict["type"][i] == "flux":
                    self.Q[boundDict["nodes"][i]] = boundDict["value"][i]
        else:
            raise ValueError("Boundary entries must have the same length")

        self.freeNodes = [int(i) for i in range(self.N) if i not in self.boundNodes]

        if verbose:
            print(f'Temperature vector: {self.T}')
            print(f'Flux vector: {self.Q}')

    def solve(self):
        solutionArray = self.Conductance
        Q = self.Q
        for i in range(len(self.boundNodes)):
            solutionArray = np.delete(solutionArray, self.boundNodes[i] - i, axis=0)
            Q = np.delete(Q, self.boundNodes[i] - i, axis=0)
            print(Q, solutionArray[:,0], self.boundNodes[i])
        
        for i in range(len(self.boundNodes)):
            Q = Q - solutionArray[:,self.boundNodes[i]] * self.T[self.boundNodes[i]]

        for i in range(len(self.boundNodes)):
            solutionArray = np.delete(solutionArray, self.boundNodes[i] - i, axis=1)

        print(solutionArray, Q)
        if len(solutionArray[0]) >= 2:
            T_unknown = np.linalg.solve(np.float64(solutionArray), np.float64(Q))
        elif len(solutionArray[0]) == 1:
            T_unknown = np.array([Q[0] / solutionArray[0,0]])

        T_sol = np.zeros([self.N])
        for i in range(len(self.freeNodes)):
            T_sol[self.freeNodes[i]] = T_unknown[i]
        for i in range(len(self.boundNodes)):
            T_sol[self.boundNodes[i]] = self.T[self.boundNodes[i]]
        
        return T_sol

if __name__ == "__main__":
    sim = Bar(k=1, A=1)
    sim.geometry(L=3, N=50, verbose=True)
    sim.assemble_conductance(verbose=True)
    boundDict = {"nodes": [0,24,49], "type": ["temp","flux","temp"], "value": [100,10,0]}
    sim.bound(boundDict, verbose=True)
    T = sim.solve()

    print(T)

    x = sim.nodes
    plt.plot(x, T)
    plt.show()

"""
TO DO:
- Fix error in solve() when more than one T input is given
    Q = Q - solutionArray[:,0] * self.T[self.boundNodes[i]]
    TypeError: unsupported operand type(s) for -: 'NoneType' and 'float'
- Make BC input easier
- Make variable names in solve() more clear
"""