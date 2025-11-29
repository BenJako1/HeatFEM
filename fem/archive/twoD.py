import numpy as np
import matplotlib.pyplot as plt
import utils

class twoD:
    def __init__(self, k, t):
        self.k = k
        self.t = t

    def geometry(self, nodal_coordinates, elemental_nodes):
        self.x = nodal_coordinates[:,0]
        self.y = nodal_coordinates[:,1]

        self.N = len(nodal_coordinates)
        
        self.nodes = np.array(list(range(self.N)))
        self.elements = elemental_nodes

        tris = nodal_coordinates[self.elements]
        x1, y1 = tris[:,0,0], tris[:,0,1]
        x2, y2 = tris[:,1,0], tris[:,1,1]
        x3, y3 = tris[:,2,0], tris[:,2,1]

        self.A = 0.5 * np.abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
            
    def assemble_conductance(self, verbose=False):
        self.Conductance = np.zeros([self.N, self.N])
        for element in range(len(self.elements)):
            y = [self.y[i] for i in self.elements[element]]
            x = [self.x[i] for i in self.elements[element]]
            
            B = 1/(2*self.A[element]) * np.array([[y[1]-y[2], y[2]-y[0], y[0]-y[1]],
                                                  [x[2]-x[1], x[0]-x[2], x[1]-x[0]]])
            
            elemental_Conductance = self.k * self.t * self.A[element] * np.transpose(B) @ B

            self.Conductance[self.elements[element][0],self.elements[element][0]] += elemental_Conductance[0,0]
            self.Conductance[self.elements[element][1],self.elements[element][0]] += elemental_Conductance[1,0]
            self.Conductance[self.elements[element][2],self.elements[element][0]] += elemental_Conductance[2,0]
            self.Conductance[self.elements[element][0],self.elements[element][1]] += elemental_Conductance[0,1]
            self.Conductance[self.elements[element][1],self.elements[element][1]] += elemental_Conductance[1,1]
            self.Conductance[self.elements[element][2],self.elements[element][1]] += elemental_Conductance[2,1]
            self.Conductance[self.elements[element][0],self.elements[element][2]] += elemental_Conductance[0,2]
            self.Conductance[self.elements[element][1],self.elements[element][2]] += elemental_Conductance[1,2]
            self.Conductance[self.elements[element][2],self.elements[element][2]] += elemental_Conductance[2,2]

        if verbose:
            print(f'Conductance matrix:\n{self.Conductance}')

    def bound(self, nodeDict, boundaryDict, elementDict, verbose=False):
        nodeDict_unpacked = utils.unpack_dict(nodeDict)
        node = nodeDict_unpacked["node"]
        nodeTypes = nodeDict_unpacked["type"]
        nodeValues = nodeDict_unpacked["value"]
        boundaryDict_unpacked = utils.unpack_dict(boundaryDict)
        boundary = boundaryDict_unpacked["boundary"]
        boundaryTypes = boundaryDict_unpacked["type"]
        boundaryValues = boundaryDict_unpacked["value"]
        elementDict_unpacked = utils.unpack_dict(elementDict)
        element = elementDict_unpacked["element"]
        elementTypes = elementDict_unpacked["type"]
        elementValues = elementDict_unpacked["value"]

        self.T = np.full(shape=[self.N], fill_value=0, dtype=float)
        self.Q = np.full(shape=[self.N], fill_value=0, dtype=float)
        self.Convection = np.full(shape=[self.N], fill_value=0, dtype=float)
        self.boundNodes = []

        for i in range(len(element)):
            if elementTypes[i] == "temp":
                for node_index in self.elements[element[i]]:
                    self.T[node_index] = elementValues[i]
                    self.boundNodes.append(node_index)
            if elementTypes[i] == "gen":
                for node_index in self.elements[element[i]]:
                    self.Q[node_index] += elementValues[i] * self.A[element[i]] * self.t / 3
        for i in range(len(boundary)):
            if boundaryTypes[i] == "temp":
                for node_index in boundary[i]:
                    self.T[node_index] = boundaryValues[i]
                    self.boundNodes.append(node_index)
            elif boundaryTypes[i] == "flux":
                for j in range(len(boundary[i])-1):
                    edge_length = np.sqrt((self.x[boundary[i][j]]-self.x[boundary[i][j+1]])**2+(self.y[boundary[i][j]]-self.y[boundary[i][j+1]])**2)
                    self.Q[boundary[i][j]] += boundaryValues[i] * self.t * edge_length / 2
                    self.Q[boundary[i][j+1]] += boundaryValues[i] * self.t * edge_length / 2
            elif boundaryTypes[i] == "conv":
                    for j in range(len(boundary[i])-1):
                        edge_length = np.sqrt((self.x[boundary[i][j]]-self.x[boundary[i][j+1]])**2+(self.y[boundary[i][j]]-self.y[boundary[i][j+1]])**2)
                        self.Q[boundary[i][j]] += boundaryValues[i][0] * boundaryValues[i][1] * self.t * edge_length / 2
                        self.Q[boundary[i][j+1]] += boundaryValues[i][0] * boundaryValues[i][1] * self.t * edge_length / 2
                        self.Convection[boundary[i][j]] += boundaryValues[i][0] * boundaryValues[i][1] * self.t * edge_length / 2
                        self.Convection[boundary[i][j+1]] += boundaryValues[i][0] * boundaryValues[i][1] * self.t * edge_length / 2
        for i in range(len(node)):
            if nodeTypes[i] == "temp":
                self.T[node[i]] = nodeValues[i]
                self.boundNodes.append(node[i])
            elif nodeTypes[i] == "flux":
                self.Q[node[i]] += nodeValues[i]
        
        self.boundNodes = np.sort(np.unique(self.boundNodes))
        self.freeNodes = np.array([int(i) for i in range(self.N) if i not in self.boundNodes])
        
        if verbose:
            print(f'Temperature vector: {self.T}')
            print(f'Flux vector: {self.Q}')

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
            Q_calc -= K_calc[:,self.boundNodes[i]] * self.T[self.boundNodes[i]]
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
    sim = twoD(k=600, t=0.1)
    nodes, elements = utils.mesh_tri(L=20, H=10, n_x=21, n_y=11)
    print(len(elements))
    sim.geometry(nodes, elements)
    sim.assemble_conductance(verbose=False)
    left = np.array([0, 11, 22, 33, 44])
    top = np.array([44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54])
    right = np.array([10, 21, 32, 43, 54])
    bottom = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #left = np.array([0, 21, 42, 63, 84, 105, 126, 147, 168, 189, 210])
    #right = np.array([20, 41, 62, 83, 104, 125, 146, 167, 188, 209, 230])
    #top = np.int16(np.linspace(210, 230, 21))
    #bottom = np.int16(np.linspace(0, 20, 21))

    convParameters = np.array([0.1, 20])
    elementDict = {"element": [f"0:{len(sim.elements)}"], "type": ["gen"], "value": [10]}
    boundaryDict = {"boundary": [left, top, right, bottom], "type": ["conv", "temp", "conv", "temp"], "value": [convParameters, 10, convParameters, 10]}
    nodeDict = {"node": [], "type": [], "value": []}    
    sim.bound(nodeDict, boundaryDict, elementDict, verbose=True)
    T, Q = sim.solve()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(sim.x, sim.y, T, cmap='inferno')

    plt.show()

"""
TO DO:
- Add convection
    - Maybe: build convection matrix and add to calcMat instead of index operations
- decide on edge_length handling in boundary 
"""