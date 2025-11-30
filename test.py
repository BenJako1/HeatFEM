import numpy as np

nodes = np.array([[0, 0, 0],
                  [1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

x1, y1, z1 = nodes[0]
x2, y2, z2 = nodes[1]
x3, y3, z3 = nodes[2]
x4, y4, z4 = nodes[3]

A = np.array([[1, x1, y1, z1],
              [1, x2, y2, z2],
              [1, x3, y3, z3],
              [1, x4, y4, z4]])

Ainv = np.linalg.inv(A)

print(Ainv)