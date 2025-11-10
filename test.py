import numpy as np

arr = np.array([[  20, -10,   0,   0],
                [ -10,  20, -10,   0],
                [   0, -10,  20, -10],
                [   0,   0, -10,  10]])

vec = np.array([2500, 0, -200, 0])

sol = np.linalg.solve(arr, vec)

print(sol)