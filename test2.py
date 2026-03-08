import numpy as np
from fem.common.utils import quad_area

def dshape(xi, eta):
    dshape_dxi =  0.25 * np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
    dshape_deta = 0.25 * np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])

    return np.vstack([dshape_dxi, dshape_deta])

def quadrature(nodes, k, t):
    # Define quadrature parameters
    gp = 1/np.sqrt(3)
    gauss_pts = np.array([[-gp, -gp], [gp, -gp], [gp,  gp], [-gp,  gp]])
    weights = np.array([1, 1, 1, 1])

    # Initialise Ke
    result = np.zeros((4, 4))

    A = quad_area(nodes)

    # Quadrature summation loop
    for (xi,eta), w in zip(gauss_pts, weights):
        # Get shape gradients in natural coordinates
        shape_xi =  0.25 * np.array([[(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)]])

        grad_xi = dshape(xi, eta)

        # Compute Jacobian and det
        J = grad_xi @ nodes[..., :, :2]
        detJ = np.linalg.det(J)

        integrand = t * shape_xi.T @ shape_xi * detJ

        result += w * integrand

    return result

if __name__ == "__main__":
    nodes = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0]])
    natural_nodes = np.array([[-1, -1, 0],
                       [1, -1, 0],
                       [1, 1, 0],
                       [-1, 1, 0]])
    
    res = quadrature(natural_nodes, k=1, t=1)

    a = (1-1/np.sqrt(3))**2
    b = (1-1/np.sqrt(3))*(1+1/np.sqrt(3))
    c = (1+1/np.sqrt(3))**2

    print((a*a+2*b*b+c*c)/16, (2*a*b+2*b*c)/16, 2*(a*c+b*b)/16)
