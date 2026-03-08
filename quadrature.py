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
    Ke = np.zeros((4, 4))

    A = quad_area(nodes)

    # Quadrature summation loop
    for (xi,eta), w in zip(gauss_pts, weights):
        # Get shape gradients in natural coordinates
        grad_xi = dshape(xi, eta)

        # Compute Jacobian and det
        J = grad_xi @ nodes[..., :, :2]
        detJ = np.linalg.det(J)

        # Find cartesian shape gradients (this is our B)
        B = np.linalg.inv(J) @ grad_xi

        # Create D matrix
        D = np.diag(np.full(2, k * A * t))

        integrand = (B.T @ D @ B)

        Ke += integrand * w * detJ

    return Ke

if __name__ == "__main__":
    nodes = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0]])
    natural_nodes = np.array([[-1, -1, 0],
                       [1, -1, 0],
                       [1, 1, 0],
                       [-1, 1, 0]])
    Ke = quadrature(nodes, k=1, t=1)

    #print(Ke)

    gp = 1/np.sqrt(3)
    gauss_pts = np.array([[-gp, -gp], [gp, -gp], [gp,  gp], [-gp,  gp]])
    for (xi, eta) in gauss_pts:
        s = dshape(xi, eta)
        print(s)