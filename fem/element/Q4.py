import numpy as np
from fem.common.utils import quad_area

class Q4:
    def get_K(self, sim, nodes, e):
        #k = sim.k[e]
        #t = sim.t[e]
        #A = sim.mesh.A[e]
        k=1
        t=1
        A=1

        # Define quadrature parameters
        gp = 1/np.sqrt(3)
        gauss_pts = np.array([[-gp, -gp], [gp, -gp], [gp,  gp], [-gp,  gp]])
        weights = np.array([1, 1, 1, 1])

        # Initialise Ke
        Ke = np.zeros((4, 4))

        # Quadrature summation loop
        for (xi,eta), w in zip(gauss_pts, weights):
            # Get shape gradients in natural coordinates
            grad_xi = self._dshape(xi, eta)

            # Compute Jacobian and det
            J = grad_xi @ nodes[..., :, :2]
            print(J)
            detJ = np.linalg.det(J)

            # Find cartesian shape gradients (this is our B)
            B = np.linalg.inv(J) @ grad_xi

            #print(B)

            # Create D matrix
            D = np.eye(2) * k * t

            integrand = (B.T @ D @ B)

            Ke += integrand * w * detJ

        return Ke
        
    def _dshape(self, xi, eta):
        dshape_dxi =  0.25 * np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
        dshape_deta = 0.25 * np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])

        return np.vstack([dshape_dxi, dshape_deta])
    
    def conv2d(self, coords, h, T_inf):
        gp = 1/np.sqrt(3)
        gauss_pts = np.array([[-gp, -gp], [gp, -gp], [gp,  gp], [-gp,  gp]])
        weights = np.array([1, 1, 1, 1])

        # Initialise Ke
        Ke = np.zeros((4, 4))
        f = np.zeros(4)

        # Quadrature summation loop
        for (xi,eta), w in zip(gauss_pts, weights):
            shape_xi =  0.25 * np.array([(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)])

            # Compute Jacobian and det
            grad_xi = self._dshape(xi, eta)
            J = grad_xi @ coords[..., :, :2]
            detJ = np.linalg.det(J)

            Ke += w * h * np.outer(shape_xi, shape_xi) * detJ
            f += w * T_inf * h * shape_xi * detJ

        return Ke, f
    
if __name__ == "__main__":
    #print((1+1/np.sqrt(3))*0.25)
    #print((1-1/np.sqrt(3))*0.25)
    nodes = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 2, 0],
                       [0, 1, 0]])
    natural_nodes = np.array([[-1, -1, 0],
                              [1, -1, 0],
                              [1, 1, 0],
                              [-1, 1, 0]])
    e = Q4()
    k = e.get_K(1, nodes, 1)

    print(k*6)