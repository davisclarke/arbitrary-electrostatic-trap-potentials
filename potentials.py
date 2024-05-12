import numpy as np 
import sympy
import logging
import math
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
class ArbitrarySymmetricPolynomialPotential():
    def __init__(self, nth_order:int, ) -> None:
        self.nth_order=nth_order
        self.coeffs= np.random.normal(scale=.05, size=(nth_order)) 
        self.opt_args=[f'{param}' for param in self.coeffs]

    def calc_static_pot(self,x:np.ndarray):
        # \sum^{N}_{n=0}\frac{c_n}{n!}x^n
        Phi=np.sum([np.sum([(self.coeffs[n])
                    *(np.power(xi,(2*n))) for n in range(self.coeffs.__len__())]) for xi in x])
        return Phi
    
    def calc_static_jac(self,x:np.ndarray):
        Phi_jac=np.array([np.sum([(self.coeffs[n]*(2*n))
                    *(np.power(xi,(2*n-1))) for n in range(self.coeffs.__len__())]) for xi in x])
        logger.info(f'jac: {Phi_jac.shape}')
        return Phi_jac
    
    def calc_static_hess(self,x:np.ndarray):
        Phi_hess=np.array([np.sum([(self.coeffs[n]*(2*n)*(2*n-1))
                    *(np.power(xi,(2*n-2))) for n in range(self.coeffs.__len__())]) for xi in x])*np.identity(x.size)
        return Phi_hess
    
    def calc_coulomb(self,x:np.ndarray):
        """
            Calc dimless Coulomb forces
            Credit to Laird Egan
        """
        N = x.size #Number of ions
        xii = np.reshape(np.tile(x, N), (N, N))
        xjj = xii.T
        xij = xii - xjj  # xi - xj for each ion in matrix form
        sign_xij = np.sign(xij)
        np.seterr(divide='ignore')  # Ignore divide by zero on diagonal
        R_inv = 1 / np.abs(xij)  # 1/Rij distance matrix
        np.fill_diagonal(R_inv, 0)  # Set Diagonal Terms to 0
        coulomb_V = np.sum(R_inv) / 2  # Coulomb Interaction Potential
        coulomb_J = -1 * np.sum(sign_xij * R_inv**2, axis=0)
        coulomb_H = -2 * R_inv**3
        np.fill_diagonal(coulomb_H, 2 * np.sum(R_inv**3, axis=0))
        return coulomb_V, coulomb_J, coulomb_H
    
    def calcV(self, x:np.ndarray):
        #Calculate potential
        return self.calc_static_pot(x)+self.calc_coulomb(x)[0]
    
    def calcVjac(self, x:np.ndarray):
        #Calculate Jacobian
        return self.calc_static_jac(x)+self.calc_coulomb(x)[1]
    
    def calcVhess(self, x:np.ndarray):
        #Calculate Hessian
        return self.calc_static_hess(x)+self.calc_coulomb(x)[2]



