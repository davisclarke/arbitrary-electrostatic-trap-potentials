import numpy as np 
import sympy
import logging
import math
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
class ArbitrarySymmetricPolynomialPotential():
    def __init__(self, coeffs:np.ndarray, fsec:float=None ) -> None:
       # assert (nth_order%2==0) #N must be even!
        self.coeffs=np.array(coeffs)
        self.nth_order=self.coeffs.__len__()
        logger.info(f'{self.nth_order}')
        #self.coeffs=np.random.normal(0,.01,(nth_order))
        self.opt_args=[f'coeffs']
        #self.init_attr()
        self.fsec=np.array(fsec)

    def calc_static_pot(self,x:np.ndarray):
        # \sum^{N}_{n=0}\frac{c_n}{n!}x^n
        Phi=np.sum([np.sum([(self.coeffs[int(n-1)]/math.factorial(2*n))
                    *(pow(xi,(2*n))) for n in range(1, self.nth_order)]) for xi in x])
        #logger.info(f'V: {Phi}')
        return Phi
    
    def calc_static_jac(self,x:np.ndarray):
        Phi_jac=np.array([np.sum([(self.coeffs[int(n-1)]*(2*n)/math.factorial(2*n))
                    *(pow(xi,(2*n-1))) for n in range(1, self.nth_order)]) for xi in x])
        #logger.info(f'jac V: {Phi_jac}')
        return Phi_jac
    
    def calc_static_hess(self,x:np.ndarray):
        Phi_hess=np.array([np.sum([(self.coeffs[int(n-1)]*(2*n)*(2*n-1)/math.factorial(2*n))
                    *pow(xi,((2*n-2))) for n in range(1, self.nth_order)]) for xi in x])*np.identity(x.size)
        #logger.info(f'Hess: {Phi_hess}')
        return Phi_hess
    
    def calc_coulomb(self,x:np.ndarray):
        """
            Calc dimless Coulomb forces.
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
        V=self.calc_static_pot(x)+self.calc_coulomb(x)[0]
        self.V=V
        return V
    
    def calcVjac(self, x:np.ndarray):
        #Calculate Jacobian
        jac=self.calc_static_jac(x)+self.calc_coulomb(x)[1]
        self.Vjac=jac
        return jac
    
    def calcVhess(self, x:np.ndarray):
        #Calculate Hessian
        hess= self.calc_static_hess(x)+self.calc_coulomb(x)[2]
        self.Vhess=hess
        return hess
    
    def optimize_params(self, c:np.ndarray):
            setattr(self, 'coeffs', c)


    





