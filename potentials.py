import numpy as np 
import sympy
import logging
import math
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
class ArbitraryPolynomialPotential():
    def __init__(self, nth_order:int, ) -> None:
        self.nth_order=nth_order
        self.coeffs= np.random.normal(scale=.05, size=(nth_order)) 
        logger.info(f'{self.coeffs}')

    def calc_static_pot(self,x):
        # \sum^{N}_{n=0}\frac{c_n}{n!}x^n
        Phi=np.sum([(self.coeffs[n]/math.factorial(n+1))
                    *(np.power(x,(n+1))) for n in range(self.coeffs.__len__())])
        logger.info(map('{}'.format, [(self.coeffs[n]/math.factorial(n+1))
                    *(np.power(x,(n+1))) for n in range(self.coeffs.__len__())]))
        return Phi
    
    def calc_static_jac(self,x):
        Phi_jac=([(self.coeffs[n]*(n+1)/math.factorial(n+1))
            *(x**(n)) for n in range(self.coeffs.__len__())])
        return Phi_jac
    
    def calc_static_hess(self,x):
        Phi_hess=([(self.coeffs[n]*(n+1)*n/math.factorial(n+1))
            *(x**(n-1)) for n in range(self.coeffs.__len__())])
        return
    
    def calc_coulomb(self,x):
        return
    
    def calc_coulomb_jac(self,x):
        return
    
    def calc_coulomb_hess(self,x):
        return
    
    
    def calc_tot_jac(self,x):
        return
    
    def calc_tot_hess(self,x):
        return
    
    def calc_tot_pot(self, x):
        return

