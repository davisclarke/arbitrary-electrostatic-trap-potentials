import numpy as np
from scipy.optimize import minimize
from potentials import ArbitrarySymmetricPolynomialPotential
import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
class TrapModel():
    def __init__(self, potential:ArbitrarySymmetricPolynomialPotential, n_ions:int) -> None:
        self.n_ions=n_ions
        self.V=potential
        self.init_pot=True
        self.callback_hist=[]
        self.ion_pos=self.calc_ion_eq_pos()

    def calc_ion_eq_pos(self, verbose:bool=True):
        #No ion_pos yet set, assume equal spacing first guess. Other
        if self.init_pot: init_pos_guess=np.linspace(-self.n_ions*1e-5, self.n_ions*1e-5, self.n_ions )
            
        pos=minimize(self.V.calcV, init_pos_guess, hess=self.V.calcVhess, jac=self.V.calcVjac, method='trust-ncg', options={
            'disp':verbose, 'gtol':1e-9})
    
    def optimize_ion_pos(self, ideal_pos:np.ndarray, verbose:bool=True):
       """
       ideal_pos assumes equal spacing as ideal if no arg given
       """
       if ideal_pos==None:
           #equal spacing assumption if 
           pass


       def c(x): 
           pred_pos=self.calc_ion_eq_pos()
           return np.sum(np.square(pred_pos-ideal_pos))

       def callback(cost):
           self.callback_hist.append(cost)

       res=minimize(c, self.ion_pos, args=self.pot.coeffs, method='Powell', disp=verbose)
    
    def initial_adiabatic_split(self, target_dist:int):
        return NotImplementedError()

