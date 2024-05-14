import numpy as np
from scipy.optimize import minimize
from potentials import ArbitrarySymmetricPolynomialPotential
import logging
from scipy.constants import electron_volt, epsilon_0, atomic_mass
M=171
k_e=electron_volt**2 / (4*np.pi*epsilon_0)
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
class TrapModel():
    def __init__(self, potential:ArbitrarySymmetricPolynomialPotential, n_ions:int, omega_0=1e6) -> None:
        self.n_ions=n_ions
        self.V=potential
        self.init_pot=True
        self.callback_hist=[]

        self.radial_f=np.zeros((n_ions, 1))
        self.radial_p=np.zeros((n_ions, n_ions))
        self.axial_f=np.zeros((n_ions, 1))
        self.axial_p=np.zeros((n_ions, n_ions))       

        self.ion_pos=np.zeros(n_ions)
        self.l0=(k_e/(atomic_mass*M*(2*np.pi*omega_0)**2))**(1/3)
        self.eVscale=4*np.pi*self.l0*epsilon_0/(electron_volt**2)
        
        #self.ion_pos=self.calc_ion_eq_pos()
    
    

    def calc_ion_eq_pos(self, verbose:bool=True):
        #No ion_pos yet set, assume equal spacing first guess. Other
        if self.init_pot: init_pos_guess=np.linspace(-(self.n_ions-1)/2, (self.n_ions-1)/2, self.n_ions,)
            
        res=minimize(self.V.calcV, init_pos_guess, hess=self.V.calcVhess, jac=self.V.calcVjac, method='trust-ncg', options={
            'disp':verbose, 'maxiter': 400})
        x=res['x']*self.l0
        #ogger.info(f'{res}')
        ion_pos=np.reshape(x, self.ion_pos.shape, order="F")
        return ion_pos
        
    
    def optimize_ion_pos(self, ideal_spacing:float, verbose:bool=True):
       """
       ideal_pos assumes equal spacing as ideal if no arg given
       """
       ideal_pos=np.linspace(-(self.n_ions-1)/2, (self.n_ions-1)/2, self.n_ions )*ideal_spacing
       logger.info(f'{ideal_pos}')
       c0=[]
       for p in self.V.opt_args:
           c0.append(getattr(self .V, p))

       logger.info(f'{c0}')
       def c(cost, args): 
           self.V.optimize_params(cost)
           pred_pos=self.calc_ion_eq_pos()
           logger.info('{}'.format(pred_pos))
           error=np.sum(np.square(pred_pos-ideal_pos))/(pred_pos.__len__())
           logger.info(f'Err {error}')
           return error

       def callback(cost):
           self.callback_hist.append(cost)
           
       options={'disp':verbose, 'maxiter':4000}    
       res=minimize(c, ideal_pos, method='L-BFGS-B', options=options, args=self.V.coeffs)
       self.ion_pos=self.calc_ion_eq_pos()

    
    def initial_adiabatic_split(self, target_dist:int):
        return NotImplementedError()
    
    def calc_mode_structure(self,):
        return
    


