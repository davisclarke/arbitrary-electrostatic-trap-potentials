import numpy as np
from scipy.optimize import minimize
from potentials import ArbitrarySymmetricPolynomialPotential
import logging
import math
import numpy.linalg as lin
from scipy.constants import electron_volt, epsilon_0, atomic_mass
YB=171
k_e=electron_volt**2 / (4*np.pi*epsilon_0)
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
class TrapModel():
    def __init__(self, potential:ArbitrarySymmetricPolynomialPotential, n_ions:int, omega_0=(3e6*np.pi)) -> None:
        assert n_ions%2==0
        self.n_ions=n_ions
        self.V=potential
        self.init_pot=True
        self.callback_hist=[]

        self.radial_f=np.zeros((n_ions, 1))
        self.radial_p=np.zeros((n_ions, n_ions))
        self.axial_f=np.zeros((n_ions, 1))
        self.axial_p=np.zeros((n_ions, n_ions))       

        self.m=np.ones(n_ions)*YB
        self.ion_pos=np.zeros(n_ions)
        self.l0=(k_e/(atomic_mass*YB*(2*np.pi*omega_0)**2))**(1/3)
        self.eVscale=4*np.pi*self.l0*epsilon_0/(electron_volt**2)
        
        #self.ion_pos=self.calc_ion_eq_pos()
    
    

    def calc_ion_eq_pos(self, verbose:bool=True):
        #No ion_pos yet set, assume equal spacing first guess. Other
        if self.init_pot: init_pos_guess=np.linspace(-(self.n_ions-1)/2, (self.n_ions-1)/2, self.n_ions,)
            
        res=minimize(self.V.calcV, init_pos_guess, hess=self.V.calcVhess, jac=self.V.calcVjac, method='trust-ncg', options={
            'disp':verbose, 'maxiter': 1000})
        x=res['x']*self.l0
        #ogger.info(f'{res}')
        ion_pos=np.reshape(x, self.ion_pos.shape, order="F")
        self.ion_pos=ion_pos
        return ion_pos
        
    
    def optimize_ion_pos(self, ideal_spacing:float, verbose:bool=True):
       """
       ideal_pos assumes equal spacing as ideal if no arg given
       """
       ideal_pos=np.linspace(-(self.n_ions-1)/2, (self.n_ions-1)/2, self.n_ions )*ideal_spacing/self.l0
       logger.info(f'{ideal_pos}')
       
       def c(cost, ): 
           self.V.optimize_params(cost)
           pred_pos=self.calc_ion_eq_pos()/self.l0
           error=np.sum(np.square(pred_pos-ideal_pos))
           return error

       def callback(cost):
           self.callback_hist.append(cost)
           
       options={'disp':verbose, 'gtol': 1e-20, 'eps': 1e-5}    
       res=minimize(c, self.V.coeffs, method='L-BFGS-B', options=options, )
       logger.info(f'{res}')
    
    def optimize_ion_pos_for_symmetric_split(self, ideal_pos_right:np.ndarray, verbose:bool=True, ):
       n_left=int(self.n_ions/2)
       ideal_pos_right=ideal_pos_right

       
       def c(cost,): 
           self.V.optimize_params(cost)
           pred_pos=self.calc_ion_eq_pos(verbose=verbose)[n_left:]
           logger.info(f'{self.ion_pos}')
           error=np.sum(np.square(pred_pos-ideal_pos_right))/pred_pos.__len__()
           return error

       def callback(cost):
           self.callback_hist.append(cost)
           
       options={'disp':verbose, 'gtol': 1e-16, 'maxiter':4000, 'ftol': 1e-17, 'eps': 1e-7}    
       res=minimize(c, self.V.coeffs, method='L-BFGS-B', options=options, )
       logger.info(f'{res}')
    
    def calc_modes(self):
        """  Calculate the motional modes of the ion chain given the specified potential (self.V)

        Returns: The radial/axial normal mode frequencies + participation matrices. Shapes specified in __init__
        """
        #Make sure the ions are at equilibrium before calculating the modes
        self.calc_ion_eq_pos()
        n_ions = self.n_ions

        M = np.diag(self.m)*atomic_mass

        # Calculate axial modes from the total hessian
        hess = self.V.Vhess
        eig, vec = lin.eig(
            lin.inv(M) @ hess * k_e / self.l0**3
        )

        axial_f = np.sqrt(eig) / 2 / math.pi
        axial_p = vec

        axial_f, axial_p = self.sort_and_normalize_modes(axial_f, axial_p, axial=True)
        # Calculate the radial modes from the coulomb potential hessian  + RF confinement
        t1, t2, hess_coul = self.V.calc_coulomb(self.ion_pos/self.l0)
        eig, vec = lin.eig(
            lin.inv(M)@ (M * (self.V.fsec*1e6 * math.pi * 2)**2 - 1 / 2 * hess_coul * k_e/self.l0**3))
        radial_f = np.sqrt(eig)/math.pi/2
        radial_p = vec

        radial_f, radial_p = self.sort_and_normalize_modes(radial_f, radial_p)

        self.axial_f = axial_f
        self.axial_p = axial_p
        self.radial_f = radial_f
        self.radial_p = radial_p

        return axial_f, axial_p, radial_f, radial_p
    
    @staticmethod
    def sort_and_normalize_modes(f_modes: np.ndarray, mode_p: np.ndarray, axial: bool=False):
        """Sorts modes so COM is always first and normalizes the participation matrix to unit vectors

        Note: It is assumed that the the indices of the frequency vector correspond to columns of the participation
        matrix, even though they may not be sorted in a monotonic order.

        Args:
            f_modes (np.ndarray): [n_ions, ] vector of normal mode frequencies
            mode_p (np.ndarray): [n_ions, n_ions] participation matrix. Columns are mode indices.
            axial (bool): Flag to set whether modes are axial/radial modes, since COM is lowest/highest frequency

        Returns:

        """
        n_ions = f_modes.size
        ind = np.argsort(f_modes, kind="mergesort")

        # For transverse modes, COM is highest
        if not axial:
            ind = np.flipud(ind)

        f_modes = f_modes[ind]
        mode_p = mode_p[:, ind]
        # Normalize to unit length
        mode_p *= np.reshape(np.tile(1 / lin.norm(mode_p, axis=0), n_ions), (n_ions, n_ions), order="C")
        # First ion always positive
        mode_p *= np.reshape(np.tile(np.sign(mode_p[0, :]), n_ions), (n_ions, n_ions), order="C")

        return f_modes, mode_p
    

    
def initial_adiabatic_split(n_ions, nth_order, n_left: int, target_dist:float, n_slices:int, ideal_spacing:float, fsec)-> list:
    ideal_eq_pos=np.linspace(-(n_ions-1)/2, (n_ions-1)/2, n_ions )*ideal_spacing
    dist_per_slice=target_dist/n_slices
    logger.info(f'{dist_per_slice}')
    left_pos, right_pos=ideal_eq_pos[:n_left],ideal_eq_pos[n_left:]
    left_list,right_list=[],[]
    
    for n in range(1, int(n_slices+1)):
        left_list.append(left_pos-(dist_per_slice*n))
        right_list.append(right_pos+(dist_per_slice*n))
    logger.info(f'{left_list}, {right_list}')
    target_split_list=[np.concatenate((left_list[n], right_list[n])) for n in range(n_slices)]
    logger.info(f'{target_split_list}')
    model_list=[]
    V=ArbitrarySymmetricPolynomialPotential(np.random.normal(loc=1e-3, scale=1e-6, size=(nth_order)), fsec)
    model=TrapModel(V, n_ions,)
    model.optimize_ion_pos(ideal_spacing)
    pos=model.calc_ion_eq_pos()
    modes=model.calc_modes()
    model_list.append((pos, modes, model.V.coeffs))
    logger.info(f'{model_list[0]}')
    for n,target_pos in enumerate(target_split_list):
        V=ArbitrarySymmetricPolynomialPotential(model_list[n-1][2], fsec)
        model=TrapModel(V, n_ions,)
        model.optimize_ion_pos_for_split(target_pos)
        pos=model.calc_ion_eq_pos()
        modes=model.calc_modes()
        model_list.append((pos, modes, model.V.coeffs))
    return model_list


