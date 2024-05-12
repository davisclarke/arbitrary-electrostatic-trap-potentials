from potentials import ArbitrarySymmetricPolynomialPotential
from trap_model import TrapModel
import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
if __name__=='__main__':
    v=ArbitrarySymmetricPolynomialPotential(3)
    logger.info(f'{v}')
    model=TrapModel(v, 5)
