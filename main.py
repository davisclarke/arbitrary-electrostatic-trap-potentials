from potentials import ArbitraryPolynomialPotential
import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
if __name__=='__main__':
    v=ArbitraryPolynomialPotential(3).calc_static_pot([2,2,3])
    logger.info(f'{v}')