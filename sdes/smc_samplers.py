import numpy as np
import particles.smc_samplers as ssp
from particles.smc_samplers import StaticModel
from particles.kalman import Kalman
from sdes.state_space_models import ReparamLinearGauss
from sdes.continuous_discrete_ssms import Reparam_OU_CDSSM

class LGStaticModel(StaticModel):
    """A Linear, Gaussian Static Model. 
    
        This Static model is not capable of doing vectorised 
        calculations over theta particles. This class should 
        only be used in the context of ideal Metropolis Hastings 
        algorithms.
        
        A parameterisation of the LGSSM is assumed that results in conjugacy for direct comparison with Gibbs samplers. 
    """

    def loglik(self, theta, t=None):
        """log-likelihood at given parameter values.

        Parameters
        ----------
        theta: dict-like
            theta['par'] is a ndarray containing the N values for parameter par
        t: int
            time (if set to None, the full log-likelihood is returned)

        Returns
        -------
        l: float numpy.ndarray
            the N log-likelihood values
        """
        linear_gauss_params = {par_name: theta[par_name] for par_name in theta.dtype.names}
        ssm = ReparamLinearGauss(**linear_gauss_params)
        if not self.check_params(linear_gauss_params):
            return np.array([np.nan])
        kf = Kalman(ssm=ssm, data=self.data)
        kf.filter()
        loglik = np.cumsum(kf.logpyt) 
        return loglik
                
    def check_params(self, lg_params):
        for k, v in lg_params.items():
            if k[:5] == 'sigma':
                if v <= 0:
                    return False
        return True
        

        # if t is None:
        #     t = self.T - 1
        # l = np.zeros(shape=theta.shape[0])
        # for s in range(t + 1):
        #     l += self.logpyt(theta, s)
        # np.nan_to_num(l, copy=False, nan=-np.inf)
        # return l
