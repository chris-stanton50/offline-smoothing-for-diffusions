"""
Module for continuous-discrete state space models (CDSSMs). These are the key building blocks for the algorithms implemented in
this package, and represent the models on which we want to conduct statistical inference.
"""

import numpy as np
import numpy.linalg as nla
from particles.state_space_models import StateSpaceModel
from particles.kalman import LinearGauss, MVLinearGauss
from sdes.state_space_models import DiscreteDiscreteSSM, DiscreteLinearGauss, MvDiscreteLinearGauss
from sdes.sdes import LinearSDE, OrnsteinUhlenbeck, MvOrnsteinUhlenbeck
from sdes.numerical_schemes import EulerMaruyama, MvEulerMaruyama
from sdes.tools import use_end_point
import particles.distributions as dists

class CDSSM:
    """
    A CDSSM is defined by:

    - A starting point: x_0: float/(1, dimX)
    - A Model SDE instance
    - Delta_s: The timestep between observations: float
    - The parameters of the observation density f_t(y_t|e_t): taken in as kwargs
    - (Optionally) A linear Gaussian proxy obs density \tilde{f}_t(y_t|e_t)

    The linear, Gaussian proxy observation density should ideally closely match the observation density.
    The proxy linear Gaussian observation density is of the form:

    Y_t | E_t = e_t \sim N(L(t) e_t, \sigma_Y(t)^2) 

    Define L(t) through the method 'LY'
    Define \sigma_Y(t) through the method 'sigmaY'

    To create a CDSSM, subclass and define:

    - 'PY' as a method that returns the observation density as distribution object.
            currently assumes that this only depends on the end point of the latent process
            and not on the path.
    - default observation density parameters in the class attribute/property 'default_params' 
    - Method 'LY' to define a proxy linear Gaussian 
    - Method 'CovY' to define the covariance matrix of Y at time t.
    
    The Subclass 'GaussianCDSSM' covers the common special case where the observation
    density is Gaussian, and LY and CovY do not depend on the time t.
    
    Other subclasses can be created for time inhomogeneous CDSSMs. 
        
    In the case where the model sde is a Linear SDE, one can also use the methods:
    
    - 'discrete_ssm_params' to define the parameters of a discrete time state space model
    - 'discrete_ssm_cls' to define the class of the discrete time state space model
    - 'discrete_ssm' to return an instance of the discrete time state space model
    """

    def __init__(self, model_sde, x0=None, delta_s = 1., **kwargs):
        self.model_sde = model_sde
        self.delta_s = delta_s
        self.x0 = self.default_x0 if x0 is None else x0
        StateSpaceModel.__init__(self, **{**kwargs, **self.sde_params})
        
    def _error_msg(self, method):
        return StateSpaceModel._error_msg(self, method)

    @classmethod
    def state_container(cls, N, T, num, delta_s, dimX=1):
        shape = [N, T]
        numerical_scheme_cls = EulerMaruyama if dimX == 1 else MvEulerMaruyama
        dummy_sde = OrnsteinUhlenbeck if dimX == 1 else MvOrnsteinUhlenbeck
        numerical_scheme = numerical_scheme_cls(dummy_sde()) # Dummy instance to access method
        state_container = numerical_scheme._create_state_container(delta_s, num, shape, dimX=dimX)
        return state_container

    @property
    def default_x0(self):
        return 0. if self.dimX == 1 else np.zeros((1, self.dimX))

    @property
    def dimX(self):
        return self.model_sde.dimX

    @property
    def dimY(self):
        covY = self.CovY(0)
        if type(covY) == float or covY.shape[0] == 1:
            return 1
        else:
            return covY.shape[0]

    @property
    def sde_params(self):
        return self.model_sde.params
    
    @property
    def obs_params(self):
        return {k: self.__dict__[k] for k in self.default_params.keys()}

    @property
    def params(self):
        return {**self.sde_params, **self.obs_params}

    def PY(self, t, xp, x):
        """Conditional distribution of Y_t, given the states.

        Inputs
        ------------
        t: float
        xp: (N, )/(N, dimX) array (dimX = 1/dimX > 1)
        x:  (N, )/(N, dimX) array (dimX = 1/dimX > 1)
        
        xp and x should represent the end points of the latent process at time t-1 and t respectively.
        """
        return self._error_msg("PY")

    def proxyPY(self, t, e_t):
        """
        e_t is a (N, dimX) array in ND
        e_t is a (N,) array in 1D
        
        We could split this our into a different class in the future.
        """
        # e_t = np.array([e_t]) if type(e_t) == float else np.ravel(e_t)
        if self.dimY == 1 and self.dimX == 1: # LY is # (1, 1), e_t is (N, ). LY could be scalar, e_t could be scalar. 
            sigma_y = self.sigmaY(t).ravel() if type(self.sigmaY(t)) == np.ndarray else self.sigmaY(t)
            e_t = np.ravel(e_t) if type(e_t) == np.ndarray else e_t
            LY = self.LY(t).ravel() if type(self.LY(t)) == np.ndarray else self.LY(t)
            return dists.Normal(loc=LY*e_t, scale=sigma_y)
        if self.dimY == 1 and self.dimX > 1: # LY is # (1, dimX), e_t is (N, dimX) or (1, dimX) #sigmaY is float or (1, 1)
            sigma_y = self.sigmaY(t).ravel() if type(self.sigmaY(t)) == np.ndarray else self.sigmaY(t)
            loc = np.einsum('ij,kj->ki', self.LY(t), e_t).ravel() # (1, dimX), (N, dimX) -> (N, 1) -> (N,)
            return dists.Normal(loc=loc, scale=sigma_y)
        if self.dimY > 1 and self.dimX == 1: # LY is # (dimY, 1), e_t is (N, ). e_t could be scalar, covY is (dimY, dimY)
            if type(e_t) == float or e_t.shape[0] == 1:
                loc = np.ravel(self.LY(t)*e_t) # (dimY, 1), (1,) -> (dimY, 1) -> (dimY,)
            else:
                loc = (self.LY(t) * e_t).T # (dimY, 1), (N,) -> (N, dimY)
            return dists.MvNormal(loc=loc, cov=self.CovY(t))
        if self.dimY > 1 and self.dimX > 1: # LY is # (dimY, dimX), e_t is (N, dimX)/(1, dimX). covY is (dimY, dimY)
            loc = np.einsum('ij,kj->ki', self.LY(t), e_t) # (dimY, dimX), (N, dimX) -> (N, dimY)
            loc = loc.ravel() if loc.shape[0] == 1 else loc # (1, dimY) -> (dimY,)
            return dists.MvNormal(loc=loc, cov=self.CovY(t))

    def CovY(self, t):
        """
        Inputs
        ------------
        t: float

        Returns
        ------------
        CovY: (dimY, dimY) array        
        """
        return self._error_msg(self, "CovY")
    
    def sigmaY(self, t):
        """
        Inputs
        ------------
        t: float

        Returns
        ------------
        CovY: (dimY, dimY) array        
        """
        cov = self.CovY(t)
        if type(cov) == float or cov.shape == (1, 1):
            return np.sqrt(cov)
        else:
            return nla.cholesky(cov) 
    
    def LY(self, t):
        """
        Inputs
        ------------
        t: float

        Returns
        ------------
        LY: (dimY, dimX) array        
        """
        return self._error_msg(self, "LY")

    def s(self, t):
        """
        Placeholder for evaluation of the observation times s_t: for now we assume that 
        they are observed at equidistant times. 
        """
        return t * self.delta_s

    def _P0_sim(self, size, num=1000):
        return self.model_sde.simulate(size, self.x0, t_start=self.s(0), t_end=self.s(1), num=num)

    def _PX_sim(self, t, xp, size, num=1000):
        return self.model_sde.simulate(xp.shape[0], xp[xp.dtype.names[-1]], t_start=self.s(t), t_end=self.s(t+1), num=num)

    def simulate_given_x(self, x):
        t_end = x[0].dtype.names[-1]
        lag_x = [None] + x[:-1]
        return [
            self.PY(t, xp, x[t_end]).rvs(size=1) for t, (xp, x) in enumerate(zip(lag_x, x))
        ]

    def simulate(self, T, num=1000):
        """Simulate state and observation processes.

        Parameters
        ----------
        T: int
            processes are simulated from time 0 to time T-1

        Returns
        -------
        x, y: lists
            lists of length T
        """
        x = [self._P0_sim(size=1, num=num)]
        for t in range(1, T):
            x.append(self._PX_sim(t, x[-1], size=1, num=num))
        y = self.simulate_given_x(x)
        return x, y

    def discrete_ssm(self):
        """
        Returns a state space model that is a proxy for the CDSSM.
        Only possible when the latent SDE is Linear, so that the transition density is tractable.
        """
        return self.discrete_ssm_cls(self) if self.discrete_ssm_cls else None
        
class GaussianCDSSM(CDSSM):
    """
    Gaussian CDSSM that involves obsersations with Gaussian distributed noise:
    Dimension of observations is by default set to match that of the latent states.
    
    Y_t | E_t=e_t \sim N_d(Ge_t, \Cov_Y).
        
    Takes the following parameters as input:

    G: (dimY, dimX) array
    CovY: (dimY, dimY) array
    
    The dimension of Y is then inferred from the inputs. 
    If these parameters are not provided, default behaviour is to observe 
    each component of the latent process with independent, additive noise.
    """
    @property
    def default_params(self):
        if self.dimX > 1:
            def_params = {'G': np.eye(self.dimX), 'covY': np.eye(self.dimX)}
        else:
            def_params = {'G': 1., 'covY': 1.}
        return def_params
    
    def __init__(self, model_sde, x0=None, delta_s = 1., **kwargs):
        super().__init__(model_sde, x0, delta_s, **kwargs)
        self._check_L_CovY_dims(self.G, self.covY)

    @property
    def dimY(self):
        return 1 if type(self.covY) == float else self.covY.shape[0]

    def PY(self, t, xp, x):
        return self.proxyPY(t, x)
    
    def LY(self, t):
        return self.G
        
    def CovY(self, t):
        return self.covY
    
    def discrete_ssm(self):
        """
        Returns a state space model that is a proxy for the CDSSM.
        Only possible when the latent SDE is Linear, so that the transition density is tractable.
        """
        return super().discrete_ssm()
       
    def gen_score_add_func(self, param_name):
        gplpx = self.model_sde.grad_param_log_px
        @use_end_point
        def add_func(t, x, xf):
            if t == 0:
                out = gplpx(self.s(0), self.s(1), self.x0, x, param_name)
                out += gplpx(self.s(1), self.s(2), x, xf, param_name)
            else:
                out = gplpx(self.s(t+1), self.s(t+2), x, xf, param_name)
                return out
        return add_func

    def _check_L_CovY_dims(self, G, covY):
        """
        TO DO: Re-write this - it is not very readable.
        """
        if type(G) == float and type(covY) == float:
            return
        if type(covY) == float:
            if type(G) != float or G.shape[0] != 1:
                raise ValueError("Dimension mismatch between parameters L and CovY")
        if type(G) == float:
            if type(covY) != float or covY.shape[0] != 1 or covY.shape[1] != 1:
                raise ValueError("Dimension mismatch between parameters L and CovY")
        if covY.shape[0] != G.shape[0]:
            raise ValueError("Dimension mismatch between parameters L and CovY")
        if covY.shape[1] != covY.shape[0]:
            raise ValueError("CovY is not a square matrix")
        if G.shape[1] != self.dimX:
            raise ValueError("Second dimension of L must match dimension of latent SDE")

    @property
    def discrete_ssm_cls(self):
        if isinstance(self.model_sde, LinearSDE):
            return DiscreteLinearGauss if (self.dimX == 1 and self.dimY == 1) else MvDiscreteLinearGauss

    @property
    def discrete_ssm_params(self):
        params = super().discrete_ssm_params
        if self.dimX == 1 and self.dimY == 1:
            params['sigmaY'] = np.sqrt(self.covY).ravel()[0]
            return params
        if self.dimX == 1 and self.dimY > 1:
            raise NotImplementedError("Case dimX == 1 and dimY > 1 not implemented")
        return params
    
class TimeSwitchingGaussianCDSSM(GaussianCDSSM):   
    """
    Not the most helpful CDSSM, but useful for testing purposes.
    This CDSSM is linear Gaussian but time inhomogeneous, so we use 
    'DisreteDiscreteSSM' as the discrete time proxy.
    """    
    @property
    def default_params(self):
        if self.dimX > 1:
            def_params = {'G_1': np.eye(self.dimX), 'G_2': np.eye(self.dimX), 
                          'covY_1': np.eye(self.dimX), 'covY_2': 0.1*np.eye(self.dimX), 't_switchY': 10}
        else:
            def_params = {'G_1': 1., 'covY_1': 1., 'G_2': 1., 'covY_2': 0.1*1., 't_switchY': 10}
        return def_params
 
    def __init__(self, model_sde, x0=None, delta_s = 1., **kwargs):
        CDSSM.__init__(self, model_sde, x0, delta_s, **kwargs)
        self._check_L_CovY_dims(self.G_1, self.covY_1)
        self._check_L_CovY_dims(self.G_2, self.covY_2)
        if type(self.G_1) is np.ndarray:
            assert self.G_1.shape == self.G_2.shape, 'Shapes of G_1 and G_2 do not match'
        if type(self.covY_1) is np.ndarray:
            assert self.covY_1.shape == self.covY_2.shape, 'Shapes of CovY_1 and CovY_2 do not match'

    @property
    def dimY(self):
        return 1 if type(self.covY_1) == float else self.covY_1.shape[0]

    def LY(self, t):
        t = t+1
        return self.G_1 if t < self.t_switchY else self.G_2
    
    def CovY(self, t):
        t = t+1
        return self.covY_1 if t < self.t_switchY else self.covY_2
    
    @property
    def discrete_ssm_cls(self):
        return DiscreteDiscreteSSM
        

"""
The following classes are now deprecated
"""

class OU_CDSSM(GaussianCDSSM):

    BenchmarkSSMCls = LinearGauss

    @property
    def benchmark_ssm_params(self):
        """
        If the model SDE of a standard CDSSM is a linear SDE that can be solved analytically, one has
        access to the transition density of the diffusion, that is linear Gaussian. Thus, one can construct 
        a LGSSM. 
        """
        linear_gauss_params = {'sigmaY': self.eta,
                       'rho': self.model_sde._a(self.s(0), self.s(1)),
                       'sigmaX': np.sqrt(self.model_sde._v(self.s(0), self.s(1))), # Assume equidistant observations for now
                       'sigma0': np.sqrt(self.model_sde._v(self.s(0), self.s(1)))
                      }
        return linear_gauss_params

class Reparam_OU_CDSSM(OU_CDSSM):

    default_params = {'rho': 0.8187307530779818, 'sigmaX_2': 0.07417798964198115}
    # Corresponds to rho=0.3, mu=0., phi=0.3, eta_sq=0.01**2 in OU_CDSSM
    
    def __init__(self, x0=0., delta_s = 1., **kwargs):
        StateSpaceModel.__init__(self, **kwargs)
        self.x0 = x0
        self.delta_s = delta_s
        model_sde_params = {'rho': -np.log(self.rho), 'mu': 0., 'phi': self.phi()}
        self.model_sde = self.ModelSDECls(**model_sde_params)
        self.eta_sq = 0.01 ** 2
    
    def phi(self):
        return np.sqrt((-2.*np.log(self.rho) * self.sigmaX_2)/(1-self.rho**2))