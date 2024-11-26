"""
Modules for standard state space models.

Implemented classes:
---------------------------------------------------------------------------------------------

DiscreteDiscreteSSM: A class that constructs a State Space model from an instance of a CDSSM.
MvDiscreteLinearGauss: A class that constructs a multivariate time homogeneous linear Gaussian model
                        from a CDSSM with a multivariate linear SDE, and a linear, Gaussian observation density.
DiscreteLinearGauss: A class that constructs a univariate latent linear Gaussian model from a CDSSM with a univariate
                        linear SDE, and a linear, Gaussian observation density.
                        
                        
We should use the DiscreteDiscreteSSM class when any of the following cases:
- The latent SDE has a known transition density that is non-Gaussian
- The observation density is not linear, Gaussian
- The CDSSM is linear, Gaussian, but time-inhomogeneous (e.g time-switching)
    (Future developement will be to extend Kalman filtering/smoothing to this case) 
- The CDSSM is linear, Gaussian, but there is a drift term (b) in the latent process. 
    (Future developement will be to extend Kalman filtering/smoothing to this case)

---------------------------------------------------------------------------------------------
"""

import numpy as np
import numpy.linalg as nla
from particles.state_space_models import StateSpaceModel, Bootstrap, GuidedPF, AuxiliaryPF
from particles.kalman import LinearGauss, MVLinearGauss
from sdes.sdes import LinearSDE, MvLinearSDE
from sdes.numerical_schemes import EulerMaruyama, MvEulerMaruyama
from sdes.tools import use_end_point
import particles.distributions as dists

from collections import OrderedDict

class DiscreteDiscreteSSM(StateSpaceModel):
    
    """
    Constructs a State Space model from an instance of a CDSSM.
    Requires that the following methods are defined for the CDSSM:
    
    - transition_dist
    - optimal_proposal_dist (optional)
    
    The observation density is taken from the input CDSSM.
    
    Kalman filtering/smoothing is not compatible with with class, however
    it can be applied to time inhomogeneous CDSSMs.
    """    
    def __init__(self, cdssm, **kwargs):
        self.cdssm = cdssm
        StateSpaceModel.__init__(self, **kwargs)

    def PX0(self):
        return self.cdssm.model_sde.transition_dist(self.cdssm.s(0), self.cdssm.s(1), self.cdssm.x0)

    def PX(self, t, xp):
        return self.cdssm.model_sde.transition_dist(self.cdssm.s(t), self.cdssm.s(t+1), xp)

    def PY(self, t, xp, x):
        return self.cdssm.PY(t, xp, x)

    def proposal0(self, data):
        cdssm = self.cdssm
        if hasattr(cdssm.model_sde, 'optimal_proposal_dist'):
            return cdssm.model_sde.optimal_proposal_dist(cdssm.s(0), cdssm.s(1), cdssm.x0, data[0], cdssm.LY(0), cdssm.sigmaY(0))
        else:
            return StateSpaceModel.proposal0(self, data) # Not implemented

    def proposal(self, t, xp, data):
        cdssm = self.cdssm
        if hasattr(cdssm.model_sde, 'optimal_proposal_dist'):
            return cdssm.model_sde.optimal_proposal_dist(cdssm.s(t), cdssm.s(t+1), xp, data[t], cdssm.LY(t), cdssm.sigmaY(t))
        else: 
            return StateSpaceModel.proposal(self, data) # Not implemented

class MvDiscreteLinearGauss(MVLinearGauss, DiscreteDiscreteSSM):
    r"""Multivariate time homogeneous linear Gaussian model, 
    constructed from a CDSSM with a multivariate linear SDE,
    and a linear, Gaussian observation density.
    
    Kalman filtering/smoothing is supported for this class.
    
    Currently, having drift in the latent SDE is not supported,
    as the MvLinearGauss class in the 'particles' package does not
    support this. 

    .. math::
        X_0 & \sim N(\mu_0, cov_0) \\
        X_t & = (F * (X_{t-1} + b) + U_t, \quad   U_t\sim N(0, cov_X) \\
        Y_t & = G * X_t + V_t,     \quad   V_t \sim N(0, cov_Y)
                        
    The only mandatory parameter is `covX`(from which the
    dimension of X_t is deduced). The
    default values for the other parameters are:

    * `mu0` : An array of zeros (of size dx)
    * `b` : An array of zeros (of size dx)
    * `cov0`: cov_X
    * `F` : Identity matrix of shape (dx, dx)
    
    Needs to be subclassed to define an observation density.
    """
    
    def __init__(self, cdssm, **kwargs):
        # Assign the CDSSM as an attribute to the class
        DiscreteDiscreteSSM.__init__(self, cdssm, **kwargs)
        # Check compatibility of the CDSSM
        if not isinstance(self.cdssm.model_sde, LinearSDE): #or not isinstance(self.cdssm, GaussianCDSSM):
            raise ValueError(f"{self.__class__.__name__} only works with Linear SDEs and Gaussian (time-homogeneous) CDSSMs")
        # Extract the parameters of the MVLinearGauss class from the CDSSM
        linear_gauss_params = self.linear_gauss_params()
        # Ensure that the drift is zero
        b = linear_gauss_params.pop('b')
        if not np.all(np.isclose(b, np.zeros_like(b))):
            raise ValueError(f"{self.__class__.__name__} only works with zero drift")
        MVLinearGauss.__init__(self, **linear_gauss_params)

    def linear_gauss_params(self):
        s0, s1 = self.cdssm.s(0), self.cdssm.s(1)
        F = self.cdssm.model_sde._a(s0, s1)[0]; covX = self.cdssm.model_sde._v(s0, s1)[0] # (dimX, dimX), (dimX, dimX)
        b = self.cdssm.model_sde._b(s0, s1)[0] # (dimX,)
        G = self.cdssm.LY(0) # (dimY, dimX)
        covY = self.cdssm.CovY(0)   # (dimY, dimY)
        mvlg_params = {'F': F,
            'covX': covX,
            'mu0': np.dot(F, self.cdssm.x0[0]) + b, # (dimX,)
            'cov0': covX,
            'b': b,
            'G': G,
            'covY': covY
            }
        return mvlg_params
    
    @property
    def params(self):
        params = ['F', 'covX', 'mu0', 'cov0', 'G', 'covY']
        return {p: getattr(self, p) for p in params}

    def upper_bound_log_pt(self, t):
        return -0.5 * self.cdssm.dimX * np.log(2 * np.pi) - 0.5 * np.log(nla.det(self.covX))
    
class DiscreteLinearGauss(LinearGauss, MvDiscreteLinearGauss):
    r"""
    A (univariate) latent linear Gaussian model.
    Latent process follows an AR(1).
    Observation density may be non-linear.
    .. math::
        X_0                 & \sim N(\mu_0, \sigma_0^2) \\
        X_t|X_{t-1}=x_{t-1} & \sim N(\rho X_{t-1} + b),\sigma_X^2) \\
        Y_t |X_t=x_t        & \sim N(x_t, \sigma_Y^2)

    Needs to be subclassed to define an observation density.
    """

    def __init__(self, cdssm, **kwargs):
        assert cdssm.dimX == 1 and cdssm.dimY == 1, "Discrete class only works for univariate state and obs models"
        MvDiscreteLinearGauss.__init__(self, cdssm, **kwargs)

    def linear_gauss_params(self):
        s0, s1 = self.s(0), self.s(1)
        rho = self.model_sde._a(s0, s1); b = self.model_sde._b(s0, s1)
        sigmaX = np.sqrt(self.model_sde._v(s0, s1))
        sigmaY = self.cdssm.sigmaY(0)
        if self.LY(0) != 1.:
            raise ValueError(f"{self.__class__.__name__} only works when Y_t |X_t=x_t is N(Lx_t, \sigma_Y^2), where L=1.")
        lg_params = {'rho': rho,
                    'b': b,
                    'sigmaX': sigmaX, # Assume equidistant observations for now
                    'sigma0': sigmaX,
                    'mu0': rho*self.x0 + b,
                    'sigmaY': sigmaY
                    }
        return lg_params
        
    @property
    def params(self):
        params = ['rho', 'sigmaX', 'sigma0', 'mu0', 'sigmaY']
        return {p: getattr(self, p) for p in params}
    
    def upper_bound_log_pt(self, t):
        return -0.5 * np.log(2 * np.pi) - np.log(self.sigmaX)

# --------------- Standard State Space Model representations of LGSSMs for numerical experiment benchmarks ----------------

class ReparamLinearGauss(LinearGauss):
    
    default_params = {'sigmaX_2': 0.2,
                      'rho': 0.9,
                      }
    """
    A LGSSM that has been reparameterised to enable conjugacy results
    to be used in joint inference. This is a nice example of how to
    reparameterise a state space model.
    """

    def __init__(self, **kwargs):
        StateSpaceModel.__init__(self, **kwargs)
        orig_params = self.params_map(**{**self.default_params, **kwargs})
        super().__init__(**orig_params)

    def params_map(self, sigmaX_2, rho):
        """
        """
        orig_params = {'sigma0': np.sqrt(sigmaX_2), 'sigmaX': np.sqrt(sigmaX_2), 'rho': rho, 'sigmaY': 0.01}
        return orig_params
    
    @classmethod
    def prior(cls, alpha_X=3., beta_X=1., lmda=1., mu=0.): #alpha_Y=3., beta_Y=1.):
        """
        Constructs a prior distribution for this state space model.
        This prior distribution is conjugate for this model.         
        """
        local_vars = locals()
        hyperparams = {k: local_vars[k] for k in ["alpha_X", "beta_X", "lmda", "mu"]}
        lgssm_prior_dict = OrderedDict()
        lgssm_prior_dict['sigmaX_2'] = dists.InvGamma(a=alpha_X, b=beta_X)
        lgssm_prior_dict['rho'] = dists.Cond(lambda theta: dists.Normal(loc=mu, scale=np.sqrt(theta['sigmaX_2']/lmda)))
        # lgssm_prior_dict['sigmaY_2'] = dists.Cond(lambda theta: dists.InvGamma(a=alpha_Y, b=beta_Y))
        prior = dists.StructDist(lgssm_prior_dict)
        prior.hyperparams = hyperparams
        return prior

    @classmethod
    def posterior(cls, x, y, alpha_X=3., beta_X=1., lmda=1., mu=0.): # , alpha_Y=3., beta_Y=3.):
        """
        Constructs a posterior distribution for the parameters of this 
        state space model, given prior parameters and data x, y.
        """
        T = len(x)
        a = np.sum(x[:-1]*x[:-1]) + lmda
        b = lmda*mu + np.sum(x[:-1]*x[1:])
        c = np.sum(x*x) + 2*beta_X + lmda * (mu ** 2)
        post_params = {}
        post_params['alpha_X'] = 0.5*T + alpha_X
        post_params['beta_X'] = 0.5*(c - b*b/a)
        post_params['mu'] = b/a
        post_params['lmda'] = a
        # post_params['alpha_Y'] = 0.5*T + alpha_Y
        # post_params['beta_Y'] = beta_Y + 0.5*np.sum((x-y)*(x-y))
        return cls.prior(**post_params)

class TVLinearGauss(StateSpaceModel):

    default_params = {"sigmaY": 0.2, "sigma_YT": 0.1, "rho": 0.9, "sigmaX": 1.0, "sigma0": None}

    def PX0(self):
        return dists.Normal(scale=self.sigma0)

    def PX(self, t, xp):
        return dists.Normal(loc=self.rho * xp, scale=self.sigmaX)

    def PY(self, t, xp, x):
        if t < self.T - 1:
            return dists.Normal(loc=x, scale=self.sigmaY)
        else:
            return dists.Normal(loc=x, scale=self.sigmaYT)

    def proposal0(self, data):
        sig2post = 1.0 / (1.0 / self.sigma0 ** 2 + 1.0 / self.sigmaY ** 2)
        mupost = sig2post * (data[0] / self.sigmaY ** 2)
        return dists.Normal(loc=mupost, scale=np.sqrt(sig2post))

    def proposal(self, t, xp, data):
        if t < self.T - 1:
            sig2post = 1.0 / (1.0 / self.sigmaX ** 2 + 1.0 / self.sigmaY ** 2)
            mupost = sig2post * (
                self.rho * xp / self.sigmaX ** 2 + data[t] / self.sigmaY ** 2
            )
        else:
            sig2post = 1.0 / (1.0 / self.sigmaX ** 2 + 1.0 / self.sigmaYT ** 2)
            mupost = sig2post * (
                self.rho * xp / self.sigmaX ** 2 + data[t] / self.sigmaYT ** 2
            )
        return dists.Normal(loc=mupost, scale=np.sqrt(sig2post))

    def simulate(self, T):
        self.T = T
        x, y = super().simulate(T)
        return x, y


class TVBootstrap(Bootstrap):

    def __init__(self, ssm=None, data=None):
        super().__init__(ssm, data)
        self.ssm.T = self.T

class TVGuidedPF(GuidedPF):

    def __init__(self, ssm=None, data=None):
        super().__init__(ssm, data)
        self.ssm.T = self.T

class TVAuxiliary(AuxiliaryPF):

    def __init__(self, ssm=None, data=None):
        super().__init__(ssm, data)
        self.ssm.T = self.T