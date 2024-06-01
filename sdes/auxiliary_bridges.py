"""
Bridges module:
Within this module, we will implement abstractions of the different choices 
of auxiliary bridges that can be used to construct an invertible mapping from the bridge 
of an SDE conditional on the starting point and the end point, to a sample path
which has distribution that is absolutely integrable with respect to the parameter
free Weiner measure.

The original bridge construction given in Yonekura and Beskos (2022) is the auxiliary
bridge process of Delyon and Hu (2006). Additional choices of auxiliary bridge process
proposed in Schauer, van der Meulen and Van Zanten (2017) along with proofs of their
equivalence to 
"""
import numpy as np
import scipy.stats as stats
from sdes.sdes import SDE, BrownianMotion, OrnsteinUhlenbeck
from sdes.numerical_schemes import EulerMaruyama
from sdes.tools import start_points_paths_to_array
from sdes.path_integrals import log_girsanov


class ForwardProposal(SDE):
    """
    Proposal SDE based on the Forward decomposition. Continuous-time likelihood between this proposal and 
    the signal is given by the Girsanov formula.

    The Forward Proposal SDE is only used for simulation, not transformation, and are in general not linear SDEs.
    Thus, for this class we are only interested in the 'simulate' method.

    To construct a ForwardProposal, one needs to subclass and define:

    'LinearSDECls' as a class attribute
    'build_linear_sde' method 
    """

    def __init__(self, sde, t_start, t_end, y, eta_sq):
        self.SDE = sde
        self.t_start = t_start
        self.t_end = t_end
        self.LinearSDE = self.LinearSDECls()
        self.y = y
        self.eta_sq = eta_sq
        self.numerical_scheme = EulerMaruyama(self)

    @property
    def t_diff(self):
        return self.t_end - self.t_start

    def b(self, t, x):
        drift = self._b_time_shifted(t, x) 
        drift += self.Cov(t, x) * self.LinearSDE.grad_log_py(t, self.t_diff, x, self.y, self.eta_sq)
        return drift
    
    def sigma(self, t, x):
        return self.SDE.sigma(self.t_start + t, x)
    
    def _b_time_shifted(self, t, x):
        return self.SDE.b(self.t_start + t, x)

    def simulate(self, size: int, x_start, num=5) -> np.ndarray:
        self.build_linear_sde(x_start)
        return super().simulate(size, x_start, 0., self.t_diff, num)

    def log_girsanov(self, x_start: np.ndarray, X: np.ndarray):
        """
        Think about whether this function will still work with a drift that depends on the 
        start point: it should be fine!
        """
        step = float(X.dtype.names[0])
        X_array = start_points_paths_to_array(x_start, X)
        b_1 = self._b_time_shifted; b_2 = self.b; Cov = self.Cov
        log_girsanov_wgts = log_girsanov(X_array, b_1, b_2, Cov, step)
        return log_girsanov_wgts
    
    def build_linear_sde(self, x_start):
        raise NotImplementedError(self._error_msg('build_linear_sde'))

class BrownianProposal(ForwardProposal):

    LinearSDECls = BrownianMotion
    
class OUProposal(ForwardProposal):
    
    LinearSDECls = OrnsteinUhlenbeck

class NoDriftBasicBrownianProp(BrownianProposal):
    """
    Forward proposal that always takes the standard Brownian motion as the linear SDE
    for evaulation of the proxy. Not adaptive to the end points of previous particles.
    May perform poorly depending of the diffusive regime.

    $$dX_t = dW_t$$
    """
    def build_linear_sde(self, x_start):
        self.LinearSDE = self.LinearSDECls()
    
class NoDriftBrownianProp(BrownianProposal):
    """
    Forward proposal that always takes the Brownian motion as the linear SDE
    for evaulation of the proxy. Diffusion of each path is given by the 
    diffusion of the signal evaluated at the end points of the previous particles.
 

    $$dX_t = \sigma(t_start, x_start) dW_t$$
    """
    def build_linear_sde(self, x_start):
        drift = 0.
        diffusion = self.sigma(0, x_start)
        self.LinearSDE = self.LinearSDECls(m=drift, s=diffusion)
    
class DriftBrownianProp(BrownianProposal):
    """
    Forward proposal that always takes the Brownian motion as the linear SDE
    for evaulation of the proxy. Drift/diffusion constants are given by the 
    drift/diffusion of the signal evaluated at the end points of the previous particles.

    $$dX_t = b(t_start, x_start)dt + \sigma(t_start, x_start)dW_t$$
    """
    def build_linear_sde(self, x_start):
        drift = self.SDE.b(self.t_start, x_start)
        diffusion = self.SDE.sigma(self.t_start, x_start)
        self.LinearSDE = self.LinearSDECls(m=drift, s=diffusion)

class LocalLinearOUProp(OUProposal):
    """
    Forward proposal that takes the OU process as the linear SDE for evaluation of the proxy.
    Drift coefficient is obtained through local linearisation of the drift of the signal 
    about the end points of the previous particles. Diffusion coefficient is given by 
    the diffusion of the signal evaluated at hte end points of the previous particles.

    Note: if the signal process is an OU-process, this proposal recovers the same OU-process,
    thus we obtain the optimal proposal.

    Requires that first derivative of the drift is defined in the underlying SDE.

    $$dX_t = [A + BX_t]dt + CdW_t$$
    A = b(t_start, x_start) - db(t_start, x_start) * x_start
    B = db(t_start, x_start)
    C = \sigma(t_start, x_start)
    """
    def build_linear_sde(self, x_start):
        A = self.SDE.b(self.t_start, x_start) - self.SDE.db(self.t_start, x_start)*x_start
        B = self.SDE.db(self.t_start, x_start)
        C = self.SDE.sigma(self.t_start, x_start)
        _rho = -1.*B
        _mu = -A/B
        _phi = C
        self.LinearSDE = self.LinearSDECls(rho=_rho, mu=_mu, phi=_phi)

class AuxiliaryBridge(SDE):
    """
    Base class for auxiliary bridges.

    We use this class to construct any 1-D auxiliary bridge process.

    Given an SDE with certain drift and diffusion coefficient, a starting time, and end time, 
    a starting point and an ending point, this defines a diffusion bridge. It is not possible
    to simulate from this diffusion bridge, as the drift of the diffusion bridge involves the 
    transition density of the SDE, which is typically intractable. An auxiliary bridge process
    is a diffusion that starts and ends at the same points as the diffusion bridge, with known 
    drift, with law that dominates that of the diffusion bridge. Further, the continuous-time 
    likelihood can be evaluated up to discretisation (i.e all the terms inside the path integrals
    are tractable). 
    """
    def __init__(self, sde, t_start, t_end, x_end):
        self.SDE = sde
        self.x_end = x_end
        self.t_start = t_start
        self.t_end = t_end
        self.numerical_scheme = EulerMaruyama(self)

    @property
    def t_diff(self):
        return self.t_end - self.t_start

    @property
    def x_end_dim(self):
        if isinstance(self.x_end, float):
            return 1
        else:
            return self.x_end.shape[0]
    
    def b(self, t, x):
        raise NotImplementedError(self._error_msg('b'))
    
    def sigma(self, t, x):
        return self.SDE.sigma(self.t_start + t, x)
    
    def _b_time_shifted(self, t, x):
        return self.SDE.b(self.t_start + t, x)

    def bridge_log_likelihood(self, x_start, X):
        raise NotImplementedError(self._error_msg('bridge_log_likelihood'))
    
    def simulate(self, size, x_start, num=5):
        if size != self.x_end_dim and self.x_end_dim > 1:
            raise ValueError(f'Simulation size {size} should match dimension of end point vector ({self.x_end_dim}), unless a single end point is specified.')
        simulation = super().simulate(size, t_start=0., t_end=self.t_diff, x_start=x_start, num=num)
        end_point = simulation.dtype.names[-1]
        simulation[end_point] = np.ones(size) * self.x_end
        return simulation
    
    def transform_W_to_X(self, W, x_start):
        self._check_end_points_match(W)
        return self.SDE.numerical_scheme.transform_W_to_X(W, x_start=x_start, transform_end_point=False)
    
    def transform_X_to_W(self, X, x_start):
        self._check_end_points_match(X)
        return self.SDE.numerical_scheme.transform_X_to_W(X, x_start=x_start, transform_end_point=False)
    
    def _check_end_points_match(self, X):
        last_name = X.dtype.names[-1]
        if not np.all(np.isclose(X[last_name], self.x_end)):
            raise ValueError('End points of paths do not match end points of auxiliary bridge.')
        

class DelyonHuBridge(AuxiliaryBridge):
    """
    The auxiliary bridge as proposed by Delyon and Hu (2006).
    """
    def b(self, t, x):
        return (self.x_end - x)/(self.t_diff - t)
    
    def sigma(self, t, x):
        return self.SDE.sigma(t + self.t_start, x)
    
    # def bridge_log_likelihood(self, x_start, X):
    #     self._check_end_points_match(X)
    #     X_array = start_points_paths_to_array(x_start, X) # (N, num+1) array
    #     names = X.dtype.names; delta = float(names[1]) - float(names[0])
    #     times = self.t_start + np.array([0.] + [float(name) for name in names])
    #     b = self.SDE.b; sigma = self.SDE.sigma
    #     B = b(times, X_array)[:, :-1]

    #     B = b(times, array_X)[:, :-1]
    #     B_2 = b_2(times, array_X)[:, :-1] 
    #     Sigma = sigma(times, array_X)[:, :-1]
    #     dX_integral = (B_2 - B_1) * (array_X[:, 1:] - array_X[:, :-1]) / Sigma
    #     dt_integral = delta*(np.square(B_2) - np.square(B_1))/Sigma
    #     dX_integral = dX_integral.sum(axis=1)
    #     dt_integral = dt_integral.sum(axis=1)
    #     return dX_integral - 0.5 * dt_integral        


class VanDerMeulenSchauerBridge(AuxiliaryBridge):
    pass     




