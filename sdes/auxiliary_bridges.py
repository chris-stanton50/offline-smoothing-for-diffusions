"""
Auxiliary Bridges module:
--------------------------
Within this module, we implement abstractions of the different choices 
of auxiliary bridges that can be used as valid proposals for the true diffusion bridge

to construct an invertible mapping from the bridge 
of an SDE conditional on the starting point and the end point, to a sample path
which has distribution that is absolutely integrable with respect to the parameter
free Weiner measure.

The original bridge construction given in Yonekura and Beskos (2022) is the auxiliary
bridge process of Delyon and Hu (2006). Additional choices of auxiliary bridge process
proposed in Schauer, van der Meulen and Van Zanten (2017) along with proofs of their
equivalence to 


Univariate Case
-----------------

Forward Proposals: 3 Brownian Prposals, 1 OU Proposal
-----------------

NoDriftBasicBrownianProp: Forward proposal that always takes the standard Brownian motion as the linear SDE
NoDriftBrownianProp: Forward proposal that always takes the Brownian motion as the linear SDE
DriftBrownianProp: Forward proposal that always takes the Brownian motion as the linear SDE

LocalLinearOUProp: Forward proposal that takes the OU process as the linear SDE

Auxiliary Bridges: 2 Delyon Hu Bridges, 2 Brownian Aux Bridges, 1 OU Aux Bridge
-----------------
DelyonHuAuxBridge: The auxiliary bridge as proposed by Delyon and Hu (2006)
DriftDelyonHuAuxBridge: The Delyon-Hu bridge, with the drift added on. This will work fine for constant diffusion coefficients.

(subclasses of VanDerMeulenSchauerAuxBridge)

NoDriftBrownianAuxBridge: Auxiliary bridge proposal that always takes the Brownian motion as the linear SDE
DriftBrownianAuxBridge: Auxiliary bridge that always takes the Brownian motion as the linear SDE
LocalLinearOUAuxBridge: Auxiliary bridge that takes the OU process as the linear SDE


Multivariate Case
-----------------

Forward Proposals: 5 Brownian Proposals, 3 OU Proposals
-----------------
MvNoDriftBasicBrownianProp: Forward proposal that always takes the standard Brownian motion as the linear SDE
MvNoDriftIndepBrownianProp: Forward proposal that always takes the standard Brownian motion as the linear SDE
MvNoDriftBrownianProp: Forward proposal that always takes the Brownian motion as the linear SDE
MvDriftIndepBrownianProp: Forward proposal that always takes the Brownian motion as the linear SDE
MvDriftBrownianProp: Forward proposal that always takes the Brownian motion as the linear SDE

MvFullOUProposal: Forward proposal that takes the OU process as the linear SDE # Not fully implemented
MvOUProposal: Forward proposal that takes the OU process as the linear SDE
MvIndepOUProposal: Forward proposal that takes the OU process as the linear SDE

Auxiliary Bridges: 2 Delyon Hu bridges, 2 Brownian Aux Bridges, 2 OU Aux Bridges
-----------------

MvDelyonHuAuxBridge: The auxiliary bridge as proposed by Delyon and Hu (2006)
MvDriftDelyonHuAuxBridge: The Delyon-Hu bridge, with the drift added on. This will work fine for constant diffusion coefficients.

(subclasses of MvVanDerMeulenSchauerAuxBridge)
MvNoDriftBrownianAuxBridge: Auxiliary bridge proposal that always takes the Brownian motion as the linear SDE
MvDriftBrownianAuxBridge: Auxiliary bridge that always takes the Brownian motion as the linear SDE

MvFullOUAuxBridge: Auxiliary bridge that takes a full OU process as the linear SDE. # Not fully implemented
MvOUAuxBridge: Auxiliary bridge that takes the OU process as the linear SDE
"""
import numpy as np
import numpy.linalg as nla
import scipy.stats as stats
from sdes.sdes import SDE, MvEllipticSDE, BrownianMotion, OrnsteinUhlenbeck, MvIndepBrownianMotion, MvBrownianMotion, MvIndepOrnsteinUhlenbeck, MvOrnsteinUhlenbeck, MvFullOrnsteinUhlenbeck, TimeSwitchingSDE
from sdes.path_integrals import log_girsanov, log_delyon_hu, log_drift_delyon_hu, log_van_der_meulen_schauer, mv_log_girsanov, mv_log_delyon_hu, mv_log_van_der_meulen_schauer
from sdes.tools import log_abs_det
from particles.distributions import VaryingCovNormal

tol=1e-7

def add_tol(t_switch, func):
    def tol_func(*args, **kwargs):
        new_args = list(args)
        t = new_args[0] # Assume that t is the first argument
        if t == t_switch:
            new_args[0] -= tol
        return func(*new_args, **kwargs)
    return tol_func
        
class TolDecorators:
    
    def _sde_tol_dec(self, func):
        cond = isinstance(self.SDE, ForwardProposal)
        sde = self.SDE.SDE if cond else self.SDE
        if isinstance(sde, TimeSwitchingSDE):
            if not cond:
                return add_tol(sde.t_switch, func)
            elif cond and self.SDE.t_end == sde.t_switch:
                return add_tol(self.t_end, func)
            else:
                return func 
        else:
            return func

    def _self_tol_dec(self, func):
        if isinstance(self.SDE, TimeSwitchingSDE) and self.t_end == self.SDE.t_switch:
            return add_tol(self.t_diff, func)
        else:
            return func
# -----------------Univariate Forward Proposals-----------------

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

    def __init__(self, sde, t_start, t_end, y, LY, sigmaY):
        self.SDE = sde
        self.t_start = t_start
        self.t_end = t_end
        self.y = y
        self.LY = LY
        self.sigmaY = sigmaY
        self.numerical_scheme = self.numerical_scheme_cls(self)
            
    @property
    def t_diff(self):
        return self.t_end - self.t_start

    def b(self, t, x):
        drift = self._b_time_shifted(t, x) 
        drift += self.Cov(t, x) * self.LinearSDE.grad_log_py(t, self.t_diff, x, self.y, self.LY, self.sigmaY)
        return drift

    def sigma(self, t, x):
        return self.SDE.sigma(self.t_start + t, x)

    def db(self, t, x):
        db = self.SDE.db(self.t_start + t, x) 
        db += self.SDE.dCov(self.t_start + t, x)*self.LinearSDE.grad_log_py(t, self.t_diff, x, self.y, self.LY, self.sigmaY)
        db += self.Cov(t, x) * self.LinearSDE.grad_grad_log_py(t, self.t_diff, self.y, self.LY, self.sigmaY)
        return db

    def dsigma(self, t, x):
        return self.SDE.dsigma(self.t_start + t, x)
    
    def _b_time_shifted(self, t, x):
        return self.SDE.b(self.t_start + t, x)

    def simulate(self, size: int, x_start, num=5) -> np.ndarray:
        self.build_linear_sde(x_start) # Check the integral and the simulation to ensure this works.
        return super().simulate(size, x_start, 0., self.t_diff, num)

    def end_point_proposal(self, x_start):
        self.build_linear_sde(x_start)
        return self.LinearSDE.optimal_proposal_dist(self.t_start, self.t_end, x_start, self.y, self.LY, self.sigmaY)

    def b_vec(self, t, x):
        drift = self._b_time_shifted(t, x) 
        drift += self.Cov(t, x) * self.LinearSDE._vec_grad_log_py(t, self.t_diff, x, self.y, self.LY, self.sigmaY)
        return drift

    def log_girsanov(self, x_start: np.ndarray, X: np.ndarray):
        """
        This function has been tested using the O-U process, and it works!

        However, the proposals must be chosen such that they don't depend on
        the start point (i.e the parameters cannot be defined as vectors
        within the LinearSDE class)
        """
        names = X.dtype.names
        step = float(names[0])
        X_array = np.stack([x_start] + [X[name] for name in names], axis=1) # (N, num+1)
        b_1 = self._b_time_shifted; b_2 = self.b_vec; Cov = self.Cov
        log_girsanov_wgts = log_girsanov(X_array, b_1, b_2, Cov, step)
        return log_girsanov_wgts
    
    def build_linear_sde(self, x_start: float):
        """
        IMPORTANT: For now, the input to this function MUST 
        be a float, and not a vector. Thus, the drift and diffusion 
        coefficient of the proposal SDE will be the same, REGARDLESS
        of the starting point. The difference in starting point will be the only 
        place where the proposal distribution differs due to different starting points.
        """
        raise NotImplementedError(self._error_msg('build_linear_sde'))

class BrownianProposal(ForwardProposal):

    LinearSDECls = BrownianMotion
    
class OUProposal(ForwardProposal):
    
    LinearSDECls = OrnsteinUhlenbeck

# -----------------Brownian Univariate Forward Proposals - 3 Classes -----------------

class NoDriftBasicBrownianProp(BrownianProposal):
    """
    Forward proposal that always takes the standard Brownian motion as the linear SDE
    for evaulation of the proxy. Not adaptive to the end points of previous particles.
    May perform poorly depending of the diffusive regime.

    $$dX_t = dW_t$$
    """
    sname='NDBBrP'
    def build_linear_sde(self, x_start):
        self.LinearSDE = self.LinearSDECls()
    
class NoDriftBrownianProp(BrownianProposal):
    """
    Forward proposal that always takes the Brownian motion as the linear SDE
    for evaulation of the proxy. Diffusion of each path is given by the 
    diffusion of the signal evaluated at the end points of the previous particles.
    $$dX_t = \sigma(t_start, x_start) dW_t$$
    """
    sname='NDBrP'
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
    sname='DBrP'
    def build_linear_sde(self, x_start):
        drift = self.SDE.b(self.t_start, x_start)
        diffusion = self.SDE.sigma(self.t_start, x_start)
        self.LinearSDE = self.LinearSDECls(m=drift, s=diffusion)

# -----------------OU Univariate Forward Proposals - 1 Class -----------------

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
    sname='OUP'
    def build_linear_sde(self, x_start):
        A = self.SDE.b(self.t_start, x_start) - self.SDE.db(self.t_start, x_start)*x_start
        B = self.SDE.db(self.t_start, x_start)
        C = self.SDE.sigma(self.t_start, x_start)
        _rho = -1.*B
        _mu = -A/B
        _phi = C
        self.LinearSDE = self.LinearSDECls(rho=_rho, mu=_mu, phi=_phi)

# -----------------Univariate Diffusion Bridge Proposals-----------------

class AuxiliaryBridge(SDE, TolDecorators):
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
        self.numerical_scheme = self.numerical_scheme_cls(self)
    
    @property
    def t_diff(self):
        return self.t_end - self.t_start

    @property
    def N(self):
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

    def _b_vec_time_shifted(self, t, x):
        return self.SDE.b_vec(self.t_start + t, x)

    def bridge_log_likelihood(self, x_start, X):
        raise NotImplementedError(self._error_msg('bridge_log_likelihood'))
    
    def simulate(self, size, x_start, num=5):
        if size != self.N and self.N > 1:
            raise ValueError(f'Simulation size {size} should match dimension of end point vector ({self.N}), unless a single end point is specified.')
        simulation = super().simulate(size, t_start=0., t_end=self.t_diff, x_start=x_start, num=num)
        end_point = simulation.dtype.names[-1]
        simulation[end_point] = np.ones(self.N)*self.x_end if type(self.x_end) == float else self.x_end
        return simulation
    
    def transform_W_to_X(self, W, x_start):
        self._check_end_points_match(W)
        return self.numerical_scheme.transform_W_to_X(W, 0., x_start=x_start, transform_end_point=False)
    
    def transform_X_to_W(self, X, x_start):
        self._check_end_points_match(X)
        return self.numerical_scheme.transform_X_to_W(X, 0., x_start=x_start, transform_end_point=False)
    
    def _check_end_points_match(self, X):
        last_name = X.dtype.names[-1]
        if not np.all(np.isclose(X[last_name], self.x_end)):
            raise ValueError('End points of paths do not match end points of auxiliary bridge.')
        
class DelyonHuAuxBridge(AuxiliaryBridge):
    """
    The auxiliary bridge as proposed by Delyon and Hu (2006).
    """
    
    sname='DH'
    
    def b(self, t, x):
        return (self.x_end - x)/(self.t_diff - t)

    def bridge_log_likelihood(self, x_start, X):
        self._check_end_points_match(X)
        b = self._b_vec_time_shifted if hasattr(self.SDE, "b_vec") else self._b_time_shifted
        Cov = self.Cov
        t_end = X.dtype.names[-1]; x_end = X[t_end] # (N, )
        X_array = np.stack([x_start] + [X[name] for name in X.dtype.names], axis=1) # (N, num+1)
        step = float(X.dtype.names[0]); num = X_array.shape[1] - 1; Delta_s = step*num
        log_density = stats.norm.logpdf(x_end, loc=x_start, scale=np.sqrt(Delta_s * self.Cov(0, x_start)))
        log_det_covs = 0.5*(np.log(Cov(0, x_start)) - np.log(Cov(Delta_s, x_end))) # (N, )
        # The path integrals
        log_path_integral_wgts = log_delyon_hu(X_array, b, Cov, step) # (N, )
        log_wgts = log_density + log_det_covs + log_path_integral_wgts
        return log_wgts

class DriftDelyonHuAuxBridge(AuxiliaryBridge):
    """
    The Delyon-Hu bridge, with the drift added on. This will work fine for constant diffusion coefficients.
    """
    sname='DDH'
    def b(self, t, x):
        return self._b_time_shifted(t, x) + (self.x_end - x)/(self.t_diff - t)
    
    def bridge_log_likelihood(self, x_start, X):
        self._check_end_points_match(X)
        b = self._b_vec_time_shifted if hasattr(self.SDE, "b_vec") else self._b_time_shifted
        Cov = self.Cov
        t_end = X.dtype.names[-1]; x_end = X[t_end] # (N, )
        X_array = np.stack([x_start] + [X[name] for name in X.dtype.names], axis=1) # (N, num+1)
        step = float(X.dtype.names[0]); num = X_array.shape[1] - 1; Delta_s = step*num
        log_density = stats.norm.logpdf(x_end, loc=x_start, scale=np.sqrt(Delta_s * self.Cov(0, x_start)))
        log_det_covs = 0.5*(np.log(Cov(0, x_start)) - np.log(Cov(Delta_s, x_end)))
        # The path integrals
        log_path_integral_wgts = log_drift_delyon_hu(X_array, b, Cov, step) # (N, )
        log_wgts = log_density + log_det_covs + log_path_integral_wgts
        return log_wgts
    
class VanDerMeulenSchauerAuxBridge(AuxiliaryBridge):
    """
    The class of guided bridge proposals based on Linear SDEs:
    """
    def __init__(self, sde, t_start, t_end, x_end):
        super().__init__(sde, t_start, t_end, x_end)
        self.build_linear_sde()

    def b(self, t, x):
        drift = self._b_time_shifted(t, x) 
        drift += self.Cov(t, x) * self.LinearSDE.grad_log_px(t, self.t_diff, x, self.x_end)
        return drift
    
    def bridge_log_likelihood(self, x_start, X):
        self._check_end_points_match(X)
        b = self._b_vec_time_shifted if hasattr(self.SDE, "b_vec") else self._b_time_shifted
        Cov = self.Cov; linear_sde = self.LinearSDE
        X_array = np.stack([x_start] + [X[name] for name in X.dtype.names], axis=1) # (N, num+1)
        step = float(X.dtype.names[0])
        linear_sde_transition_dist = linear_sde.transition_dist(0., self.t_diff, x_start)
        log_linear_sde_density = linear_sde_transition_dist.logpdf(self.x_end)
        # The path integrals
        log_path_integral_wgts = log_van_der_meulen_schauer(X_array, b, Cov, linear_sde, step) # (N, )
        log_wgts = log_linear_sde_density + log_path_integral_wgts
        return log_wgts

    def build_linear_sde(self):
        raise NotImplementedError(self._error_msg('build_linear_sde'))

class BrownianAuxBridge(VanDerMeulenSchauerAuxBridge):
    LinearSDECls = BrownianMotion

class OUAuxBridge(VanDerMeulenSchauerAuxBridge):
    LinearSDECls = OrnsteinUhlenbeck

# -----------------Brownian Univariate Auxiliary Bridges - 2 Classes -----------------

class NoDriftBrownianAuxBridge(BrownianAuxBridge):
    """
    Auxiliary bridge proposal that always takes the Brownian motion as the linear SDE
    for evaulation of the proxy. 

    $$dX_t = \sigma(t_end, x_end) dW_t$$
    """
    sname='NDBr'
    def build_linear_sde(self):
        drift = 0.
        diffusion = self.SDE.sigma(self.t_end-tol, self.x_end)
        self.LinearSDE = self.LinearSDECls(m=drift, s=diffusion)
    
class DriftBrownianAuxBridge(BrownianAuxBridge):
    """
    Auxiliary bridge that always takes the Brownian motion as the linear SDE
    for evaulation of the proxy. Diffusion of each path is given by the 
    diffusion of the signal evaluated at the end points of the previous particles.
 
    $$dX_t = b(t_end, x_end)dt + \sigma(t_end, x_end) dW_t$$
    """
    sname='DBr'
    def build_linear_sde(self):
        drift = self.SDE.b(self.t_end, self.x_end)
        diffusion = self.SDE.sigma(self.t_end-tol, self.x_end)
        self.LinearSDE = self.LinearSDECls(m=drift, s=diffusion)

# -----------------OU Univariate Auxiliary Bridges - 1 Class -----------------

class LocalLinearOUAuxBridge(OUAuxBridge):
    """
    Auxiliary bridge that takes the OU process as the linear SDE for evaluation of the proxy.
    Drift coefficient is obtained through local linearisation of the drift of the signal 
    about the end points of the previous particles. Diffusion coefficient is given by 
    the diffusion of the signal evaluated at hte end points of the previous particles.

    Note: if the signal process is an OU-process, this proposal recovers the same OU-process,
    thus we obtain the optimal proposal.

    Requires that first derivative of the drift is defined in the underlying SDE.

    $$dX_t = [A + BX_t]dt + CdW_t$$
    A = b(t_end, x_end) - db(t_end, x_end) * x_end
    B = db(t_end, x_end)
    C = \sigma(t_end, x_end)
    """
    sname='OU'
    def build_linear_sde(self):
        A = self.SDE.b(self.t_end, self.x_end) - self.SDE.db(self.t_end, self.x_end)*self.x_end
        B = self.SDE.db(self.t_end, self.x_end)
        C = self.SDE.sigma(self.t_end-tol, self.x_end)
        _rho = -1.*B
        _mu = -A/B
        _phi = C
        self.LinearSDE = self.LinearSDECls(rho=_rho, mu=_mu, phi=_phi)

# ----------------- Multivariate Forward Proposals-----------------

class MvForwardProposal(MvEllipticSDE, ForwardProposal):

    """
    Proposal SDE based on the Forward decomposition. Continuous-time likelihood between this proposal and 
    the signal is given by the Girsanov formula.

    The Forward Proposal SDE is only used for simulation, not transformation, and are in general not linear SDEs.
    Thus, for this class we are only interested in the 'simulate' method.

    To construct a ForwardProposal, one needs to subclass and define:

    'LinearSDECls' as a class attribute
    'build_linear_sde' method 
    
    To do:  - Implement the log_girsanov method in the multivariate case
            - Think about how to implement 'build_linear_sde' for the multivariate case.
    """    

    def __init__(self, sde, t_start, t_end, y, LY, sigmaY):
        ForwardProposal.__init__(self, sde, t_start, t_end, y, LY, sigmaY)
        if not isinstance(self.SDE, MvEllipticSDE):
            raise ValueError('The underlying SDE must be an elliptic SDE for forward proposals.')

    @property
    def dimX(self):
        return self.SDE.dimX

    @property
    def N(self):
        return self.LinearSDE.N
        
    @property
    def _diag_cov(self):
        return self.SDE._diag_cov

    def b(self, t, x):
        """
        Input: float, (N, dimX)
        Returns: (N, dimX)
        """
        drift = self._b_time_shifted(t, x) # Inherited from ForwardProposal
        drift += np.einsum('ijk,ik->ij', self.Cov(t, x), self.LinearSDE.grad_log_py(t, self.t_diff, x, self.y, self.LY, self.sigmaY)) # (N, dimX)
        return drift

    def sigma(self, t, x):
        """
        Input: (float, (N, dimX))
        Returns: (N, dimX, dimX)
        """
        return self.SDE.sigma(self.t_start + t, x)

    def db(self, t, x): 
    # Used to construct a linear SDE for an auxiliary bridge for a forward proposal. 
    # Come back to this an implement if you really need it.
        db = self.SDE.db(self.t_start + t, x) # (N, dimX, dimX)
        # For now, we omit the general case where the diffusion coefficient is state-dependent.
        # db += 2.*self.SDE.dsigma(self.t_start + t, x)*self.sigma(t, x)*self.LinearSDE.grad_log_py(t, self.t_diff, x, self.y, self.LY, self.sigmaY)
        # db += self.Cov(t, x) * self.LinearSDE.grad_grad_log_py(t, self.t_diff, self.y, self.LY, self.sigmaY)
        return db

    def dsigma(self, t, x):
        """
        Inputs: float, (N, dimX)
        Returns: float, (N, dimX, dimX, dimX)
        """
        return self.SDE.dsigma(self.t_start + t, x)
    
    def simulate(self, size: int, x_start, num=5) -> np.ndarray:
        return super().simulate(size, x_start, num)

    def end_point_proposal(self, x_start):
        self.build_linear_sde(x_start)
        return self.LinearSDE.optimal_proposal_dist(self.t_start, self.t_end, x_start, self.y, self.LY, self.sigmaY)

    def log_girsanov(self, x_start: np.ndarray, X: np.ndarray):
        """
        Inputs: 
        ------------
        x_start: (N, dimX) array
        X: structured array with fields '0.0', '0.1', ..., '1.0' (N, num)
        
        Returns:
        ------------
        (N, ) Array of weights
        """
        step = float(X.dtype.names[0])
        X_array = np.stack([x_start] + [X[name] for name in X.dtype.names], axis=0) # (num+1, N, dimX)
        b_1 = self._b_time_shifted; b_2 = self.b; Cov = self.Cov
        log_girsanov_wgts = mv_log_girsanov(X_array, b_1, b_2, Cov, step)
        return log_girsanov_wgts
    
    def build_linear_sde(self, x_start: float):
        """
        Inputs: x_start: (N, dimX)
        """
        raise NotImplementedError(self._error_msg('build_linear_sde'))

class MvIndepBrownianProposal(MvForwardProposal):
    LinearSDECls = MvIndepBrownianMotion
    
class MvBrownianProposal(MvForwardProposal):
    LinearSDECls = MvBrownianMotion

#----------------- Brownian Multivariate Forward Proposals: 5 Classes-----------------    

class MvNoDriftBasicBrownianProp(MvIndepBrownianProposal):
    """
    Forward proposal that always takes the standard Brownian motion as the linear SDE
    for evaulation of the proxy. Not adaptive to the end points of previous particles.
    May perform poorly depending of the diffusive regime.

    $$dX_t = dW_t$$
    """
    sname = 'MvNDBBrP'
    
    def build_linear_sde(self, x_start):
        N, dimX = x_start.shape
        m = np.zeros(x_start.shape); s = np.ones(x_start.shape)
        self.LinearSDE = self.LinearSDECls(N=N, dimX=dimX, m=m, s=s)
    
class MvNoDriftIndepBrownianProp(MvIndepBrownianProposal):
    """
    Forward proposal that always takes the Brownian motion as the linear SDE
    for evaulation of the proxy. Diffusion of each path is given by the 
    diffusion of the signal evaluated at the end points of the previous particles.
    
    Only the diagonal elements of the diffusion matrix are used.
 
    $$dX_t = \sigma(t_start, x_start) dW_t$$
    """
    sname = 'MvNDIBrP'
        
    def build_linear_sde(self, x_start):
        N, dimX = x_start.shape
        m = np.zeros(x_start.shape)
        sigma = self.SDE.sigma
        s_mat = sigma(self.t_start, x_start)
        s = np.diag(s_mat[0]).reshape(N, dimX) if N == 1 else np.stack([np.diag(s_mat[i]) for i in range(N)], axis=0)
        self.LinearSDE = self.LinearSDECls(N=N, dimX=dimX, m=m, s=s)

class MvNoDriftBrownianProp(MvBrownianProposal):
    """
    Forward proposal that always takes the Brownian motion as the linear SDE
    for evaulation of the proxy. Diffusion of each path is given by the 
    diffusion of the signal evaluated at the end points of the previous particles.
    
    Allows for a full diffusion matrix.
 
    $$dX_t = \sigma(t_start, x_start) dW_t$$
    """
    sname = 'MvNDBrP'
    
    def build_linear_sde(self, x_start):
        N, dimX = x_start.shape
        m = np.zeros(x_start.shape)
        sigma = self.SDE.sigma
        s = sigma(self.t_start, x_start)
        self.LinearSDE = self.LinearSDECls(N=N, dimX=dimX, m=m, s=s)

class MvDriftIndepBrownianProp(MvIndepBrownianProposal):
    """
    Forward proposal that always takes the Brownian motion as the linear SDE
    for evaulation of the proxy. Drift/diffusion constants are given by the 
    drift/diffusion of the signal evaluated at the end points of the previous particles.

    Only the diagonal elements of the diffusion matrix are used.
    
    $$dX_t = b(t_start, x_start)dt + \sigma(t_start, x_start)dW_t$$
    """
    sname = 'MvDIBrP'
        
    def build_linear_sde(self, x_start):
        N, dimX = x_start.shape
        m = self.SDE.b(self.t_start, x_start)
        sigma = self.SDE.sigma
        s_mat = sigma(self.t_start, x_start)
        s = np.diag(s_mat[0]).reshape(N, dimX) if N == 1 else np.stack([np.diag(s_mat[i]) for i in range(N)], axis=0)
        self.LinearSDE = self.LinearSDECls(N=N, dimX=dimX, m=m, s=s)
        
class MvDriftBrownianProp(MvBrownianProposal):
    """
    Forward proposal that always takes the Brownian motion as the linear SDE
    for evaulation of the proxy. Drift/diffusion constants are given by the 
    drift/diffusion of the signal evaluated at the end points of the previous particles.

    The full diffusion matrix is used.
        
    $$dX_t = b(t_start, x_start)dt + \sigma(t_start, x_start)dW_t$$
    """
    
    sname = 'MvDBrP'
    def build_linear_sde(self, x_start):
        N, dimX = x_start.shape
        m = self.SDE.b(self.t_start, x_start)
        sigma = self.SDE.sigma
        s_mat = sigma(self.t_start, x_start)
        s = s_mat[0] if N == 1 else s_mat
        self.LinearSDE = self.LinearSDECls(N=N, dimX=dimX, m=m, s=s)


#----------------- OU Multivariate Forward Proposals 3 Classes-----------------    


class MvFullOUProposal(MvForwardProposal):
    
    LinearSDECls = MvFullOrnsteinUhlenbeck
    
    """
    Forward proposal that takes the OU process as the linear SDE for evaluation of the proxy.
    Drift coefficient is obtained through local linearisation of the drift of the signal 
    about the end points of the previous particles. Diffusion coefficient is given by 
    the diffusion of the signal evaluated at the end points of the previous particles.

    Note: if the signal process is an OU-process, this proposal recovers the same OU-process,
    thus we obtain the optimal proposal.

    Requires that first derivative of the drift is defined in the underlying SDE.

    $$dX_t = [A + BX_t]dt + CdW_t$$
    A = b(t_start, x_start) - db(t_start, x_start) * x_start
    B = db(t_start, x_start)
    C = \sigma(t_start, x_start)
    """

    sname = 'MvFOUP'
            
    def _gen_full_ou_params(self, x_start):
        b = self.SDE.b; db = self.SDE.db; sigma = self.SDE.sigma
        A = b(self.t_start, x_start) - np.einsum('ijk,ik->ij', db(self.t_start, x_start), x_start) # (N, dimX)
        B = db(self.t_start, x_start) # (N, dimX, dimX)
        C = sigma(self.t_start, x_start) # (N, dimX, dimX)
        _rho = -1.*B
        _mu = nla.solve(-1.*B, A) # (N, dimX)
        _phi = C
        return _rho, _mu, _phi

    def build_linear_sde(self, x_start):
        N, dimX = x_start.shape
        _rho, _mu, _phi = self._gen_full_ou_params(x_start) # (N, dimX, dimX) (N, dimX), (N, dimX, dimX)
        self.LinearSDE = self.LinearSDECls(N=N, dimX=dimX, rho=_rho, mu=_mu, phi=_phi)

class MvOUProposal(MvFullOUProposal):
    
    LinearSDECls = MvOrnsteinUhlenbeck

    sname = 'MvOUP'
        
    def build_linear_sde(self, x_start):
        N, dimX = x_start.shape
        _rho, _mu, _phi = self._gen_full_ou_params(x_start) # (N, dimX, dimX) (N, dimX), (N, dimX, dimX)
        _rho = np.stack([np.diag(_rho[i]) for i in range(N)], axis=0) if N > 1 else np.diag(_rho[0]).reshape((N, dimX)) # (N, dimX): Only take diagonals
        _phi = _phi if N > 1 else _phi[0]
        self.LinearSDE = self.LinearSDECls(N=N, dimX=dimX, rho=_rho, mu=_mu, phi=_phi)
                          
class MvIndepOUProposal(MvFullOUProposal):

    LinearSDECls = MvIndepOrnsteinUhlenbeck
    
    sname = 'MvIOUP'
    
    def build_linear_sde(self, x_start):
        N, dimX = x_start.shape
        _rho, _mu, _phi = self._gen_full_ou_params(x_start)
        _phi = np.stack([np.diag(_phi[i]) for i in range(N)], axis=0) # (N, dimX): Only take diagonals
        _rho = np.stack([np.diag(_rho[i]) for i in range(N)], axis=0) # (N, dimX): Only take diagonals
        self.LinearSDE = self.LinearSDECls(N=N, dimX=dimX, rho=_rho, mu=_mu, phi=_phi)

# ----------------- Multivariate Diffusion Bridge Proposals-----------------

class MvEllipticAuxiliaryBridge(MvEllipticSDE, AuxiliaryBridge):

    def __init__(self, *args):
        AuxiliaryBridge.__init__(self, *args)
        if self.x_end.shape[1] != self.SDE.dimX:
            raise ValueError(f'Second dimension of end point array must match dimension of underlying SDE: {self.x_end.shape[1]}!={self.SDE.dimX}')

    @property
    def dimX(self):
        return self.SDE.dimX
    
    @property
    def N(self):
        return self.x_end.shape[0]
    
    def b(self, t, x):
        raise NotImplementedError(self._error_msg('b'))
    
    def sigma(self, t, x):
        return self.SDE.sigma(self.t_start + t, x)
    
    # def _b_vec_time_shifted(self, t, x): # This inherits from AuxiliaryBridge, but you need to implement b_vec for the underlying
    #     return self.SDE.b_vec(self.t_start + t, x)

    def bridge_log_likelihood(self, x_start, X):
        raise NotImplementedError(self._error_msg('bridge_log_likelihood'))
    
    def simulate(self, size, x_start, num=5):
        N = self.N; x_end = self.x_end
        if N == 1 and size != 1:
            x_end = np.concatenate([self.x_end]*size, axis=0) # (size, dimX)
        if size != N and N > 1:
            raise ValueError(f'Simulation size {size} should match number of end point vectors ({self.N}).')
        if x_start.shape not in  [(self.N, self.dimX), (1, self.dimX)] and N>1:
            raise ValueError(f'Starting point array shape {x_start.shape} should be (N, dimX) ({self.N}, {self.dimX}) or (1, dimX) for N>1: N={self.N}.')
        if x_start.shape == (1, self.dimX) and N > 1:
            x_start = np.concatenate([x_start]*N)
        simulation = super().simulate(size, x_start=x_start, num=num)
        end_point = simulation.dtype.names[-1]
        simulation[end_point] = x_end
        return simulation
    
    def transform_W_to_X(self, W, x_start):
        self._check_end_points_match(W)
        return self.numerical_scheme.transform_W_to_X(W, 0., x_start=x_start, transform_end_point=False)
    
    def transform_X_to_W(self, X, x_start):
        self._check_end_points_match(X)
        return self.numerical_scheme.transform_X_to_W(X, 0., x_start=x_start, transform_end_point=False)
    
    def _check_end_points_match(self, X):
        last_name = X.dtype.names[-1]
        if not np.all(np.isclose(X[last_name], self.x_end)):
            raise ValueError('End points of paths do not match end points of auxiliary bridge.')

class MvDelyonHuAuxBridge(MvEllipticAuxiliaryBridge, DelyonHuAuxBridge):
    """
    The auxiliary bridge as proposed by Delyon and Hu (2006).
    """
    sname = 'MvDH'
    def b(self, t, x):
        return (self.x_end - x)/(self.t_diff - t)

    def bridge_log_likelihood(self, x_start, X):
        self._check_end_points_match(X)
        tol_dec = self._self_tol_dec
        b = tol_dec(self._b_time_shifted); Cov = tol_dec(self.Cov)
        X_array = np.stack([x_start] + [X[name] for name in X.dtype.names], axis=0) # (num+1, N, dimX)
        step = float(X.dtype.names[0]); num = X_array.shape[0] - 1; Delta_s = step*num
        log_density = VaryingCovNormal(loc=x_start, cov=Delta_s*Cov(0, x_start)).logpdf(self.x_end) # (N, )
        log_det_covs = 0.5 * (log_abs_det(Cov(0, x_start)) - log_abs_det(Cov(Delta_s, self.x_end))) # (N, )
        # The path integrals
        log_path_integral_wgts = mv_log_delyon_hu(X_array, b, Cov, step) # (N, )
        log_wgts = log_density + log_det_covs + log_path_integral_wgts # (N, )
        return log_wgts
    
class MvDriftDelyonHuAuxBridge(MvEllipticAuxiliaryBridge, DriftDelyonHuAuxBridge):
    """
    The Delyon-Hu bridge, with the drift added on. This will work fine for constant diffusion coefficients.
    """
    
    sname = 'MvDDH'
    def b(self, t, x):
        return (self._b_time_shifted(t, x) + (self.x_end - x)/(self.t_diff - t))
    
    def bridge_log_likelihood(self, x_start, X):
        self._check_end_points_match(X)
        tol_dec = self._self_tol_dec
        b = tol_dec(self._b_time_shifted); Cov = tol_dec(self.Cov)
        X_array = np.stack([x_start] + [X[name] for name in X.dtype.names], axis=0) # (num+1, N, dimX)
        step = float(X.dtype.names[0]); num = X_array.shape[1] - 1; Delta_s = step*num
        log_density = VaryingCovNormal(loc=x_start, scale=Delta_s*Cov(0, x_start)).logpdf(self.x_end)
        log_det_covs = 0.5 * (log_abs_det(Cov(0, x_start)) - log_abs_det(Cov(Delta_s, self.x_end)))
        # The path integrals
        log_path_integral_wgts = mv_log_delyon_hu(X_array, b, Cov, step) # (N, )
        log_wgts = log_density + log_det_covs + log_path_integral_wgts
        return log_wgts

class MvVanDerMeulenSchauerAuxBridge(MvEllipticAuxiliaryBridge, VanDerMeulenSchauerAuxBridge):
    """
    The class of guided bridge proposals based on Linear SDEs:
    """
    def __init__(self, sde, t_start, t_end, x_end):
        VanDerMeulenSchauerAuxBridge.__init__(self, sde, t_start, t_end, x_end)

    def b(self, t, x):
        drift = self._b_time_shifted(t, x) # (N, dimX)
        drift += np.einsum('ijk,ik->ij', self.Cov(t, x), self.LinearSDE.grad_log_px(t, self.t_diff, x, self.x_end))
        return drift

    @property
    def sigma_x_end(self):
        """
        The value of \sigma(t, x_end). The 'matching condition' requires that the diffusion covariance of the underlying SDE at the end points
        must match the diffusion of the auxiliary bridge at the end points. For this to hold, it is sufficient for the diffusion coefficients to 
        match at the end points.
        """
        tol_dec = self._sde_tol_dec
        sigma = tol_dec(self.SDE.sigma)
        sigma_x_end = sigma(self.t_end, self.x_end) # (N, dimX, dimX)
        if self._diag_cov:
            sigma_x_end = np.stack([np.diag(sigma_x_end[i]) for i in range(self.N)], axis=0) if self.N > 1 else np.diag(sigma_x_end[0])
        return sigma_x_end
    
    @property
    def _diag_cov(self):
        if hasattr(self.SDE, '_diag_cov') and self.SDE._diag_cov:
            return True
        else:
            return False
        
    def bridge_log_likelihood(self, x_start, X):
        self._check_end_points_match(X)
        tol_dec = self._self_tol_dec
        b = tol_dec(self._b_time_shifted); Cov = tol_dec(self.Cov)
        linear_sde = self.LinearSDE; step = float(X.dtype.names[0])
        X_array = np.stack([x_start] + [X[name] for name in X.dtype.names], axis=0) # (num+1, N, dimX)
        linear_sde_transition_dist = linear_sde.transition_dist(0., self.t_diff, x_start)
        log_linear_sde_density = linear_sde_transition_dist.logpdf(self.x_end)
        # The path integrals
        log_path_integral_wgts = mv_log_van_der_meulen_schauer(X_array, b, Cov, linear_sde, step) # (N, )
        log_wgts = log_linear_sde_density + log_path_integral_wgts
        return log_wgts

    def build_linear_sde(self):
        raise NotImplementedError(self._error_msg('build_linear_sde'))
 
class MvBrownianAuxBridge(MvVanDerMeulenSchauerAuxBridge):
    LinearSDECls = MvBrownianMotion

class MvOUAuxBridge(MvVanDerMeulenSchauerAuxBridge):
    LinearSDECls = MvOrnsteinUhlenbeck
    
# ---------------Multivariate Brownian Auxiliary Bridge Proposals: 2 Classes ----------------

class MvNoDriftBrownianAuxBridge(MvVanDerMeulenSchauerAuxBridge, NoDriftBrownianAuxBridge):
    """
    Auxiliary bridge proposal that takes the Brownian motion as the linear SDE
    for evaulation of the proxy, without the drift. 

    $$dX_t = \sigma(t_end, x_end) dW_t$$
    """
    sname = 'MvNDBr'    
    @property
    def LinearSDECls(self):
        return MvIndepBrownianMotion if self._diag_cov else MvBrownianMotion

    def build_linear_sde(self):
        N, dimX = self.N, self.dimX
        m = np.zeros(self.x_end.shape); s = self.sigma_x_end
        self.LinearSDE = self.LinearSDECls(N=N, dimX=dimX, m=m, s=s)
        
class MvDriftBrownianAuxBridge(MvVanDerMeulenSchauerAuxBridge, NoDriftBrownianAuxBridge):
    """
    Auxiliary bridge proposal that takes the Brownian motion as the linear SDE
    for evaulation of the proxy, without the drift. 

    $$dX_t = b(t_end, x_end)dt + \sigma(t_end, x_end) dW_t$$
    """
    sname = 'MvDBr'
    @property
    def LinearSDECls(self):
        return MvIndepBrownianMotion if self._diag_cov else MvBrownianMotion

    def build_linear_sde(self):
        N, dimX = self.N, self.dimX; b = self._sde_tol_dec(self.SDE.b)
        m = b(self.t_end, self.x_end); s = self.sigma_x_end
        self.LinearSDE = self.LinearSDECls(N=N, dimX=dimX, m=m, s=s)


# ---------------Multivariate OU Auxiliary Bridge Proposals: 2 Classes ----------------

class MvLLFullOUAuxBridge(MvVanDerMeulenSchauerAuxBridge, OUAuxBridge):
    
    LinearSDECls = MvFullOrnsteinUhlenbeck

    """
    Auxiliary bridge that takes the OU process as the linear SDE for evaluation of the proxy.
    Drift coefficient is obtained through local linearisation of the drift of the signal 
    about the end points of the previous particles. Diffusion coefficient is given by 
    the diffusion of the signal evaluated at the end points of the previous particles.

    Note: if the signal process is an OU-process, this proposal recovers the same OU-process,
    thus we obtain the optimal proposal.

    Requires that first derivative of the drift is defined in the underlying SDE.

    Constructed Linear SDE requires evaluation of the Matrix exponential to solve.
    
    $$dX_t = [A + BX_t]dt + CdW_t$$
    A = b(t_end, x_end) - db(t_end, x_end) * x_end
    B = db(t_end, x_end)
    C = \sigma(t_end, x_end)
    """
        
    sname = 'MvFOU'
    def _gen_full_ou_params(self):
        tol_dec = self._sde_tol_dec
        b = tol_dec(self.SDE.b); db = tol_dec(self.SDE.db); sigma = tol_dec(self.SDE.sigma)
        A = b(self.t_end, self.x_end) - np.einsum('ijk,ik->ij', db(self.t_end, self.x_end), self.x_end) # (N, dimX)
        B = db(self.t_end, self.x_end) # (N, dimX, dimX)
        C = sigma(self.t_end, self.x_end) # (N, dimX, dimX)
        _rho = -1.*B
        _mu = nla.solve(-1.*B, A) # (N, dimX)
        _phi = C
        return _rho, _mu, _phi

    def build_linear_sde(self):
        N, dimX = self.N, self.dimX
        _rho, _mu, _phi = self._gen_full_ou_params() # (N, dimX, dimX) (N, dimX), (N, dimX, dimX)
        self.LinearSDE = self.LinearSDECls(N=N, dimX=dimX, rho=_rho, mu=_mu, phi=_phi)
        
class MvLLOUAuxBridge(MvLLFullOUAuxBridge):
    """
    Auxiliary bridge that takes the OU process as the linear SDE for evaluation of the proxy.
    Drift coefficient is obtained through local linearisation of the drift of the signal 
    about the end points of the previous particles. Diffusion coefficient is given by 
    the diffusion of the signal evaluated at the end points of the previous particles.

    Note: if the signal process is an OU-process, this proposal recovers the same OU-process,
    thus we obtain the optimal proposal.

    Requires that first derivative of the drift is defined in the underlying SDE.

    Constructed Linear SDE requires evaluation of the Matrix exponential to solve.
    
    $$dX_t = [A + BX_t]dt + CdW_t$$
    A = b(t_start, x_end) - db(t_start, x_end) * x_end
    B = db(t_start, x_end)
    C = \sigma(t_start, x_end)
    """
    sname = 'MvOU'
    
    @property
    def LinearSDECls(self):
        return MvIndepOrnsteinUhlenbeck if self._diag_cov else MvOrnsteinUhlenbeck
    
    def build_linear_sde(self):
        N, dimX = self.N, self.dimX
        _rho, _mu, _phi = self._gen_full_ou_params() # (N, dimX, dimX) (N, dimX), (N, dimX, dimX)
        _rho = np.stack([np.diag(_rho[i]) for i in range(N)], axis=0) if N > 1 else np.diag(_rho[0]) # (N, dimX): Only take diagonals
        _phi = self.sigma_x_end
        self.LinearSDE = self.LinearSDECls(N=N, dimX=dimX, rho=_rho, mu=_mu, phi=_phi)


# Add 'FullOUAuxBridge' afer Full OU is implemented to mv_auxiliary_bridges
# Add 'FullOUProposal' after Full OU is implemented to mv_forward_proposals       
univ_forward_proposals = [NoDriftBasicBrownianProp, NoDriftBrownianProp, DriftBrownianProp, LocalLinearOUProp]
univ_auxiliary_bridges = [DelyonHuAuxBridge,  NoDriftBrownianAuxBridge, DriftBrownianAuxBridge, LocalLinearOUAuxBridge]
mv_forward_proposals = [MvNoDriftBasicBrownianProp, MvNoDriftIndepBrownianProp, MvNoDriftBrownianProp, MvDriftIndepBrownianProp, MvDriftBrownianProp, MvOUProposal, MvIndepOUProposal]
mv_auxiliary_bridges = [MvDelyonHuAuxBridge, MvNoDriftBrownianAuxBridge, MvDriftBrownianAuxBridge, MvLLOUAuxBridge] 
