# -*- coding: utf-8 -*-
"""
SDEs module
"""
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
from particles.distributions import Normal, VaryingCovNormal, MvNormal
from particles.kalman import MeanAndCov, filter_step
from sdes.numerical_schemes import EulerMaruyama, LinearExact, MvEulerMaruyama, MvLinearExact
from sdes.tools import vectorise_param, grad_log_linear_gaussian, grad_grad_log_linear_gaussian, mv_grad_log_linear_gaussian, mv_grad_grad_log_linear_gaussian 
from sdes.tools import get_methods, get_attrs, get_properties, vec_grad_log_linear_gaussian


class SDE(object):
    """
    Base class for 1D SDEs. To create an SDE, the following methods need to be defined:

    b: The drift coefficient
    sigma: The diffusion coefficient

    Optionally, one can also define: 

    db: First derivative of the drift coefficient
    dsigma: First derivative of the diffusion coefficient

    Given an SDE, there two primary functionalities: simulation, and transformation.

    To simulate a sample path from an SDE, one can use the method `simulate', to 
    generate samples from a start time to an end time, for given starting value.

    A 1D SDE implies existence of a unique function from the driving noise to the 
    SDE solution, a discretisation of this map, which is derived from the natural 
    approach to discretising integrals, is implemented as the method 
    `transform_W_to_X'. 

    A (non-degenerate) 1D SDE is always elliptic, so there also exists a function 
    from the SDE solution to the driving noise. This function is implemented,
    again through a natural discretisation, using the method `transform_X_to_W'.

    All 3 of these methods are implemented in a vectorised form, (see the functions
    `X_to_W_step', `W_to_X_step' and 'univ_simulation_step'). In practice, this 
    means that one can say generate N simulations from N different starting points
    from a start time t_start to an end time t_end, using a single call to the method.

    Example:

    class OrnsteinUhlenbeck(SDE):

        default_params = {'theta': 1.,
                        'mu': 0.,
                        'sigma': 1.
                        }
        
        def b(self, x):
            return self.theta * (self.mu - x)
        
        def sigma(self, x):
            return self.sigma
    """
    dimX = 1
    dimW = 1
    numerical_scheme_cls = EulerMaruyama

    def __init__(self, **kwargs):
        if hasattr(self, 'default_params'):
            self.__dict__.update(self.default_params)
        self.__dict__.update(kwargs)

    @property
    def params(self):
        return {k: self.__dict__[k] for k in self.default_params.keys()}

    def _error_msg(self, method):
        return ('method ' + method + ' not implemented in SDE class %s' %
                self.__class__.__name__)

    def b(self, t: float, x):
        """
        Placeholder for drift coeffciient of an SDE
        Input a scalar/1D ndarray for x
        Output a scalar/1D ndarray
        respectively
        """
        raise NotImplementedError(self._error_msg('b'))

    def sigma(self, t: float, x):
        """
        Placeholder for the diffusion coefficient of the SDE
        Input a scalar/1D ndarray for x
        Output a scalar/1D ndarray
        respectively
        """
        raise NotImplementedError(self._error_msg('sigma'))

    def db(self, t: float, x):
        """
        Placeholder for first derivative w.r.t x of the diffusion coefficient of the SDE
        Input a scalar/1D ndarray for x
        Output a scalar/1D ndarray
        respectively
        """
        return NotImplementedError(self._error_msg('db'))

    def dsigma(self, t: float, x):
        """
        Placeholder for first derivative w.r.t x of the diffusion coefficient of the SDE
        Input a scalar/1D ndarray for x
        Output a scalar/1D ndarray
        respectively
        """
        raise NotImplementedError(self._error_msg('dsigma'))

    def Cov(self, t: float, x: float):
        return self.sigma(t, x) ** 2

    def dCov(self, t: float, x: float):
        return 2 * self.sigma(t, x) * self.dsigma(t, x)

    def simulate(self, size: int, x_start=None, t_start: float =0., t_end: float =1., num=5) -> np.ndarray:
        """
        Method to generate sample paths from a 1D SDE: implementation is vectorised, so simulation 
        using multiple start points is possible.

        Inputs
        ------------
        size (int):           The number of simulations to generate. If x_start is a vector, must match dimension.
        x_start (array-like): The starting point(s) for the simulation. For a single start point, can be 
                              set to a float, otherwise set to a vector. If set to a vector, size must 
                              match the vector dimension.
        t_start (float):      The starting time of the simulation. 
        t_end (float):        The ending time of the simulation.
        num (int):            The number of imputed points used in the simulation. Low values will have 
                              greater bias but reduce time cost.

        Returns
        ------------
        simulations (np.ndarray): A structured array (structured by timestamps) containing simulation outputs. 
        """
        self.numerical_scheme = self.numerical_scheme_cls(self)
        simulations = self.numerical_scheme.simulate(size=size, t_start=t_start, t_end=t_end, x_start=x_start, num=num)
        return simulations
    
    def transform_X_to_W(self, X: np.ndarray, t_start: float, x_start) -> np.ndarray:
        """
        Method to apply the map from the solution of the SDE to the driving noise, using discretisation.
        Transform is uniquely determined by a drift and diffusion coefficient, a start time, end time and
         a start point. End time is implied by the input X, so is not an input to the function.
        Implementation is vectorised in the start point, so a different transform can be applied to a 
        each of the simulations in the input X.

        Inputs
        ------------
        X (np.ndarray):         Structured array of simulations to which to apply the transform
        t_start (float):        Start time of the SDE
        x_start (array-like)    Start value for the transform. Can either be a float, in which case the 
                                same transform is applied to every simulation

        Returns
        ------------
        W (np.ndarray):         Structured array of the same shape and dtype of 'X', that contains the 
                                transform outputs.
        """
        W =self.numerical_scheme_cls.transform_X_to_W(X, t_start=t_start, x_start=x_start, transform_end_point=True)
        return W
    
    def transform_W_to_X(self, W: np.ndarray, t_start: float, x_start) -> np.ndarray:
        """
        Method to apply the map from the driving noise to the solution of the SDE, using discretisation.
        Transform is uniquely determined by a drift and diffusion coefficient, a start time, end time and
         a start point. End time is implied by the input W, so is not an input to the function.
        Implementation is vectorised in the start point, so a different transform can be applied to each
        of the simulations in the input W.

        Inputs
        ------------
        W (np.ndarray):         Structured array of simulations to which to apply the transform
        t_start (float):        Start time of the SDE
        x_start (array-like):   Start value for the transform. Can either be a float, in which case the 
                                same transform is applied to every simulation

        Returns
        ------------
        X (np.ndarray):         Structured array of the same shape and dtype of 'W', that contains the 
                                transform outputs.
        """
        X = self.numerical_scheme.transform_W_to_X(W, t_start=t_start, x_start=x_start, transform_end_point=True)
        return X

# Some example univariate SDEs:

class LinearSDE(SDE):
    """
    Linear SDE, of the form:

    $$dX_t = [A(t) + B(t)X_t] dt + C(t)dW_t$$

    Includes the methods
    'log_px'
    'grad_log_px'
    'grad_log_py'
    'exact_simulate'
    'transition_dist'
    'optimal_proposal_dist'

    Linear SDEs have a tractable Gaussian transition density, and so can be simulated from exactly:

    p_{s,t}(x_t| x_s) = \phi(x_t, a x_s + b, v)

    a, b and v correspond to _a, _b and _v methods.

    The _a, _b and _v methods need to be implemented for a given subclass of LinearSDE. 
    Doing so defines the transition density of the Linear SDE, thus enabling exact sampling,
    and evaluation of the transition density.

    For the general linear SDE the quantities involve evaluating various integrals.
    We may come back to implement this in the future.
    """
    exact_scheme = LinearExact

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vec_params = False

    def A(self, t):
        raise NotImplementedError(self._error_msg('A'))
    
    def B(self, t):
        raise NotImplementedError(self._error_msg('B'))

    def C(self, t):
        raise NotImplementedError(self._error_msg('C'))

    def _a(self, s, t):
        "Needs to be implemented to use LinearSDEs for subclassing"
        return NotImplementedError(self._error_msg('_a'))

    def _b(self, s, t):
        "Needs to be implemented to use LinearSDEs for subclassing"
        return NotImplementedError(self._error_msg('_b')) 
    
    def _v(self, s, t):
        "Needs to be implemented to use LinearSDEs for subclassing"
        return NotImplementedError(self._error_msg('_v'))

    def _a_vec(self, s, t):
        "Needs to be implemented to use LinearSDEs for subclassing"
        return NotImplementedError(self._error_msg('_a_vec'))

    def _b_vec(self, s, t):
        "Needs to be implemented to use LinearSDEs for subclassing"
        return NotImplementedError(self._error_msg('_b_vec')) 
    
    def _v_vec(self, s, t):
        "Needs to be implemented to use LinearSDEs for subclassing"
        return NotImplementedError(self._error_msg('_v_vec'))
    
    def b(self, t, x):
        return self.A(t) + self.B(t) * x
    
    def sigma(self, t, x):
        return self.C(t)

    def b_vec(self, t, x):
        return self.A_vec(t) + self.B_vec(t) * x
    
    def sigma_vec(self, t, x):
        return self.C_vec(t)

    def Cov_vec(self, t, x):
        return self.sigma_vec(t, x) ** 2

    def db(self, t, x):
        return self.B(t)
    
    def dsigma(self, t, x):
        return 0.
    
    def grad_log_px(self, s: float, t: float, x_s: np.ndarray, x_t: np.ndarray):
        """
        Gradient of the log transition density
        """

        # When simulating:

        # s: float
        # t: float
        # x_s: (N, )
        # x_t: (N, )

        return grad_log_linear_gaussian(x_s, x_t, self._a(s, t), self._b(s, t), self._v(s, t))

    def _vec_grad_log_px(self, s: float, t: float, x_s: np.ndarray, x_t: np.ndarray):
        """
        Vectorised implementation of the gradient of the log transition density, for use in 
        the evaluation of path integrals.
        s: (num+1, )
        t: float    
        x_s (N, num+1)
        x_t (N, )
        """
        return vec_grad_log_linear_gaussian(x_s, x_t, self._a_vec(s, t), self._b_vec(s, t), self._v_vec(s, t))
    
    def _vec_grad_grad_log_px(self, s: float, t: float):
        """
        s: (num+1, )
        t: float
        """
        return grad_grad_log_linear_gaussian(self._a_vec(s, t), self._b_vec(s, t), self._v_vec(s, t))
    
    def grad_log_py(self, s: float, t: float, x_s: np.ndarray, y_t: np.ndarray, LY: float, sigmaY: float):
        """
        Gradient of the log transition density of Y_t | X(s), where Y_t | E_t \sim N(Le_t, \sigma_Y^2)

        # When simulating:

        # s: float
        # t: float
        # x_s: (N, ) 
        # y_t: (1, ) / (1, dimY)
        # eta_sq: float / (dimY, dimY)
        """
        dimY = y_t.shape[1] if type(y_t) is np.ndarray and y_t.ndim > 1 else 1
        if dimY == 1:
            v = (LY**2)*self._v(s, t) + (sigmaY ** 2)
            return grad_log_linear_gaussian(x_s, y_t, LY*self._a(s, t), LY*self._b(s, t), v)
        else:
            """
            Gradient of the log transition density of Y_t | X(s), where Y_t | E_t=e_t \sim N(Le_t, \sigma_Y \sigma_Y^T)
            
            To do: Think about feeding in the Cholesky decomposition instead of the covariance to save compute time.
            # s: float
            # t: float
            # x_s: float/ (N, )
            # y_t: (1, dimY)
            # LY: (dimY, 1)
            # sigmaY: (dimY, dimY)
            
            _a(s, t) -> (N, 1 , 1)
            _b(s, t) -> (N, 1)
            _v(s, t) -> (N, 1, 1)
            """
            x_s = np.ndarray([x_s]).reshape((-1, 1)) if type(x_s) is float else x_s.reshape((-1, 1))
            N = x_s.shape[0]
            _a = np.array(self._a(s, t)).reshape((N, 1, 1))
            _b = np.array(self._b(s, t)).reshape((N, 1))
            _v = np.array(self._v(s, t)).reshape((N, 1, 1))
            _a = np.einsum('ij,hjk->hik', LY, _a) #(N, dimY, dimX)
            _b = (LY @ _b.T).T #(N, dimY)
            V_L_T = np.einsum('ijk,kl->ijl', _v, LY.T) #(N, dimX, dimY)
            L_V_L_T = np.einsum('ij,hjk->hik', LY, V_L_T) #(N, dimY, dimY)
            V = L_V_L_T + sigmaY @ sigmaY.T #(N, dimY, dimY)
            return mv_grad_log_linear_gaussian(x_s, y_t, _a, _b, V)

    def grad_grad_log_py(self, s: float, t: float, y_t: np.ndarray, LY: float, sigmaY: float):
        """
        Second derivative of the gradient of the log transition density of Y_t | X(s), where Y_t | E_t \sim N(Le_t, \sigma_Y^2)
        s: float
        t: float
        y_t: (1, ) / (1, dimY)
        LY: float / (dimY, 1)
        sigmaY: float / (dimY, dimY)
        """
        dimY = y_t.shape[1] if type(y_t) is np.ndarray and y_t.ndim > 1 else 1
        if dimY == 1:
            v = (LY**2)*self._v(s, t) + sigmaY ** 2
            return grad_grad_log_linear_gaussian(self._a(s, t), self._b(s, t), v)
        else:
            """
            The hessian of the log of a linear Gaussian transition density w.r.t x_s 
            X_t | X_s = x_s \sim \mathcal{N}(A x_s + b, S)

            $$ \nabla_{x_s} \nabla_{x_s}^T \log(p_{s, t}(x_t|x_s)) = -A^T S^{-1} A $$
            
            Standard dimensions of the inputs:

            x_s (N, dimX)
            x_t (N, dimY)
            A (N, dimY, dimX)
            b (N, dimY)
            S (N, dimY, dimY)
            
            Dimension of output: 
            (N, dimX)
            """
            _a = np.array(self._a(s, t)).reshape((1, 1, 1))
            _b = np.array(self._b(s, t)).reshape((1, 1))
            _v = np.array(self._v(s, t)).reshape((1, 1, 1))
            return mv_grad_grad_log_linear_gaussian(_a, _b, _v)
    
    def _vec_grad_log_py(self, s: float, t: float, x_s: np.ndarray, y_t: np.ndarray, LY: float, sigmaY: float):
        """
        When implementing a Girsanov integral:

        s (num+1, )
        t: float ()
        x_s: (N, num+1)
        y_1: float 
        eta_eq: float        
        """
        if not self.vec_params:
            self.generate_vec_params(len(s))
        v = (LY**2)*self._v_vec(s, t) + sigmaY ** 2
        return grad_log_linear_gaussian(x_s, y_t, LY*self._a_vec(s, t), LY*self._b_vec(s, t), v)
    
    def exact_simulate(self, size: int, x_start, t_start: float = 0., t_end: float = 1., num=5) -> np.ndarray:
        """
        Exact simulation of the linear SDE
        """
        return self.exact_scheme.simulate(size=size, t_start=t_start, t_end=t_end, x_start=x_start, num=num)

    def transition_dist(self, s: float, t: float, x_s: np.ndarray):
        """
        Transition density of the linear SDE, output as a distribution object:
        """
        a = self._a(s, t); b = self._b(s, t); v = self._v(s,t)
        return Normal(loc=a*x_s + b, scale=np.sqrt(v))

    def optimal_proposal_dist(self, s: float, t: float, x_s: np.ndarray, y_t: np.ndarray, LY: float, sigmaY: float):
        """
        Proposal for the end point using the exact distribution $E_t | E_{t-1}=e_{t-1}, Y_t = y_t$ from a Linear SDE.
        x_s: float / (N, )
        y_t: (1, )/ (1, dimY)
        LY: float / (dimY, 1)
        sigmaY: float / (dimY, dimY)
        """    
        a = self._a(s, t); b = self._b(s, t); v = self._v(s,t); sigmaY_sq = sigmaY ** 2
        opt_prop_mean = (a*x_s + b) + (LY*v)/((LY*v) + sigmaY_sq) * (y_t - LY*(a*x_s+b))
        opt_prop_var = v * (1 - (LY*v)/((LY*v) + sigmaY_sq))
        return Normal(loc=opt_prop_mean, scale=np.sqrt(opt_prop_var))
    
    def generate_vec_params(self, num_plus_1):
        vec_params = {}
        for k in self.default_params.keys():
            param = self.__dict__[k]
            vec_params['vec_' + k] = vectorise_param(param, num_plus_1)
        self.__dict__.update(vec_params)
        self.vec_params = True
    
    def grad_param_log_px(self, s, t, x, xf, param_name):
        v=self._v;  k = self._k
        dv = self.df_dp('v', param_name); dk = self.df_dp('k', param_name)
        grad_param_log_px = -0.5*dv(s, t)
        grad_param_log_px +=-0.5*(v(s, t)*dk(s, t, x, param_name) - k(s, t, x, xf)*dv(s, t))/(v(s, t) ** 2)
        return grad_param_log_px

    def _k(self, s, t, x, xf):
        return (xf - self._a(s, t) * x - self._b(s, t)) ** 2
    
    def df_dp(self, f: str, param_name: str):
        if f == 'k':
            da_dp = self.df_dp('a', param_name=param_name); db_dp = self.df_dp('b', param_name=param_name)
            def dk_dp(s, t, x):
                return -2. * (da_dp(s, t) * x + db_dp(s, t))
            return dk_dp
        elif f in ['a', 'b', 'v']:
            return getattr(self, f'_d{f}_d{param_name}')
        else: 
            raise ValueError("Input 'f' must be in [a, b, v, k]")

class TimeLinearSDE(LinearSDE):
    """
    Linear SDE that is not state dependent.
    $$dX_t = A(t)dt + C(t)dW_t$$.

    To use this class, one needs subclass, and to define methods: 
    'A', 'int_A', 'C', 'int_C_sq'.

    In this class of Linear SDEs, the transition density does not depend on start point 'x_0'
    """
    def B(self, t):
        return 0.

    def B_vec(self, t):
        return 0.
       
    def int_A(self, t):
        raise NotImplementedError(self._error_msg('int_A'))
    
    def int_A_vec(self, t):
        raise NotImplementedError(self._error_msg('int_A_vec'))

    def int_C_sq(self, t):
        raise NotImplementedError(self._error_msg('int_C_sq'))

    def int_C_sq_vec(self, t):
        raise NotImplementedError(self._error_msg('int_C_sq_vec'))
    
    def _a(self, s, t):
        return 1.

    def _a_vec(self, s, t):
        return 1.
    
    def _b(self, s, t):
        return self.int_A(t) - self.int_A(s)

    def _b_vec(self, s, t):
        return self.int_A_vec(t) - self.int_A_vec(s)
        
    def _v(self, s, t):
        return self.int_C_sq(t) - self.int_C_sq(s)
    
    def _v_vec(self, s, t):
        return self.int_C_sq_vec(t) - self.int_C_sq_vec(s)

class BrownianMotion(TimeLinearSDE):
    """
    SDE of the Brownian Motion:
    $$dX_t = m dt + s dW_t$$
    """
    default_params = {'m': 0.,
                      's': 1.
                      }

    def A(self, t):
        return self.m

    def A_vec(self, t):
        if not self.vec_params:
            self.generate_vec_params(len(t))
        return self.vec_m
    
    def int_A(self, t):
        return self.m * t

    def int_A_vec(self, t):
        return self.vec_m * t
    
    def C(self, t):
        return self.s

    def C_vec(self, t):
        return self.vec_s
    
    def int_C_sq(self, t):
        return (self.s ** 2) * t
    
    def int_C_sq_vec(self, t):
        return (self.vec_s ** 2) * t

class OrnsteinUhlenbeck(LinearSDE):
    """
    The Ornstein-Uhlenbeck SDE, given by:
    $dX_t = \rho(\mu - X_t)dt + \phi dW_t$
    """
    default_params = {'rho': 0.2,
                      'mu': 0.,
                      'phi': 1.
                      }
    
    def A(self, t):
        return self.rho * self.mu
    
    def B(self, t):
        return -1. * self.rho

    def C(self, t):
        return self.phi

    def A_vec(self, t):
        if not self.vec_params:
            self.generate_vec_params(len(t))
        return self.vec_rho * self.vec_mu

    def B_vec(self, t):
        return -1 * self.vec_rho

    def C_vec(self, t):
        return self.vec_phi

    def _a(self, s, t):
        return np.exp(-self.rho * (t-s))

    def _a_vec(self, s, t):
        return np.exp(-self.vec_rho * (t-s))

    def _da_drho(self, s, t):
        return -(t-s) * self._a(s, t)
    
    def _da_dphi(self, s, t):
        return 0.
    
    def _da_dmu(self, s, t):
        return 0.

    def _b(self, s, t):
        return self.mu * (1. - self._a(s, t))

    def _b_vec(self, s, t):
        return self.vec_mu *(1. - self._a_vec(s, t))

    def _db_drho(self, s, t):
        return -self.mu * (t-s) * self._a(s, t)

    def _db_dphi(self, s, t):
        return 0.
    
    def _db_dmu(self, s, t):
        return (1. - self._a(s, t))
    
    def _v(self, s, t):
        return ((self.phi ** 2)/(2*self.rho)) * (1 - np.exp(-2*self.rho*(t-s)))

    def _v_vec(self, s, t):
        return ((self.vec_phi ** 2)/(2*self.vec_rho)) * (1 - np.exp(-2*self.vec_rho*(t-s)))
    
    def _dv_drho(self, s, t):
        return ((self.phi ** 2)/(2*self.rho)) * ((2.*((t-s) + (1./(2.*self.rho)))*np.exp(-2.*self.rho*(t-s)))- (1./self.rho))

    def _dv_dphi(self, s, t):
        return 2./self.phi * self._v(s, t)

    def _dv_dmu(self, s, t):
        return 0.
    
class TVOrnsteinUhlenbeck(OrnsteinUhlenbeck):
    """
    Time varying Ornstein Uhlenbeck process to use to test the smoothing algorithms:
    
    $dX_t = \rho(\mu - X_t)dt + C(t) dW_t$

    Where C(t) = \phi_1 for t \in [0,1]
               = \phi_2 for t > 1 
    """

    default_params = {'rho': 0.2,
                      'mu': 0.,
                      'phi_1': 0.3,
                      'phi_2': 0.1
                      }
    
    def C(self, t):
        return self.phi_1 * (t < 1) + self.phi_2 *(t >= 1) 
        
    def C_vec(self, t):
        return self.vec_phi_1 * (t < 1) + self.vec_phi_2 * (t >= 1)

    def _v(self, s, t):
        return (s>=1)*self._v_s_geq1(s, t) + (t <= 1)*self._v_t_leq1(s, t) + (s<1) * (t>1) * self._v_t_else(s, t)
    
    def _v_s_geq1(self, s, t):
        return ((self.phi_2 ** 2)/(2*self.rho)) * (1 - np.exp(-2.*self.rho*(t-s)))
    
    def _v_t_leq1(self, s, t):
        return ((self.phi_1 ** 2)/(2*self.rho)) * (1 - np.exp(-2.*self.rho*(t-s)))
    
    def _v_t_else(self, s, t):
            v_t  = (self.phi_2 ** 2)* (1 - np.exp(-2.*self.rho*(t-1.)))
            v_t += (self.phi_1 ** 2)* (np.exp(-2.*self.rho*(t-1.)) - np.exp(-2.*self.rho*(t-s)))
            v_t = v_t/(2.*self.rho)
            return v_t
    
    def _v_vec(self, s, t):
        return (s>=1)*self._v_s_geq1_vec(s, t) + (t <= 1)*self._v_t_leq1_vec(s, t) + (s<1) * (t>1) * self._v_t_else_vec(s, t)
    
    def _v_s_geq1_vec(self, s, t):
        return ((self.vec_phi_2 ** 2)/(2*self.vec_rho)) * (1 - np.exp(-2.*self.vec_rho*(t-s)))
    
    def _v_t_leq1_vec(self, s, t):
        return ((self.vec_phi_1 ** 2)/(2*self.vec_rho)) * (1 - np.exp(-2.*self.vec_rho*(t-s)))
    
    def _v_t_else_vec(self, s, t):
            v_t  = (self.vec_phi_2 ** 2)* (1 - np.exp(-2.*self.vec_rho*(t-1.)))
            v_t += (self.vec_phi_1 ** 2)* (np.exp(-2.*self.vec_rho*(t-1.)) - np.exp(-2.*self.vec_rho*(t-s)))
            v_t = v_t/(2.*self.vec_rho)
            return v_t

class BrownianBridge(LinearSDE):
    """
    dX_t = \frac{x^* - X_t}{T-t}dt + dW_t

    The Brownian Bridge SDE.

    Parameters:
    End point: x^*
    End time: T (This bit could be reformulated so that this is a Bridge construction)

    This definition could be extended to add a general constant in the diffusion coefficient:
    """
    default_params = {'x_end': 0.,
                      'T': 1.}
        
    def A(self, t):
        self.x_end/(self.T - t)

    def B(self, t):
        return -1./(self.T - t)
    
    def C(self, t):
        return 1.

    def _a(self, s, t):
        return (self.T - t)/(self.T-s)

    def _b(self, s, t):
        return (t-s)/(self.T - s) * self.x_end
        
    def _v(self, s, t):
        return (t-s)*(self.T-t)/(self.T - s)

class SinDiffusion(SDE):
    """
    The sin diffusion SDE, given by:
    $dX_t = sin(x -\theta_1) dt + \theta_2 dW_t$
    """
    default_params = {'theta_1': 0., 
                      'theta_2': 1.,
                      'sigma_0': 1.}
    
    def b(self, t, x):
        return np.sin(x - self.theta_1)

    def sigma(self, t, x):
        return self.theta_2

    def db(self, t, x):
        return np.cos(x - self.theta_1)
    
    def dsigma(self, t, x):
        return 0.

class ArctanDiffusion(SDE):
    """
    The arctan diffusion, as used in VanDerMeulen & Schauer (2017), Example 4.3:

    $dX_t = \alpha \arctan(X_t) + \beta dt + \phi dW_t$

    Under certain conditions on the parameters, the process mean reverts to -tan(\beta/\alpha)
    """
    default_params = {'alpha': -2.,
                      'beta': 0.,
                      'phi': 0.75}
    
    def b(self, t, x):
        return self.alpha * np.arctan(x) + self.beta
    
    def sigma(self, t, x):
        return self.phi
    
    def db(self, t, x):
        return self.alpha/(1 + (x ** 2))
    
    def dsigma(self, t, x):
        return 0.

class ManualOptOU_Linear(OrnsteinUhlenbeck):

    """
    To do: define the methods _a ,_b and _v.

    These require us to evaluate the integral y(t) = \int_0^t B(s)ds 
    """
    default_params = {'rho': 0.2,
                    'mu': 0.,
                    'phi': 1.,
                    'y': 1.,
                    'LY': 1.,
                    'sigmaY': 0.01,
                    'T': 1.
                    }

    def A(self, t):
        A = (self.phi ** 2)/(self.LY ** 2 * super()._v(t, self.T) + (self.sigmaY ** 2))
        A = A * (self.y - super()._b(t, self.T)) * super()._a(t, self.T)
        return A
    
    def B(self, t):
        B = (super()._a(t, self.T) ** 2) * (self.phi ** 2)
        B = B/(self.LY ** 2 * super()._v(t, self.T) + (self.sigmaY ** 2))
        B = -self.rho - B
        return B

    def integ_factor(self, t):
        K = (self.LY ** 2) * (self.phi ** 2) / (2 * self.rho)
        sigma_Y_sq = self.sigmaY ** 2
        num = K + sigma_Y_sq - K*np.exp(-2*self.rho*self.T)
        denom = K + sigma_Y_sq - K*np.exp(-2*self.rho*(self.T-t))
        exponent = (self.phi ** 2) / (2 * self.rho)
        return np.exp(self.rho * t) * ((num/denom) ** exponent)

                                                                        
class ManualOptOU(SDE):
    """
    Manual implementation of the optimal proposal for the 1D OU SDE.
    """
    default_params = {'rho': 0.2,
                    'mu': 0.,
                    'phi': 1.,
                    'y': 1.,
                    'LY': 1.,
                    'sigmaY': 0.01,
                    'T': 1.
                    }
    
    def b(self, t, x):
        return -self.rho * x + (self.phi ** 2) * self.grad_log_py(t, x)

    def sigma(self, t, x):
        return self.phi

    def db(self, t, x):
        return -self.rho - (self.phi ** 2) * self.grad_grad_log_py(t, x)

    def dsigma(self, t, x):
        return 0.

    def _a(self, s, t):
        return np.exp(-self.rho*(t-s))

    def _b(self, s, t):
        return 0.
    
    def _v(self, s, t):
        return (self.phi ** 2)/(2*self.rho) * (1 - np.exp(-2*self.rho*(t-s)))

    def var_py(self, t):
        return ((self.LY ** 2) * self._v(t, self.T) + self.sigmaY ** 2)
    
    def loc_py(self, t, x):
        return self._a(t, self.T)*x + self._b(t, self.T)
    
    def grad_log_py(self, t, x):
        return self._a(t, self.T) * (self.y - self.loc_py(t, x))/self.var_py(t)
    
    def grad_grad_log_py(self, t, x):
        return (self._a(t, self.T) ** 2)/self.var_py(t)

# Multivariate SDEs:

class MvSDE(SDE):
    """
    Subclass this class to create multivariate SDEs.
    You will need to define:

    dimX: the dimension of the state X
    dimW: The dimension of the driving Brownian noise

    b: The Drift function: Should map from (1, dimX) to (1, dimX).
    sigma: The Covariance function: Should map from (1, dimX) to (dimX, dimW)
    
    Note that dimX <= dimW is a necessary condition to have an invertible covariance matrix.
    """

    def b(self, t: float, x: np.ndarray):
        """  
            Input  an array-like of shape (N, dimX)
            Output an ndarray of shape (N, dimX)
            Should broadcast an (N, dimX) input to an (N, dimX) output.
        """
        raise NotImplementedError(self._error_msg('b'))
    
    def sigma(self, t: float, x: np.ndarray):
        """  
            Input an array-like of shape (N, dimX)
            Output an ndarray of shape (N, dimX, dimW)
        """
        raise NotImplementedError(self._error_msg('sigma'))

    def db(self, t: float, x: np.ndarray):
        """  
            Input  an array-like of shape (N, dimX)
            Output an ndarray of shape (N, dimX, dimX)
        """
        raise NotImplementedError(self._error_msg('b'))

    def dsigma(self, t: float, x: np.ndarray):
        """  
            Input  an array-like of shape (N, dimX)
            Output an ndarray of shape (N, dimX, dimX, dimX)
            Should broadcast an (N, dimX) input to an (N, dimX) output.
        """
        raise NotImplementedError(self._error_msg('b'))
    
    def Cov(self, t: float, x: np.ndarray):
        """  
            Input an array-like of shape (N, dimX)
            Output an ndarray of shape (N, dimX, dimX)
        """
        rt_cov = self.sigma(t, x)
        rt_cov_T = np.einsum('ijk->ikj', rt_cov)
        return np.einsum('ijk,ikl->ijl', rt_cov, rt_cov_T)

    # def dCov(self, t, x):
    #     # self.sigma(t, x) (N, dimX, dimX)
    #     return np.einsum('ijkl,',self.sigma(t, x), self.dsigma(t, x)) + np.einsum(''self.dsigma(t, x), self.sigma(t, x))
        
    @property
    def _diag_cov(self):
        if hasattr(self.__class__, 'diag_cov'):
            return self.__class__.diag_cov
        else:
            return False

class MvEllipticSDE(MvSDE):

    numerical_scheme_cls = MvEulerMaruyama

    @property
    def dimW(self):
        return self.dimX
    
class MvLinearSDE(MvSDE, LinearSDE):
    """
    Multivariate linear SDEs. These objects are used in forward and backward proposals within DA particle filters.
    
    When initialised, the __init__method should do the following things:
    - Check that the parameters of the SDE, given the above are well-defined.
    
    Also, make a note of how the input parameters are used build the linear SDE.
    
    Implements the following methods:
        
    'grad_log_px' - Used in simulation/transformations of diffusion bridges in backward proposals.
    'grad_log_py' - Used in simulation/transformations of forward proposals.
    'exact_simulate' - Not used currently in SMC algorithms.
    'optimal_proposal_dist' - Used to construct end point proposals in backward proposals.
    'transition_dist' - Used in weights for VanderMeulenSchauer proposals.
    
    When initialised, one needs to provide:
    
    N - The number of particles
    dimX - The dimension of the SDE
    
    One also needs to provide parameters for each of the N proposal SDEs. Each N corresponds to a different 
    proposal bridge. 
    
    To do:
    ------------
    
    - Some of the MvLinear SDEs that you have defined have diagonal drift and covariance matrices for simplicity.
        In these cases, change the form of the output of A, B, C and _a, _b, _v so that they have different dimensions.
        - Then, change the methods b, and sigma and Cov so that they are computed faster for these special cases.
        - Then, write a new version of 'optimal_proposal_dist' that impelments the special case faster.
        - Then, change the code of 'mv_grad_log_linear_gaussian' to take advantage of the special case.
            Give it a kwarg diag=True for this case to speed up computations. 
        - When you do this, you may want to think about a version of the code that only takes the diagonal of the covariance matrix. 
    - Start writing the code for the evaluation of the weights.
    """
    exact_scheme = MvLinearExact
    
    def __init__(self, N=1, dimX=2, **kwargs):
        self.dimX = dimX
        self.N = N
        super().__init__(**kwargs)
        self.check_input_params()

    def A(self, t):
        """
        Inputs:
        -------
        t: float: The time step
        
        Returns
        -------
        A: ndarray: A matrix of dimension (self.N, dimX)
        """
        raise NotImplementedError(self._error_msg('A'))
    
    def B(self, t):
        """
        Inputs:
        -------
        t: float: The time step
        
        Returns
        -------
        B: ndarray: A tensor of dimension (self.N, dimX, dimX)
        """
        NotImplementedError(self._error_msg('B'))
    
    def C(self, t):
        """
        Inputs:
        -------
        t: float: The time step
        
        Returns
        -------
        C: ndarray: A matrix of dimension (self.N, dimX, dimX)
        """
        raise NotImplementedError(self._error_msg('C'))
            
    def b(self, t, x):
        """
        Input (N, dimX)
        Output (N, dimX)
        """
        self._check_input_x(x)
        M = x.shape[0]; N = self.N
        A = np.concatenate([self.A(t)]*M, axis=0) if N == 1 else self.A(t)
        B = np.concatenate([self.B(t)]*M, axis=0) if N == 1 else self.B(t)
        return A + np.einsum('ijk,ik->ij', B, x)
    
    def sigma(self, t, x):
        """
        Input (N, dimX)
        Output (N, dimX, dimW)
        """
        self._check_input_x(x)
        M = x.shape[0]; N = self.N
        C = np.concatenate([self.C(t)]*M, axis=0) if N == 1 else self.C(t)
        return C

    # def b_vec(self, t, x):
    #     return self.A_vec(t) + self.B_vec(t) * x
    
    # def sigma_vec(self, t, x):
    #     return self.C_vec(t)

    # def Cov_vec(self, t, x):
    #     return self.sigma_vec(t, x) ** 2

    def db(self, t, x):
        """
        db/dx is a 2D Jacobian, which we evaluate for each of the N samples.
        
        Inputs: t: the time step, x: (N, dimX)
        Returns: (N, dimX, dimX)
        """
        self._check_input_x(x); M = x.shape[0]
        B = np.concatenate([self.B(t)]*M, axis=0) if self.N == 1 and M > 1 else self.B(t)
        return B
            
    def dsigma(self, t, x):
        """
        dsigma/dx is a 3D tensor, which we evaluate for each of the N samples.
        """
        self._check_input_x(x); M = x.shape[0]
        return np.zeros((M, self.dimX, self.dimX, self.dimX))

    def grad_log_px(self, s: float, t: float, x_s: np.ndarray, x_t: np.ndarray):
        """
        Gradient of the log transition density
        """
        # When simulating:
        
        # s: float
        # t: float
        # x_s: (N, dimX)
        # x_t: (N, dimX)

        return mv_grad_log_linear_gaussian(x_s, x_t, self._a(s, t), self._b(s, t), self._v(s, t))

    def grad_grad_log_px(self, s: float, t: float):
        return mv_grad_grad_log_linear_gaussian(self._a(s, t), self._b(s, t), self._v(s, t))
                    
    def grad_log_py(self, s: float, t: float, x_s: np.ndarray, y_t: np.ndarray, LY: np.ndarray, sigmaY: np.ndarray):
        """
        Gradient of the log transition density of Y_t | X(s), where Y_t | E_t=e_t \sim N(Le_t, \sigma_Y \sigma_Y^T)
        
        To do: Think about feeding in the Cholesky decomposition instead of the covariance to save compute time.
        # s: float
        # t: float
        # x_s: (N, dimX)
        # y_t: (N, dimY)
        # LY: (dimY, dimX)
        # sigmaY: (dimY, dimY)
        """
        _a = np.einsum('ij,hjk->hik', LY, self._a(s, t)) #(N, dimY, dimX)
        _b = (LY @ self._b(s, t).T).T #(N, dimY)
        V_L_T = np.einsum('ijk,kl->ijl', self._v(s, t), LY.T) #(N, dimX, dimY)
        L_V_L_T = np.einsum('ij,hjk->hik', LY, V_L_T) #(N, dimY, dimY)
        V = L_V_L_T + sigmaY @ sigmaY.T #(N, dimY, dimY)
        return mv_grad_log_linear_gaussian(x_s, y_t, _a, _b, V)

    # def grad_grad_log_py(self, s: float, t: float, x_s: np.ndarray, y_t: np.ndarray, LY: np.ndarray, sigmaY: np.ndarray):
    #     """
    #     Gradient of the log transition density of Y_t | X(s), where Y_t | E_t=e_t \sim N(Le_t, \sigma_Y \sigma_Y^T)
        
    #     To do: Think about feeding in the Cholesky decomposition instead of the covariance to save compute time.
    #     # s: float
    #     # t: float
    #     # y_t: (1, dimY)
    #     # LY: (dimY, dimX)
    #     # sigmaY: (dimY, dimY)
    #     """
    #     _a = np.einsum('ij,hjk->hik', LY, self._a(s, t)) #(N, dimY, dimX)
    #     _b = (LY @ self._b(s, t).T).T #(N, dimY)
    #     V_L_T = np.einsum('ijk,kl->ijl', self._v(s, t), LY.T) #(N, dimX, dimY)
    #     L_V_L_T = np.einsum('ij,hjk->hik', LY, V_L_T) #(N, dimY, dimY)
    #     V = L_V_L_T + sigmaY @ sigmaY.T #(N, dimY, dimY)
    #     return mv_grad_log_linear_gaussian(x_s, y_t, _a, _b, V)

    def _a(self, s, t):
        """
        Should be a (N, dimX, dimX) matrix
        """
        return NotImplementedError(self._error_msg('_a'))
    
    def _b(self, s, t):
        """
        Should be a (N, dimX) vector.
        """
        return NotImplementedError(self._error_msg('_b'))
    
    def _v(self, s, t):
        """
        Should be a (N, dimX, dimX) matrix
        """
        return NotImplementedError(self._error_msg('_v'))

    def transition_dist(self, s: float, t: float, x_s: np.ndarray):
        """
        Transition density of the linear SDE, output as a distribution object:
        Object is vectorised over N particles.
        
        Inputs
        ------------
        s: float
        t: float
        x_s: (N, dimX) array
        
        Returns
        ------------
        VaryingCovNormal (particles.distributions.dists): a distribution object.
        """
        if self.N > 1:
            a = self._a(s, t); b = self._b(s, t); v = self._v(s,t) # (N, dimX, dimX)
            return VaryingCovNormal(loc=np.einsum('ijk,ik->ij', a, x_s) + b, cov=v)
        else:
            return MvNormal(loc=(x_s @ self._a(s, t)[0].T) + self._b(s, t)[0], cov=self._v(s, t)[0])
    
    def optimal_proposal_dist(self, s: float, t: float, x_s: np.ndarray, y_t: np.ndarray, LY: np.ndarray, sigmaY: np.ndarray):
        """
        Proposal for the end point using the exact distribution $E_t | E_{t-1}=e_{t-1}, Y_t = y_t$ from a Linear SDE.
        
        x_s: (N, dimX)
        y_t: (1, dimY)
        LY: (dimY, dimX)
        sigmaY: (dimY, dimY)
        """        
        N = self.N; dimX = self.dimX; dimY = y_t.shape[1]
        # if N == 1:
        #     trans_dist = self.transition_dist(s, t, x_s)
        #     pred_mean, pred_cov = trans_dist.loc, trans_dist.scale*trans_dist.cov
        #     pred = MeanAndCov(mean=pred_mean, cov=pred_cov)
        #     filter_mean_cov = filter_step(LY, sigmaY @ sigmaY.T, pred, y_t) # This won't work: the filter_step function is not vectorised...!
        #     return MvNormal(loc=filter_mean_cov.mean, cov=filter_mean_cov.cov)
        if x_s.shape != (N, dimX) and N != 1:
            raise ValueError('Input x_s must have shape (N, dimX)')
        if y_t.shape[0] != 1:
            raise ValueError('Input y_t must have first dimension 1')
        A = self._a(s, t); b = self._b(s, t); S = self._v(s,t); CovY = sigmaY @ sigmaY.T
        if N == 1:
            M = x_s.shape[0]
            A = np.concatenate([A]*M); b = np.concatenate([b]*M); S = np.concatenate([S]*M)
            LY = np.stack([LY]*M); CovY = np.stack([CovY]*M)
            y_t = np.concatenate([y_t]*M, axis=0)
        else:
            LY = np.stack([LY]*N); CovY = np.stack([CovY]*N)
            y_t = np.concatenate([y_t]*N, axis=0)
        jt_mu_x = np.einsum('ijk,ik->ij', A, x_s) + b # (N, dimX, dimX), (N, dimX) -> (N, dimX)
        jt_mu_y = np.einsum('ijk,ik->ij', LY, jt_mu_x) # (N, dimY, dimX), (N, dimX) -> (N, dimY)
        jt_cov_xy = np.einsum('ijk,ilk->ijl', S, LY) # (N, dimX, dimX), (N, dimY, dimX) -> (N, dimX, dimY)
        jt_cov_yx = np.einsum('ijk->ikj', jt_cov_xy) # (N, dimY, dimX)
        jt_cov_y =  np.einsum('ijk,ikl->ijl', LY, jt_cov_xy) + CovY # (N, dimY, dimX), (N, dimX, dimY) -> (N, dimY, dimY)
        opt_prop_loc = jt_mu_x + np.einsum('ijk,ik->ij', jt_cov_xy, nla.solve(jt_cov_y, y_t - jt_mu_y)) # (N, dimX)
        opt_prop_cov = nla.solve(jt_cov_y, jt_cov_yx) # (N, dimY, dimX)
        opt_prop_cov = S - np.einsum('ijk,ikl->ijl', jt_cov_xy, opt_prop_cov) # (N, dimX, dimY) (N, dimY, dimX) -> (N, dimX, dimX)
        opt_prop_dist = VaryingCovNormal(loc=opt_prop_loc, cov=opt_prop_cov) if N > 1 else MvNormal(loc=opt_prop_loc, cov=opt_prop_cov[0])
        return opt_prop_dist

    def check_input_params(self):
        """
        Utility method to check that the input parameters are well-defined 
        for the given SDE, assuming a given number of particles and SDE dimension.
        """
        for name, shapes in self.param_shapes.items():
            N = self.N
            param_shape = self.__dict__[name].shape
            if N > 1 and param_shape != shapes[0]:
                raise ValueError(f"Parameter {name} must have shape {shapes[0]} for N>1. Input shape: {param_shape}")
            if N == 1 and len(shapes) == 1 and param_shape != shapes[0]:
                raise ValueError(f"If N=1, then parameter {name} must be of shape {shapes[0]}. Input shape: {param_shape}")
            if N == 1 and len(shapes) > 1 and param_shape != shapes[1]:
                raise ValueError(f"If N=1, then parameter {name} must be of shape {shapes[1]}. Input shape: {param_shape}")
                
    def _check_input_x(self, x):
        if x.shape[0] != self.N and self.N > 1:
            raise ValueError('Input x must match number of particles in the first dimension.')
        if x.shape[1] != self.dimX:
            raise ValueError('Input x must have the same dimension as the SDE in the second dimension.')

class MvIndepBrownianMotion(MvLinearSDE, MvEllipticSDE):
    """
    Multivariate Scaled Brownian Motion, given by:
    dX_t  = m dt + s dW_t
    
    With m = (m_1, ..., m_d)^T, s = diag(s_1, ..., s_d)
    Independence between the components means that we don't need matrix exponentials.
    
    Input parameters:
    'm': (N, dimX) drift vector
    's': (N, dimX) diagonal diffusion matrix vector
    """
    
    @property
    def default_params(self):
        N, dx = self.N, self.dimX
        return {'m': np.zeros((N, dx)),
                's': np.ones((N, dx))
                }

    @property
    def param_shapes(self):
        N, dx = self.N, self.dimX
        return {'m': [(N, dx)], 's': [(N, dx)]}

    def A(self, t):
        return self.m

    def B(self, t):
        return np.zeros((self.N, self.dimX, self.dimX))
    
    def C(self, t):
        return np.stack([np.diag(self.s[i, :]) for i in range(self.N)])
        
    def _a(self, s, t):
        """
        Should be a (N, dimX, dimX) matrix.
        """
        return np.stack([np.eye(self.dimX)]*self.N)

    def _b(self, s, t):
        """
        Should be a (N, dimX) vector.
        """
        return (t-s) * self.m
        
    def _v(self, s, t):
        """
        Should be a (N, dimX, dimX) matrix
        """
        cov = np.stack([np.diag(self.s[i, :]) for i in range(self.N)])
        return (t-s) * cov
    
class MvBrownianMotion(MvIndepBrownianMotion):
    """
    Multivariate Brownian Motion, given by:
    dX_t  = m dt + s dW_t
    
    With m = (m_1, ..., m_d)^T, s is a general dxd matrix.
    
    Input parameters:

    'm': (N, dimX) drift vector
    's': (N, dimX, dimX) if N>1 else (dimX, dimX) diffusion matrix

    """    
    @property
    def default_params(self):
        N, dx = self.N, self.dimX
        default_params = {'m': np.zeros((N, dx))}
        default_params['s'] = np.eye(dx) if N == 1 else np.stack([np.eye(dx)]*N)
        return default_params

    @property
    def param_shapes(self):
        N, dx = self.N, self.dimX
        return {'m': [(N, dx)], 's': [(N, dx, dx), (dx, dx)]}
        
    def C(self, t):
        C = self.s.reshape((self.N, self.dimX, self.dimX)) if self.N == 1 else self.s
        return C
        
    def _v(self, s, t):
        """
        Should be a (N, dimX, dimX) matrix.
        """
        dummy_x = np.zeros((self.N, self.dimX))
        cov = self.Cov(t, dummy_x)        
        return (t-s) * cov

    def check_input_params(self):
        if self.m.shape != (self.N, self.dimX):
            raise ValueError('Drift vector m must be of shape (N, dimX).')            
        if self.s.shape not in [(self.N, self.dimX, self.dimX), (self.dimX, self.dimX)]:
            raise ValueError("Diffusion matrix diagonal s must be of shape (N, dimX, dimX) for N>1 or (dimX, dimX) for N=1")
    
class MvIndepOrnsteinUhlenbeck(MvLinearSDE, MvEllipticSDE):
    """
    Multivariate Ornstein-Uhlenbeck process, given by:
    dX_t  = \rho(\mu - X_t) dt + \phi dW_t
    
    With \rho = diag(\rho_1, ..., \rho_d), C = \diag(\phi_1, \dots, \phi_d)
    and \mu = (\mu_1, \dots, \mu_d)^T.
    Independence between the components means that we don't need matrix exponentials.
    
    Input parameters: 
    'rho': (N, dimX) vector of reversion rates
    'mu': (N, dimX) vector of means
    'phi': (N, dimX) vector of diffusion diagonals
    """

    @property
    def default_params(self):
        N, dx = self.N, self.dimX
        default_params = {'rho': 0.5*np.ones((N, dx)),
                        'mu': np.zeros((N, dx)),
                        'phi': np.ones((N, dx))
                        }
        return default_params

    @property
    def param_shapes(self):
        N, dx = self.N, self.dimX
        return {'rho': [(N, dx)], 'mu': [(N, dx)], 'phi': [(N, dx)]}
    
    def A(self, t):
        return self.mu * self.rho
    
    def B(self, t):
        return np.stack([np.diag(-1.*self.rho[i, :]) for i in range(self.N)])
    
    def C(self, t):
        return np.stack(np.diag(self.phi[i, :]) for i in range(self.N))

    def _a(self, s, t):
        """
        Should be a (N, dimX, dimX) matrix.
        """
        return np.stack([np.diag(np.exp(-self.rho[i, :]*(t-s))) for i in range(self.N)])

    def _b(self, s, t):
        """
        Should be a (N, dimX) vector.
        """
        return (1. - np.exp(-self.rho*(t-s))) * self.mu
        
    def _v(self, s, t):
        """
        Should be an (N, dimX, dimX) matrix.
        """
        def univ_v(s, t, rho, phi):
            return (phi ** 2)/(2*rho) * (1 - np.exp(-2*rho*(t-s)))
        return np.stack([np.diag(univ_v(s, t, self.rho[i, :], self.phi[i, :]) )for i in range(self.N)])

class MvOrnsteinUhlenbeck(MvIndepOrnsteinUhlenbeck, MvEllipticSDE):
    """
    Multivariate Ornstein-Uhlenbeck process, given by:
    dX_t  = \rho(\mu - X_t) dt + \phi dW_t
    
    With \rho a dxd diagonal matrix
    \mu a 1xd vector
    and \phi a dxd matrix
    
    Diagonal \rho means matrix exponential is not required.
    
    Input parameters: 
    'rho': (N, dimX) matrix of reversion rates
    'mu': (N, dimX) vector of means
    'phi': (N, dimX, dimX)/(dimX, dimX) diffusion matrix
    """


    @property
    def default_params(self):
        N, dx = self.N, self.dimX
        default_params = {'rho': 0.5*np.ones((N, dx)),
                        'mu': np.zeros((N, dx)),
                        'phi': np.eye(dx) if N == 1 else np.stack([np.eye(dx)]*N)
                        }
        return default_params

    @property
    def param_shapes(self):
        N, dx = self.N, self.dimX
        return {'rho': [(N, dx)], 'mu': [(N, dx)], 'phi': [(N, dx, dx), (dx, dx)]}

    def C(self, t):
        C = self.phi.reshape(self.N, self.dimX, self.dimX) if self.N == 1 else self.phi
        return C
    
    def _v(self, s, t):
        diff_cov = np.einsum('ijk,ilk->ijl', self.phi, self.phi) if self.N > 1 else (self.phi @ self.phi.T).reshape(1, self.dimX, self.dimX) # (N, dimX, dimX)
        A_plus_A_T = lambda rho: rho.reshape(1, self.dimX) + rho.reshape(self.dimX, 1) #\rho_ii + \rho_kk
        rho_sums = np.stack([A_plus_A_T((self.rho[i, :])) for i in range(self.N)]) # (N, dimX, dimX)
        cov = diff_cov/rho_sums * (1 - np.exp(-2*rho_sums*(t-s))) # (N, dimX, dimX)
        return cov
        
        
class MvFullOrnsteinUhlenbeck(MvOrnsteinUhlenbeck, MvEllipticSDE):
    """
    Multivariate Ornstein-Uhlenbeck process, given by:
    dX_t  = \rho(\mu - X_t) dt + \phi dW_t
    
    With \rho a dxd matrix
    \mu a 1xd vector
    and \phi a dxd matrix
    
    Input parameters: 
    'rho': (N, dimX, dimX)/(dimX, dimX) matrix of reversion rates
    'mu': (N, dimX) vector of means
    'phi': (N, dimX, dimX)/(dimX, dimX) diffusion matrix
    
    WARNING: Evaluation of the matrix exponential is required.
    
    To do: Implement the _v(s,t) method for this class. 
    Requires a general expression for the covariance matrix of the random vector
    \int_{s}^t {exp[-\rho(t-u)]\phi} dW_u
    
    Where \rho is a dxd matrix so we have a matrix exponential, \phi is dxd matrix and W_t is a dx1 Brownian motion.
    """ 

    @property
    def default_params(self):
        N, dx = self.N, self.dimX
        default_params = {'rho': 0.5*np.eye(N) if N == 1 else np.stack([0.5*np.eye(dx)]*N),
                    'mu': np.zeros((N, dx)),
                    'phi': np.eye(N) if N == 1 else np.stack([np.eye(dx)]*N)
                }
        return default_params
    
    @property
    def param_shapes(self):
        N, dx = self.N, self.dimX
        return {'rho': [(N, dx, dx), (dx, dx)], 'mu': [(N, dx)], 'phi': [(N, dx, dx), (dx, dx)]}

    def A(self, t):
        return np.einsum('ijk,ik->ij', self.rho, self.mu)
    
    def B(self, t):
        return -self.rho
    
    def _a(self, s, t):
        """
        Should be a (N, dimX, dimX) matrix.
        """
        return sla.expm(-self.rho*(t-s))
    
    def _b(self, s, t):
        """
        Should be a (N, dimX) vector.
        """
        return np.einsum('ijk,ik->ij', np.stack([np.eye(self.dimX)]*self.N) - sla.expm(-self.rho*(t-s)), self.mu)
    
    def _v(self, s, t):
        """
        Should be an (N, dimX, dimX) matrix.
        """
        return NotImplementedError(self._error_msg('_v'))

class FitzHughNagumo(MvEllipticSDE):
    dimX = 2
    default_params = {'rho': np.array([1.4, 1.5, 10.]),
                      'phi': np.array([0.25, 0.2])
                        }
    diag_cov = True

    def b(self, t, x):
        x_1 = self.rho[0] * ((-x[:, 0] ** 3) + x[:, 0] - x[:, 1] + 0.5) # (N, )
        x_2 = self.rho[1] * x[:, 0] - x[:, 1] + self.rho[2] # (N, )
        return np.stack([x_1, x_2], axis=1)

    def sigma(self, t, x):
        N = x.shape[0]
        return np.stack([np.diag(self.phi)]*N, axis=0)
    
    def db(self, t, x):
        N = x.shape[0]
        db_1_dx_1 = self.rho[0] * (3 * (x[:, 0] ** 2) + 1) # (N, )
        db_1_dx_2 = np.array([-self.rho[1]]*N)
        db_2_dx_1 = np.array([self.rho[2]]*N)
        db_2_dx_2 = np.array([-1.]*N)
        db = np.stack([db_1_dx_1, db_1_dx_2, db_2_dx_1, db_2_dx_2], axis=1).reshape(N, 2, 2)
        return db

    # def db_diag(self, t, x):
    #     N = x.shape[0]
    #     db_1_dx_1 = self.rho[0] * (3 * (x[:, 0] ** 2) + 1) # (N, )
    #     db_2_dx_2 = np.array([-1.]*N) #(N, )
    #     return np.stack([db_1_dx_1, db_2_dx_2], axis=1) # (N, 2)
        
    def dsigma(self, t, x):
        N = x.shape[0]
        return np.zeros((N, self.dimX, self.dimX, self.dimX))
    
class TimeSwitchingSDE(SDE):
    """
    General class for a diffusion process that switches between two possible regimes.
    
    To do: you could come back to this and change the API a bit, so that 
    the init method takes in the parameters of the two sdes and the switching time, 
    as opposed to the 2 SDEs themselves.
    
    Parameter to supply to the __init__ method are:
    - The parameters from sde1, appended with _1
    - The parameters from sde2, appended with _2
    - The switching time t_switch
    
    To constuct your own time-switching SDE, subclass and 
    define the two SDEs that you want to switch between as the class attributes
    
    - 'sde1_cls' 
    - 'sde2_cls'
    'simulate': None, 
    """
    t_switching_methods = {'simulate': None, 'b': 0, 'sigma': 0, 'db': 0, 'dsigma': 0, 'A': 0, 'B': 0, 'C': 0, '_a': 1, '_b': 1, '_v': 1}
    sde1_cls = None
    sde2_cls = None

    @property
    def params(self):
        sde1_params = {name + '_1': param for name, param in self.sde1.params.items()}
        sde2_params = {name + '_2': param for name, param in self.sde2.params.items()}
        params = {**sde1_params, **sde2_params, **{'t_switch': self.t_switch}}
        return params
    
    def __init__(self, **kwargs):
        sde1_kwargs = {k[:-2]: v for k, v in kwargs.items() if k.endswith('_1')}
        sde2_kwargs = {k[:-2]: v for k, v in kwargs.items() if k.endswith('_2')}
        if 'dimX' in kwargs.keys():
            sde1_kwargs.update({'dimX': kwargs['dimX']})
            sde2_kwargs.update({'dimX': kwargs['dimX']})
            del(kwargs['dimX'])
        self.sde1 = self.sde1_cls(**sde1_kwargs)
        self.sde2 = self.sde2_cls(**sde2_kwargs)
        self.N = 1
        SDE.__init__(self, **kwargs)
        self.base_sde_cls = self._gen_base_sde_cls()
        
        # Check that sde1 and sde2 have compatible dimensions
        assert self.sde1.dimX == self.sde2.dimX, 'Attribute dimX must be the same for both SDEs'
        assert self.sde1.dimW == self.sde2.dimW, 'Attribute dimW must be the same for both SDEs'

        sde1_methods = set(get_methods(self.sde1))
        sde2_methods = set(get_methods(self.sde2))
        methods = sde1_methods.intersection(sde2_methods)
        t_switching_methods = set(self.t_switching_methods.keys()).intersection(methods)
        base_methods = methods - t_switching_methods
        t_switching_methods = {method: self.t_switching_methods[method] for method in t_switching_methods}
        # Dynamically add methods t_switching methods from sde1 and sde2
        self._add_t_switching_methods(t_switching_methods)

        # Dynamically add methods from base_sde_cls
        self._add_base_methods(base_methods)

    @property
    def dimX(self):
        return self.sde1.dimX
    
    @property
    def dimW(self):
        return self.sde1.dimW

    @property
    def _diag_cov(self):
        return True if self.sde1._diag_cov and self.sde2._diag_cov else False
    
    @property
    def default_params(self):
        sde1_params = {name + '_1': param for name, param in self.sde1.default_params.items()}
        sde2_params = {name + '_2': param for name, param in self.sde2.default_params.items()}
        params = {**sde1_params, **sde2_params, **{'t_switch': 10}}
        return params

    def _add_base_methods(self, method_names):
        # Dynamically bind methods from base_sde_cls
        for method_name in method_names:
            if hasattr(self.base_sde_cls, method_name):
                method = getattr(self.base_sde_cls, method_name)
                # Bind the method to the instance
                setattr(self, method_name, method.__get__(self))

    def _gen_base_sde_cls(self):
        sde1_mro = self.sde1.__class__.mro()
        sde2_mro = self.sde2.__class__.mro()
        for cls in sde1_mro:
            if cls in sde2_mro:
                return cls
            
    def _add_t_switching_methods(self, method_dict):
        for method_name, t_idx in method_dict.items():
            m1 = getattr(self.sde1, method_name)
            m2 = getattr(self.sde2, method_name)
            method = self._t_switching_method(m1, m2, t_idx)
            setattr(self, method_name, method.__get__(self))

    def _t_switching_method(self, m1, m2, t_idx):
        def method(_self, *args, **kwargs):
            t = args[t_idx] if t_idx is not None else kwargs['t_end']
            cond = t < _self.t_switch if t_idx == 0 else t <= _self.t_switch
            return m1(*args, **kwargs) if cond else m2(*args, **kwargs)
            # else:
            #     pass   
            #     # Placeholder to deal with vectorised time in 1D case.             
            #     # cond = np.where(t - tol < _self.t_switch, 1., 0.)
            #     # return cond*m1(*args, **kwargs) 
        return method
        
class TS_MvOrnsteinUhlenbeck(TimeSwitchingSDE, MvLinearSDE, MvEllipticSDE):
    sde1_cls = MvOrnsteinUhlenbeck
    sde2_cls = MvOrnsteinUhlenbeck

class TS_OrnsteinUhlenbeck(TimeSwitchingSDE, LinearSDE):
    sde1_cls = OrnsteinUhlenbeck
    sde2_cls = OrnsteinUhlenbeck