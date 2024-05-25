# -*- coding: utf-8 -*-
"""
SDEs module
"""
import numpy as np 
import scipy.stats as stats
from sdes.numerical_schemes import EulerMaruyama, Milstein, LinearExact
from sdes.tools import grad_log_linear_gaussian

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

    def __init__(self, **kwargs):
        if hasattr(self, 'default_params'):
            self.__dict__.update(self.default_params)
        self.__dict__.update(kwargs)

    @property
    def isunivariate(self):
        return (self.dimX == 1 and self.dimW == 1)

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

    def simulate(self, size: int, x_start, t_start: float =0., t_end: float =1., num=5, milstein=False) -> np.ndarray:
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
        milstein (bool):       Whether to use the higher order Milstein scheme. Is False by default. Requires 'dsigma'
                              to use.    

        Returns
        ------------
        simulations (np.ndarray): A structured array (structured by timestamps) containing simulation outputs. 
        """
        self.numerical_scheme = EulerMaruyama(self) if not milstein else Milstein(self)
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
        W =self.numerical_scheme.transform_X_to_W(X, t_start=t_start, x_start=x_start, transform_end_point=True)
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

    # def distribution(self, **kwargs):
    #     """
    #     Use of distribution functionality is currently deprecated.
    #     It does not make much sense to construct distribution objects for SDEs 
    #     in the same way that the particles package does for finite-dimensional distributions.
    #     The algorithms are designed under the 'simulation-projection' paradigm, so we work
    #     directly with discretisations of continuous-time likelihood expressions, as 
    #     opposed to under `projection-simulation', where we work with the product of 
    #     transition densities.
    #     """
    #     return self.numerical_scheme.distribution(**kwargs)

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

    Linear SDEs have a tractable Gaussian transition density, and so can be simulated from exactly:

    p_{s,t}(x_t| x_s) = \phi(x_t, a x_s + b, v)

    a, b and v correspond to _a, _b and _v methods.

    The _a, _b and _v methods need to be implemented for a given subclass of LinearSDE. 
    Doing so defines the transition density of the Linear SDE, thus enabling exact sampling,
    and evaluation of the transition density.

    For the general linear SDE the quantities involve evaluating various integrals.
    We may come back to implement this in the future.
    """
        
    def A(self, t):
        raise NotImplementedError(self._error_msg('A'))
    
    def B(self, t):
        raise NotImplementedError(self._error_msg('B'))

    def C(self, t):
        raise NotImplementedError(self._error_msg('C'))

    def _a(self, s, t, x_0):
        "Needs to be implemented to use LinearSDEs for subclassing"
        return NotImplementedError(self._error_msg('_a'))

    def _b(self, s, t, x_0):
        "Needs to be implemented to use LinearSDEs for subclassing"
        return NotImplementedError(self._error_msg('_b')) 
    
    def _v(self, s, t):
        "Needs to be implemented to use LinearSDEs for subclassing"
        return NotImplementedError(self._error_msg('_v'))
    
    def b(self, t, x):
        return self.A(t) + self.B(t) * x
    
    def sigma(self, t, x):
        return self.C(t)

    def db(self, t, x):
        return self.B(t)
    
    def dsigma(self, t, x):
        return 0.
    
    def log_px(self, s: float, t: float, x_s: np.ndarray, x_t: np.ndarray):
        """
        log transition density
        """
        px_mean = self._a(s, t)*x_s + self._b(s, t)
        px_sd = np.sqrt(self._v(s, t))
        return stats.norm.logpdf(x_t, loc = px_mean, scale = px_sd)
    
    def grad_log_px(self, s: float, t: float, x_s: np.ndarray, x_t: np.ndarray):
        """
        Gradient of the log transition density
        """
        return grad_log_linear_gaussian(x_s, x_t, self._a(s, t), self._b(s, t), self._v(s, t))

    def grad_log_py(self, s: float, t: float, x_s: np.ndarray, y_t: np.ndarray, eta_sq: float):
        """
        Gradient of the log transition density of Y_t | X_s, where Y_t | X_t \sim N(0, \eta^2)
        """
        return grad_log_linear_gaussian(x_s, y_t, self._a(s, t), self._b(s, t), self._v(s, t) + eta_sq)
    
    def exact_simulate(self, size: int, x_0, x_start, t_start: float = 0., t_end: float = 1., num=5) -> np.ndarray:
        """
        Exact simulation of the linear SDE
        """
        self.x_0 = x_0 #Transition density of a general Linear SDE may depend on start point x_0.
        exact_scheme = LinearExact(self)
        return exact_scheme.simulate(size=size, t_start=t_start, t_end=t_end, x_start=x_start, num=num)


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
    
    def int_A(self, t):
        raise NotImplementedError(self._error_msg('A'))
    
    def int_C_sq(self, t):
        raise NotImplementedError(self._error_msg('int_C_sq'))

    def _a(self, s, t):
        return 1.

    def _b(self, s, t):
        return self.int_A(t) - self.int_A(s)
    
    def _v(self, s, t):
        return self.int_C_sq(t) - self.int_C_sq(s)

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

    def int_A(self, t):
        return self.m * t
    
    def C(self, t):
        return self.s
    
    def int_C_sq(self, t):
        return (self.s ** 2) * t  

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

    def _a(self, s, t):
        return np.exp(-self.rho * (t-s))
    
    def _b(self, s, t):
        return self.mu * (1. - np.exp(-self.rho * (t-s)))
    
    def _v(self, s, t):
        return ((self.phi ** 2)/(2*self.rho)) * (1 - np.exp(-2*self.rho*(t-s)))

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
    # Default dimensionality settings

    def __init__(self, **kwargs):
        if self.isunivariate:
            raise Exception('dimX and dimW must be specified as class attributes for a MvSDE')
        super().__init__(**kwargs)

    def b(self, t: float, x: np.ndarray):
        """  
            Input  an array-like of shape (1, dimX)
            Output an ndarray of shape (1, dimX)
            Should broadcast an (N, dimX) input to an (N, dimX) output.
        """
        raise NotImplementedError(self._error_msg('b'))
    
    def sigma(self, t: float, x: np.ndarray):
        """  
            Input an array-like of shape (1, dimX)
            Output an ndarray of shape (dimX, dimW)
        """
        raise NotImplementedError(self._error_msg('sigma'))
    
    def Cov(self, t: float, x: np.ndarray): 
        rt_cov = self.sigma(t, x)          
        return np.matmul(rt_cov, rt_cov.T)