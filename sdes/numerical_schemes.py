"""
Numerical Schemes module:

In this module, we define objects representing numercial approximation schemes 
that can be used to approximate the solution of an SDE. If the transition density
of the numerical scheme is available, then its transition density is also implemented.

Both weak and strong Ito-Taylor approximations are possible, see the Kloeden and Platten (1992)
for standard SDEs, and see Platen, Bruti Liberati (2010) for the case of SDEs with jumps.

If the SDE in question is of dimension one (in both the solution and the driving noise), then it
is possible to generate exact simulations from the SDE. These may be implemented here too.  
"""

from sdes.tools import method_match_first_dim
import numpy as np
import scipy.linalg as sla
import numpy.linalg as nla
import scipy.stats as stats


class NumericalScheme(object):
    """
    Abstract base class for building numerical simulation schemes for SDEs.
    """
    def __init__(self, SDE):
        self.SDE = SDE

    @property
    def default_x_start(self):
        if self.SDE.dimX == 1:
            return 0.
        if hasattr(self.SDE, 'N'):
            return np.zeros((self.SDE.N, self.SDE.dimX))
        else: 
            return np.zeros((1, self.SDE.dimX))

    def _error_msg(self, method):
        return ('method ' + method + ' not implemented in SDE numerical scheme class%s' %
                self.__class__.__name__)

    def simulate(self, size, t_start=0., t_end=1., x_start=None, num=5):
        """
        Generates a simulation of sample paths of the SDE using the numerical scheme.
        
        For MV case: x_start is a (N, dimX) array,
                    size = N
        
        """
        x_start = self.default_x_start if x_start is None else x_start
        t_diff = t_end - t_start
        sims = self._create_state_container(t_diff, num, size, dimX=self.SDE.dimX)
        param_names = sims.dtype.names
        step, first_param = t_diff/num, param_names[0]
        sims[first_param] = self.simulation_step(size, t_start, x_start, step)
        for i in range(1, num):
            prev_time, curr_param, prev_param = t_start + i*step, param_names[i], param_names[i-1]
            sims[curr_param] = self.simulation_step(size, prev_time, sims[prev_param], step)
        return sims
    
    def _create_state_container(self, t_diff, num, size, dimX=1):
        param_names = self._gen_param_names(t_diff, num)
        if dimX == 1:
            dtype = [(param_name, 'float64') for param_name in param_names]
        else:
            dtype = [(param_name, 'float64', dimX) for param_name in param_names]
        state_container = np.empty(size, dtype=dtype)
        return state_container

    def _state_container_size(self, X: np.ndarray, x_start):
        X_shape_idx = 1 if X.shape == () else X.shape[0]
        x_start_shape_idx = 1 if x_start.shape == () else x_start.shape[0]
        size = max(X_shape_idx, x_start_shape_idx)
        return size

    def _gen_param_names(self, T, num):
        delta = T/num
        n_sig_figs = self._get_rounding(delta)
        param_names = [str(round(i*delta, n_sig_figs)) for i in range(1, num+1)]
        return param_names 

    def _get_rounding(self, delta, max_round=5):
        c = 0
        while (round(delta) != delta) & (c <= max_round):
            delta *= 10
            c += 1
        return c

class MvNumericalScheme(NumericalScheme):

    def mv_state_container_size(X: np.ndarray, x_start):
        """ May need to change this later"""
        return max(X.shape[0], x_start.shape[0])

class EulerMaruyama(NumericalScheme):

    def __init__(self, SDE):
        if SDE.dimX != SDE.dimW:
            raise ValueError('SDE must be elliptic (dimX = dimW) for Euler-Maruyama scheme')
        super().__init__(SDE)

    def simulation_step(self, size, t, x, step):
        """
        Given starting point(s) 'x' and a time 't', uses the Euler-Maruyama scheme
        to simulate the SDE at time 't+step':

        This is a *critical* function. This will be called multiple times in any SMC
        algorithm, so speeding this up could have significant impacts on algorithm performance.
        
        Inputs
        -------------
        size (int): The number of SDE paths.
        t (float): The current time 
        x (float/np.ndarray): Either a float, or a (size, ) array for the number of samples
        step (float): The step size

        Returns
        -------------
        x_step (np.ndarray): An array of shape (size, ) for the SDE paths at time t+step

        Needs the drift and diffusion coefficients of the SDE to broadcast (N, ) input to an (N, ) output.
        """
        Z = stats.norm.rvs(size=size)
        x_step = x + self.SDE.b(t, x)*step + self.SDE.sigma(t, x) * np.sqrt(step) * Z
        return x_step
        
    def transform_X_to_W(self, X, t_start=0., x_start=0., transform_end_point=True):
        """
        Use cases:

        For forward method, reparameterised fk models, transforms the X process into Brownian noise.

        X: ND struct array of different paths (or could stack the input and feed it like this)
        t_start: 0.
        x_start: (N, ) (dimX=1) / (N, dimX) (dimX > 1) array of end points of paths from particles from previous timestep
        transform_end_point: False 
        """
        t_s = X.dtype.names; 
        num, delta_t, t_end = len(t_s), float(t_s[0]), float(t_s[-1])
        t_diff, curr_t = t_end - t_start, t_start        
        size = self._state_container_size(X, x_start)
        W = self._create_state_container(t_diff, num, size, dimX=self.SDE.dimW)
        W[t_s[0]] = self._X_to_W_step(curr_t, np.zeros(X[t_s[0]].shape), x_start, X[t_s[0]], delta_t)
        for i in range(1, len(t_s) - 1):
            curr_t += delta_t
            W[t_s[i]] = self._X_to_W_step(curr_t, W[t_s[i-1]], X[t_s[i-1]], X[t_s[i]], delta_t)
        if transform_end_point:
            curr_t += delta_t
            W[t_s[-1]] = self._X_to_W_step(curr_t, W[t_s[-2]], X[t_s[-2]], X[t_s[-1]], delta_t)
        else:
            W[t_s[-1]] = X[t_s[-1]]        
        return W
        
    def transform_W_to_X(self, W, t_start=0., x_start=0., transform_end_point=True):
        """
        An discretised transform from the simulation of a Brownian motion to the solution of the SDE:
        We can change the initialisation so we don't have to stack the 1D struct array.

        Use cases:
        Transforming noise to aux bridge in backward sampling algos:

        W: 1D struct array of a single path (or could stack the input and feed it like this)
        t_start: 0.
        x_start: (N, ) array of end points of paths from particles from previous timestep
        transform_end_point: False

        Evaluating potentials G_t in reparameterised fk models:

        W: ND struct array of different paths (or could stack the input and feed it like this)
        t_start: 0.
        x_start: (N, ) array of end points of paths from particles from previous timestep
        transform_end_point: False
        
        Evaluating unnormalised density of theta for MWG updates within Particle Gibbs:

        W: (1, ) struct array of different paths
        t_start: 0.
        x_start: (1, ) array of end points of paths from particles from previous timestep
        transform_end_point: False  
        """
        t_s = W.dtype.names
        num, delta_t, t_end = len(t_s), float(t_s[0]), float(t_s[-1])
        t_diff, curr_t = t_end - t_start, t_start
        size = self._state_container_size(W, x_start)
        X = self._create_state_container(t_diff, num, size, dimX=self.SDE.dimX)
        x_end = W[t_s[-1]]
        X[t_s[0]] = self._W_to_X_step(curr_t, x_start, 0., W[t_s[0]], delta_t)
        for i in range(1, len(t_s) - 1):
            curr_t += delta_t
            X[t_s[i]] = self._W_to_X_step(curr_t, X[t_s[i-1]], W[t_s[i-1]], W[t_s[i]], delta_t)
        if transform_end_point:
            curr_t += delta_t
            X[t_s[-1]] = self._W_to_X_step(curr_t, X[t_s[-2]], W[t_s[-2]], W[t_s[-1]], delta_t)
        else:
            X[t_s[-1]] = x_end*np.ones(size) if type(x_end) == float else x_end
        return X
    
    def _W_to_X_step(self, t, x, w, w_next, step):
        """       
        """  
        x_next =  x + self.SDE.b(t, x)*step + self.SDE.sigma(t, x) * (w_next - w)
        return x_next

    def _X_to_W_step(self, t, w, x, x_next, step):
        """
        """
        w_next = w + ((x_next - x) - self.SDE.b(t, x)*step)/self.SDE.sigma(t, x)
        return w_next
    
class Milstein(NumericalScheme):
    """
    The Milstein scheme. Usage requires that the first derivative of the diffusion coefficient: dsigma has been defined.
    """
    def simulation_step(self, size, t, x, step):
        """
        Simulation step for the Milstein scheme (Kloeden & Platen 1990), p345.        
        """
        Z = stats.norm.rvs(size=size)
        a = self.SDE.b(t, x) - 0.5*self.SDE.sigma(t, x)*self.SDE.dsigma(t ,x)
        b = self.SDE.sigma(t, x)
        c = 0.5*self.SDE.sigma(t, x)*self.SDE.dsigma(t ,x)
        x_step = x + a*step + b * np.sqrt(step) * Z + c * step * np.square(Z) 
        return x_step

class LinearExact(NumericalScheme):
    """
    Exact simulation from a linear SDE.
    """
    def simulation_step(self, size, t, x, step):
        Z = stats.norm.rvs(size=size)
        loc = self.SDE._a(t, t+step)* x + self.SDE._b(t, t+step)
        scale = np.sqrt(self.SDE._v(t, t+step))
        x_step = loc + scale * Z
        return x_step    

class MvEulerMaruyama(EulerMaruyama, MvNumericalScheme):
    
    def simulation_step(self, size, t, x, step):
        """
        Given starting point(s) 'x' and a time 't', uses the Euler-Maruyama scheme
        to simulate the SDE at time 't+step':

        This is a *critical* function. This will be called multiple times in any SMC
        algorithm, so speeding this up could have significant impacts on algorithm performance.
        
        Inputs
        -------------
        size (int): The number of SDE paths.
        t (float): The current time 
        x (np.ndarray): A (size, dimX) array for the current point on the SDE.
        step (float): The step size

        Returns
        -------------
        x_step (np.ndarray): An array of shape (size, dimX) for the SDE paths at time t+step
        """
        # Compute drift and diffusion
        drift = self.SDE.b(t, x)  # shape (N, d)
        diffusion = self.SDE.sigma(t, x)  # shape (n_sim, d, m)
        # Generate Brownian increments
        dW = np.sqrt(step) * stats.norm.rvs(size=(size, self.SDE.dimW))
        # Euler-Maruyama update
        x_step = x + drift * step + np.einsum('ijk,ik->ij', diffusion, dW)
        return x_step

    @method_match_first_dim
    def _W_to_X_step(self, t, x, w, w_next, step):
        """       
        """
        drift = self.SDE.b(t, x) # shape (1/N, d)
        diffusion = self.SDE.sigma(t, x) # shape (1/N, d, d)  
        x_next =  x + drift*step + np.einsum('ijk,ik->ij', diffusion, w_next - w)
        return x_next

    @method_match_first_dim
    def _X_to_W_step(self, t, w, x, x_next, step):
        """
        """
        drift = self.SDE.b(t, x) # shape (1/N, d)
        diffusion = self.SDE.sigma(t, x) # shape (1/N, d, d)  
        # inv_diff = nla.inv(diffusion) # Talk to Alex about this: this is unstable!
        # w_next = w + np.einsum('ijk,ik->ij', inv_diff, ((x_next - x) - drift*step))
        w_next = w + nla.solve(diffusion, ((x_next - x) - drift*step))
        return w_next
        
class MvLinearExact(MvNumericalScheme):
    
    def simulation_step(self, size, t, x, step):
        """
        Inputs: 
        ---------
        size: int, number of particles
        t: float, current time
        x: np.ndarray, (size, dimX)/(1, dimX), current state
        step: float, step size
        
        Returns:
        --------
        x_step: np.ndarray, (size, dimX), next state
        """
        N = self.SDE.N
        if size != N and N != 1:
            raise ValueError('size must be equal to the number of particles')
        if x.shape[0] == 1 and N == 1:
            x = np.concatenate([x]*size, axis=0)
        a = self.SDE._a(t, t+step); b = self.SDE._b(t, t+step); v = self.SDE._v(t, t+step)
        _a = a if N > 1 else np.concatenate([a]*size) # (size, dimX, dimX)
        _b = b if N > 1 else np.concatenate([b]*size) # (size, dimX)
        _v = v if N > 1 else np.concatenate([v]*size) # (size, dimX, dimX)
        Z = stats.norm.rvs(size=(size, self.SDE.dimX))
        loc = np.einsum('ijk,ik->ij', _a, x) + _b # (size, dimX)
        scale = sla.cholesky(_v, lower=True) # (size, dimX, dimX)
        x_step = loc + np.einsum('ijk,ik->ij', scale, Z) # (N, dimX)
        return x_step

"""
If we consider univariate SDEs only, one can immediately define the Milstein scheme and simulate from it by going through the derivations
of the transition density of the Milstein scheme and implementing it as a new distribution (subclass of ProbDist). Care will need to be 
taken to ensure that when N simulations are being generated, the broadcasting of the 1darrays of shape (N,) is done correctly. It will
also be necessary to define the first derivative of the sigma function w.r.t x. This could be done directly  by the user, or through using 
automatic differentiation (e.g through autograd/jax).

If one wishes to implement further numerical schemes in one dimension, it may be helpful to construct methods in the numerical_scheme 
e.g `simulate`, `univ_simulation_step`, `mv_simulation_step`, `simulation_step` that are used to generate a simulation from a numerical scheme.
The output of the `simulate` function can be the same structured dtype ndarray output as one constructed from building the distribution object
with the `distribution` method and simulating from that distribution with its `rvs` method. The `univ_simulation_step` and `mv_simulation_step`
functions can then be designed seperately for each different numerical scheme. Again, care will need to be taken to ensure that the broadcasting
is done correctly when N particles are simulated.

Once new numerical methods have been implemented, the way that they interact with the SDE classes can be changed so that it is possible to select
a different numerical scheme with which to run the simulation.
"""