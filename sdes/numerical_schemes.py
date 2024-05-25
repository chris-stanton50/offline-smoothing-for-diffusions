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

import particles.distributions as dists
import sdes.distributions as sdists
import numpy as np
import scipy.stats as stats


class NumericalScheme(object):
    """
    Abstract base class for building numerical simulation schemes for SDEs.
    """
    def __init__(self, SDE):
        self.SDE = SDE

    @property
    def default_x_start(self):
        return 0. if self.SDE.isunivariate else np.zeros((1, self.SDE.dimX))
    
    def _error_msg(self, method):
        return ('method ' + method + ' not implemented in SDE numerical scheme class%s' %
                self.__class__.__name__)

    def simulate(self, size, t_start=0., t_end=1., x_start=None, num=5):
        """
        Generates a simulation of sample paths of the SDE using the numerical scheme.
        """
        if self.SDE.dimX > 1 or self.SDE.dimW > 1:
            raise NotImplementedError('Simulation function yet to be implemented for SDES that are not univariate.')
        x_start = self.default_x_start if x_start is None else x_start
        t_diff = t_end - t_start
        param_names = self._gen_param_names(t_diff, num)
        if self.SDE.dimX == 1:
            dtype = [(param_name, 'float64') for param_name in param_names]
        else:
            # This will be needed when extending to higher dimensions
            dtype = [(param_name, 'float64', self.SDE.dimX) for param_name in param_names]
        sims = np.empty(size, dtype=dtype)
        step, first_param = t_diff/num, param_names[0]
        sims[first_param] = self.univ_simulation_step(size, t_start, x_start, step)
        for i in range(1, num):
            prev_time, curr_param, prev_param = t_start + i*step, param_names[i], param_names[i-1]
            sims[curr_param] = self.univ_simulation_step(size, prev_time, sims[prev_param], step)
        return sims

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

    # @property
    # def initial_dist(self):
    #     return self.univ_initial_dist if self.SDE.isunivariate else self.mv_initial_dist
    
    # @property
    # def transition_dist(self):
    #     return self.univ_transition_dist if self.SDE.isunivariate else self.mv_transition_dist

    # Distribution functionality currently not being used.

    # def univ_initial_dist(self, t, x, step):
    #     """
    #     This method expects input x to be a scalar for x
    #     """
    #     raise NotImplementedError(self._error_msg('univ_initial_dist'))

    # def mv_initial_dist(self, t, x, step):
    #     """
    #     This method expects input x to be an ndarray of shape (1, dimX) for x
    #     """
    #     raise NotImplementedError(self._error_msg('mv_initial_dist'))

    # def univ_transition_dist(self, t, x, step):
    #     """
    #     This method expects input ndarray of shape (N,) for x
    #     """
    #     raise NotImplementedError(self._error_msg('univ_transition_dist'))
    
    # def mv_transition_dist(self, t, x, step):
    #     """
    #     This method expects input ndarray of shape (N, dimX) for x
    #     """
    #     raise NotImplementedError(self._error_msg('mv_transition_dist'))
    
    # def distribution(self, t_start=0., t_end=1., x_start=None, num=5):
    #     """
    #     Creates a distribution object that represents the joint distribution 
    #     of the SDE at regular time intervals.
    #     """
    #     x_start = self.default_x_start if x_start is None else x_start
    #     t_diff = t_end - t_start
    #     param_names = self._gen_param_names(t_diff, num)
    #     law_dict, step, first_param = {}, t_diff/num, param_names[0]
    #     law_dict[first_param] = self.initial_dist(t_start, x_start, step)
    #     for i in range(1, num):
    #         prev_time, curr_param, prev_param = t_start + i*step, param_names[i], param_names[i-1]
    #         cond_law_fn = self._cond_law_fn_builder(prev_time, prev_param, step)
    #         law_dict[curr_param] = dists.Cond(cond_law_fn, dim=self.SDE.dimX)
    #     return dists.StructDist(law_dict)

    # def _cond_law_fn_builder(self, prev_time, label, step):
    #     def cond_law_fn(x):
    #             return self.transition_dist(prev_time, x[label], step)
    #     return cond_law_fn

class EulerMaruyama(NumericalScheme):

    def univ_simulation_step(self, size, t, x, step):
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
        A discretised transform from a structured array of solutions to the SDE, to the Brownian motion.
        """
        W = np.empty_like(X); t_s = X.dtype.names; delta_t = float(t_s[0]); curr_t = t_start
        W[t_s[0]] = self._X_to_W_step(curr_t, np.zeros_like([X[t_s[0]]]), x_start, X[t_s[0]], delta_t)
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
        """
        t_s = W.dtype.names; delta_t = float(t_s[0]); curr_t = t_start
        X = np.empty_like(W); x_end = W[t_s[-1]]
        X[t_s[0]] = self._W_to_X_step(curr_t, x_start, np.zeros_like(W[t_s[0]]), W[t_s[0]], delta_t)
        for i in range(1, len(t_s) - 1):
            curr_t += delta_t
            X[t_s[i]] = self._W_to_X_step(curr_t, X[t_s[i-1]], W[t_s[i-1]], W[t_s[i]], delta_t)
        if transform_end_point:
            curr_t += delta_t
            X[t_s[-1]] = self._W_to_X_step(curr_t, X[t_s[-2]], W[t_s[-2]], W[t_s[-1]], delta_t)
        else:
            X[t_s[-1]] = x_end
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

    # def _cond_loc(self, t, x, step):
    #     return x + self.SDE.b(t, x) * step

    # def _cond_cov(self, t, x, step):
    #     return self.SDE.Cov(t, x) * step

    # def univ_initial_dist(self, t, x, step):
    #     return self.univ_transition_dist(t, x, step)
    
    # def univ_transition_dist(self, t, x, step):
    #     trans_loc = self._cond_loc(t, x, step)
    #     trans_scale = np.sqrt(self._cond_cov(t, x, step))
    #     return dists.Normal(loc=trans_loc, scale=trans_scale)

    # def mv_initial_dist(self, t, x, step): # Expected Input of 'x' is (1, dimX)
    #     """
    #     """
    #     trans_loc = self._cond_loc(t, x, step)
    #     trans_cov = self._cond_cov(t, x, step)
    #     return dists.MvNormal(loc=trans_loc, cov=trans_cov)
    
    # def mv_transition_dist(self, t, x, step): # Expected Input of 'x' is (N, dimX)
    #     """
    #     Note: when 'MvNormal' class is instantiated, the covariance matrix fed in
    #     is Cholesky factorised. This will add extra compute cost. This could be 
    #     corrected in the future, but leave it for now.

    #     The correction will involve redefining the '__init__' method of
    #     the MvNormal class, so that the square root matrix is taken as
    #     an input instead of the covariance matrix.
    #     """
    #     if x.shape[0] == 1: # This would occur when we are only simulating one particle, so covariance does not vary.
    #         return self.mv_initial_dist(t, x, step)
    #     else:
    #         trans_loc = self._cond_loc(t, x, step) # This should be fine: we would expect this to naturally map from (N, dimX) to (N, dimX)
    #         apply_fn = lambda y: self._cond_cov(t, y.reshape((1, self.SDE.dimX)), step)
    #         trans_cov = np.apply_along_axis(apply_fn, 1, x)  # These lines ensure that each (1, dimX) vector is input seperately.
    #         return sdists.VaryingCovNormal(loc=trans_loc, cov=trans_cov)

class Milstein(NumericalScheme):
    """
    The Milstein scheme. Usage requires that the first derivative of the diffusion coefficient: dsigma has been defined.
    """
    def univ_simulation_step(self, size, t, x, step):
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
    def univ_simulation_step(self, size, t, x, step):
        Z = stats.norm.rvs(size=size)
        loc = self.SDE._a(t, t+step)* x + self.SDE._b(t, t+step)
        scale = np.sqrt(self.SDE._v(t, t+step))
        x_step = loc + scale * Z
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