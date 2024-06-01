"""
We use this module for functions that evaluate path integrals that are used as weights within particle filters in continuous time.
"""
import numpy as np

def log_girsanov(X: np.ndarray, b_1, b_2, Cov, step) -> np.ndarray:
    """
    Function to evaluate the weights of sample paths according to Girsanov's formula.
    Two SDEs must have common drift and diffusion coefficients.


    Inputs
    ------------
    X (np.ndarray):     A (N, num+1) array of N sample paths imputed at num+1 times
    b_1:                Drift function of the target SDE
    b_2:                Drift function of the simulated SDE
    Cov:                The common SDE covariance matrix of both SDEs
    step (float):       The timestep between each imputation. So the end time is step * num

    Returns
    ------------
    log_wgts (np.ndarray): A (N, ) array of the log weight of each sample path.
    """
    num_plus_1 = X.shape[1]
    N = X.shape[0]
    def dX_integrand(t, x):
        return (b_1(t, x) - b_2(t, x))/Cov(t, x)   
    def dt_integrand(t, x):
        return (np.square(b_1(t, x)) - np.square(b_2(t, x)))/Cov(t, x)
    times = np.linspace(0., step*(num_plus_1-1), num=num_plus_1)
    # times = np.stack([times for _ in range(N)], axis=0) # (N, num+1) array of times
    log_wgts = _log_girsanov_eval(X, times, dX_integrand, dt_integrand)
    return log_wgts

def _log_girsanov_eval(X: np.ndarray, times: np.ndarray, dX_integrand, dt_integrand) -> np.ndarray:
    """
    Critical function: all of the calculations that are used to evaluate the weights happen here.

    Inputs
    ------------
    X (np.ndarray):     A (N, num+1) array of N sample paths imputed at num+1 times
    times (np.ndarray): A (num+1, ) array of the path times
    dX_integrand:       Function of the path that is taken inside the dX integral of Girsanov
    dt_integrand:       Function of the path that is taken inside the dt integral of Girsanov

    Returns
    ------------
    log_wgts (np.ndarray): A (N, ) array of the log weight of each sample path.
    """
    dX_integrand_vals = dX_integrand(times, X) # (N, num+1) # (num+1, ), (N, num+1)
    dt_integrand_vals = dt_integrand(times, X) # (N, num+1)
    dXs = X[:, 1:] - X[: , :-1]
    dts = times[1:] - times[:-1]
    dX_integral_vals = dX_integrand_vals[:, :-1] * dXs #(N, num), (N, num)
    dX_integral_vals = dX_integral_vals.sum(axis=1)
    dt_integral_vals = dt_integrand_vals[:, :-1] * dts # (N, num), (N,num)
    dt_integral_vals = dt_integral_vals.sum(axis=1)
    log_wgts = dX_integral_vals - 0.5 * dt_integral_vals
    return log_wgts

# def log_delyon_hu(X: np.ndarray, b, Cov, step):
#     num_plus_1 = X.shape[1]
#     Delta_s = (num_plus_1 - 1) * step
#     N = X.shape[0]
#     def dX_integrand(t, x):
#         return (b(t, x))/Cov(t, x)   
#     def dt_integrand(t, x):
#         return np.square(b(t, x))/Cov(t, x)
#     x_end = X[:, -1]
#     def dZ_integrand(t, x):
#         numer = (x_end - x.T).T # x_end (N, ), x is (num+1, N) so use x.T so that broadcasting works.
#         numer = numer[:, :-1] # Remove end point to avoid zero division error
#         denom = Delta_s - times
#         denom = 
#         return numer/denom
#     times = np.linspace(0., step*Delta_s, num=num_plus_1)
#     # times = np.stack([times for _ in range(N)], axis=0) # (N, num+1) array of times
#     log_wgts = _log_girsanov_eval(X, times, dX_integrand, dt_integrand)
#     return log_wgts
   

# def numer_integrate(integrand, integrator):
#     """
#     Inputs
#     ------------
#     integrand (np.ndarray): ()
#     integrator (np.ndarray): (num+1, ) array 

#     Returns
#     ------------
#     integral (np.ndarray): A (N, ) array of the log weight of each sample path.
#     """    """