"""
For now, use this module as a miscellaneous dump for various useful functions.
These functions are predominantly used in notebooks to speed up the presentation
of some results.
"""

import numpy as np
import matplotlib.pyplot as plt


def grad_log_linear_gaussian(x_s: np.ndarray, x_t: np.ndarray, a, b, s):
    """
    The gradient of the log of a linear Gaussian transition density 
    X_t | X_s = x_s \sim \mathcal{N}(a x_s + b, s)
    """
    return (a* (x_t - a*x_s - b))/s

def struct_array_to_array(struct_X):
    """
    Utility function to convert structured array consisting of paths from the proposal SDE to 
    a unstructured numpy array. For use in the context of 1D SDEs.

    Inputs
    ----------
    struct_X: Structured array, containing the sample paths generated from an SDE object.
    
    Returns
    ----------
    X: Unstructured array, of dimension (num, N)

    where num is the number of imputed points, and N is the number of particles
    """
    X = np.array([struct_X[name] for name in struct_X.dtype.names])
    return X

def start_points_paths_to_array(x_start, X):
    """
    Utility function to convert structured array of paths from the proposal SDE 
    and an unstructured vector of start points into an unstrucutured numpy array
    that contains the start points follow by the paths. For use in the context of 
    1D SDEs.

    Inputs
    ----------
    x_start: np.array of shape (N, ) where N is the number of particles
    X: Structured array, containing the sample paths.

    Returns
    --------
    X_array: An unstructured numpy array of shape (N, num+1)

    Where N is the number of particles, and num is the number of imputed points. 
    """
    N = len(x_start)
    x_start = x_start.reshape(1, N)
    X_array = struct_array_to_array(X)
    X_array = np.concatenate([x_start, X_array]).T
    return X_array

def log_girsanov(X, target_sde, proposal_sde):
    """
    First stab at building a function to calculate weights according to Girsanov's formula.
    Proposal and target SDEs must be univariate, and must have the same diffusion coefficient. 
    Move this into your practical implementation in the future.

    The log Girsanov weight is given by:

    $$ \int_{0}^T (b_2(X_t) - b_1(X_t))/\sigma(X_t) dX_t - 0.5 \int_{0}^T (b_2^2(X_t) - b_1^2(X_t))/\sigma(X_t)dt $$
    """
    array_X = struct_array_to_array(X).T
    names = X.dtype.names; delta = float(names[1]) - float(names[0])
    times = np.array([float(name) for name in names])
    b_1 = proposal_sde.b; b_2 = target_sde.b; sigma = target_sde.sigma
    B_1 = b_1(times, array_X)[:, :-1]
    B_2 = b_2(times, array_X)[:, :-1] 
    Sigma = sigma(times, array_X)[:, :-1]
    dX_integral = (B_2 - B_1) * (array_X[:, 1:] - array_X[:, :-1]) / Sigma
    dt_integral = delta*(np.square(B_2) - np.square(B_1))/Sigma
    dX_integral = dX_integral.sum(axis=1)
    dt_integral = dt_integral.sum(axis=1)
    return dX_integral - 0.5 * dt_integral


def univariate_simulation_test(sde, nums, dist_kwargs):
    fig, ax = plt.subplots()
    for num in nums:
        dist_kwargs['num'] = num
        rvs = sde.simulate(size=1, **dist_kwargs)
        t_s = [dist_kwargs['t_start']] + [float(t) for t in rvs.dtype.names]
        X_ts = struct_array_to_array(rvs); X_ts = np.concatenate([np.array([dist_kwargs['init_x']]), X_ts])
        ax.plot(t_s, X_ts, label=f'n_points = {num}')
    # Configure axis settings
    ax.legend(); ax.set_xlabel('t'); ax.set_ylabel('X_t'); ax.grid(visible=True)
    return fig, ax

def plot_simulations(X):
    fig, ax = plt.subplots()
    t_s = [float(name) for name in X.dtype.names]
    data = np.stack([X[name] for name in X.dtype.names])
    ax.plot(t_s, data) # Might need to add a transpose here
    return fig, ax