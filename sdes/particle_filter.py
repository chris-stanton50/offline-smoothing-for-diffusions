import numpy as np
import scipy.stats as stats

# Length of the data T, size of the data N:
# First, come up with a basic particle filter in one dimension, then consider higher dimensions.

default_params = {'sigma_x': 1.,
          'sigma_y': 1.,
          'sigma_x_0': 1.,
          }

def log_p_0(x_0, sigma_x_0=1.):
    """ N(0, \sigma_{x_0})"""
    return stats.norm.logpdf(x_0, scale=sigma_x_0)

def sim_p_0(N, sigma_x=1.):
    """ """
    return stats.norm.rvs(scale=sigma_x, size=N)

def log_p_t(x_t, x_t_prev, sigma_x=1.):
    """N(x_{t-1}, \sigma_x)"""
    return stats.norm.logpdf(x_t, loc=x_t_prev, scale=sigma_x)

def log_f_t(y_t, x_t, sigma_y=1.):
    """N(x_t, \sigma_y)"""
    return stats.norm.logpdf(y_t, loc=x_t, scale=sigma_y)

def sim_p_t(N, x_t_prev, sigma_x=1.):
    return stats.norm.rvs(loc=x_t_prev, scale=sigma_x, size=N)

def sim_f_t(N, x_t, sigma_y=1.):
    return stats.norm.rvs(loc=x_t, scale=sigma_y, size=N)

def simulate(T, params):
    X = np.empty(T)
    Y = np.empty(T)
    X[0] = sim_p_0(1, sigma_x=params['sigma_x'])
    Y[0] = sim_f_t(1, X[0], sigma_y=params['sigma_y'])
    for t in range(1, T):
        X[t] = sim_p_t(1, X[t-1], sigma_x=1.)
        Y[t] = sim_f_t(1, X[t], sigma_y=1.)
    return X, Y

def particle_filter(N, T, data, params):
    """
    """
    X = np.empty((N, T)); W = np.empty((N, T)); A = np.empty((N, T-1))
    X[:, 0] = sim_p_0(N, sigma_x=params['sigma_x'])
    logwgts = log_f_t(data[0], X[:, 0], sigma_y=params['sigma_y'])
    wgts = np.exp(logwgts)
    W[:, 0] = wgts/np.sum(wgts)
    for t in range(1, T):
        A[:, t-1] = np.random.choice(np.arange(N), size=N) # Resample at every step: no adaptive resampling
        X[:, t] = sim_p_t(N, X[:, t-1], sigma_x=params['sigma_x'])
        logwgts = log_f_t(data[t], X[:, t], sigma_y=params['sigma_y'])
        wgts = np.exp(logwgts)
        W[:, t] = wgts/np.sum(wgts)
    return X, W, A

# # Test the particle filter:
# T=100; N=100

# X, Y = simulate(T, default_params)
# X, W, A = particle_filter(N, T, Y, default_params)

# test = X[:, 0:10] * W[:, 0:10]

# print(Y[0:10])
# print(test.mean(axis=0))


