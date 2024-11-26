"""
We use 3 different types of plot to evaluate the performance of the multiple run particle filter/smoother methods:

1. Boxplots: for moments (across dimension/time, DT plots), log-likelihoods (across time, T plots), and CPU times
2. Mean Squared Error (MSE) plots (across dimension, d plots)
3. Effective Sample Size (ESS) plots 
"""

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from particles import SMC


# Boxplots
#--------------------------------------------------------------------------

# Summary functions for the boxplots:

# We may be interested in boxplots of:

# Moments; for filtering/smoothing
# Log-likelihoods for filtering
# CPU times for filtering/smoothing

class FilterFunc(object):
    """
    Wrapper for function that takes a 
    SMC/CDSSM_SMC object as input and returns a
    summary statistic (e.g CPU time, log-likelihood, first moment)
    """
    def __init__(self, func, t=None, d=None):
        self.func = func
        self.t = t
        self.d = d
    
    def __call__(self, r):
        if self.t is None and self.d is None:
            return self.func(r)
        if self.t is not None and self.d is None:
            return self.func(r, self.t)
        if self.t is not None and self.d is not None:
            return self.func(r, self.t, self.d)
    
    @property
    def name(self):
        out = self.func.__name__
        out += '' if self.t is None else f'_t={self.t+1}'
        out += '' if self.d is None else f'_d={self.d}'        
        return out


class SmootherFunc(object):
    """
    Wrapper for function that takes a 
    SMC/CDSSM_SMC object as input and returns a
    summary statistic (e.g CPU time, log-likelihood, first moment)
    """
    def __init__(self, func, t=None, d=None):
        self.func = func
        self.t = t
        self.d = d
    
    def __call__(self, r):
        if self.t is None and self.d is None:
            return self.func(r)
        if self.t is not None and self.d is None:
            return self.func(r, self.t)
        if self.t is not None and self.d is not None:
            return self.func(r, self.t, self.d)
    
    @property
    def name(self):
        out = self.func.__name__
        out += '' if self.t is None else f'_t={self.t+1}'
        out += '' if self.d is None else f'_d={self.d}'
        return out

def filt_mean(r, t, d):
    return r['output'].summaries.moments[t]['mean'][d]

def filt_var(r, t, d):
    return r['output'].summaries.moments[t]['var'][d]

def filt_std(r, t, d):
    return r['output'].summaries.moments[t]['std'][d]

def smth_mean(r, t, d):
    return r['ests']['1_mom'][t, d]

def smth_var(r, t, d):
    return r['ests']['2_mom'][t, d] - r['ests']['1_mom'][t, d]**2

def smth_std(r, t, d):
    return np.sqrt(smth_var(r, t, d))

def gen_mom_funcs(mom_func, times, dimX):
    """
    We may be interested in moments across different dimensions:
    """
    wrapper = FilterFunc if mom_func.__name__.startswith('filt') else SmootherFunc
    mom_funcs = [[wrapper(mom_func, t, d) for t in times] for d in range(dimX)] # (times, dimX)
    return mom_funcs

def logLt(r, t):
    return r['output'].summaries.logLts[t]

def gen_logLt_funcs(times):
    """
    We may be interested in log-likelihoods across different dimensions:
    """
    logLt_funcs = [FilterFunc(logLt, t, d=None) for t in times] # (times, dimX)
    return logLt_funcs

def filt_cpu_time(r):
    return r['output'].cpu_time

filt_cpu_time = FilterFunc(filt_cpu_time, t=None, d=None)

def smth_cpu_time(r):
    return r['cpu']

smth_cpu_time = SmootherFunc(smth_cpu_time, t=None, d=None)

def boxplot(results, out_func, fk_names, ax, true_value=None):
    """
    A single boxplot of different FK names across a single axis.
    """
    sb.boxplot(x=[r['fk'] for r in results if r['fk'] in fk_names], y=[out_func(r) for r in results if r['fk'] in fk_names], ax=ax)
    ax.set_ylabel(out_func.name)
    if true_value is not None:
        ax.axhline(y=true_value, ls=':', color='k')
    return plt.gcf(), ax

def boxplots(results, out_funcs, fk_names, true_values=None):
    # example_out = results[0]['output']
    # smoothing = False if isinstance(example_out, SMC) else True
    # cdssm_res = next(r for r in results if isinstance(r['output'], CDSSM))
    # results = results_dict['results']; n_runs = results_dict['n_runs']; num = results_dict['num']
    l= len(out_funcs)
    fig, axes = plt.subplots(l, 1, figsize=(10, 10*l), sharex=True)
    true_values = [None]*len(out_funcs) if true_values is None else true_values
    for i, out_func in enumerate(out_funcs):
        axis = axes[i] if l > 1 else axes
        _, ax = boxplot(results, out_func, fk_names, axis, true_value=true_values[i])
    return fig, axes

# MSE Plots
#--------------------------------------------------------------------------


def get_phi_x_x(r): 
    return np.stack([mom['var'] + mom['mean'] ** 2 for mom in r.moments], axis=1) # (dimX, T)

def get_logLt(r):
    return np.array(r.logLts) # (T, )

def get_phi_x(r):
    return r['ests']['phi_x'].T # (dimX, T)

def get_phi_x_x(r):
    return r['ests']['phi_x_x'].T # (dimX, T)

def mse_plot(results, fk_names, true_vals, out_func):
    """
    Generate a plot of the Monte Carlo MSEs of the estimators of the first moment of each of the 
    marginal filtering/smoothing distributions.
    
    Used for presentation of results after multi-SMC runs.
    
    Parameters
    ----------
    results: list of dictionaries
        The results of the Monte Carlo simulations.
    fk_names: list of strings giving the names of the fk_models to plot
    true_vals: numpy array of shape (dimX, T+1)
        The true values of the estimated quantity.
        Generated from either Kalman filter/RTS Smoother or a particle filter/smoother with large N.
    smoothing: bool
    
    Returns
    -------
    fig, axes: matplotlib figure and axes objects
    """
    dimX = 1 if len(true_vals.shape) == 1 else true_vals.shape[0]
    T = true_vals.shape[0] if dimX == 1 else true_vals.shape[1]
    true_vals = true_vals.reshape((1, -1)) if dimX == 1 else true_vals # (dimX, T)
    fig, axes = plt.subplots(1, dimX, figsize=(10*dimX, 5)) 
    models = fk_names; times = np.arange(T) + 1
    for mod in models:
        errors = np.stack([out_func(r)-true_vals for r in results if r['fk']==mod], axis=0) # (n_runs, dimX, T)
        mses = np.sqrt(np.mean(errors**2, axis=0))
        if dimX > 1:
            for i, ax in enumerate(axes):
                ax.plot(times, mses[i], label=mod, linewidth=2)
                ax.set_xlabel(r'$t$')
        else:
            axes.plot(times, mses[0], label=mod, linewidth=2)
            axes.set_xlabel(r'$t$')
    plt.legend()
    return fig, axes

# ESS Plots
#--------------------------------------------------------------------------

def ess_plot(results, fk_names):
    plt.figure()
    for model in fk_names:        
        summary = next(r['output'] for r in results if r['fk']==model)
        N = next(r['N'] for r in results if r['fk']==model)
        T = len(summary.ESSs)
        plt.plot(np.arange(1, T+1), summary.ESSs,label=model, linewidth=2)
    plt.axis([1,T,0,N])
    plt.legend()
    plt.xlabel(r'$t$')
    plt.ylabel('ESS')
    return plt.gcf(), plt.gca()

# Update rates
def update_rate(x):
    """Update rate.

    Parameters
    ----------
    x: (N,T) or (N,T,d) array

    Returns
    -------
    a (T,) or (T,d) array containing the frequency at which each
    component was updated (along axis 0)
    """
    return np.mean(x[1:] != x[:-1], axis=0)