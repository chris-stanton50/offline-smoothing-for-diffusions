import numpy as np
import time
import types

from particles.core import SMC
import particles.resampling as rs

from sdes.feynman_kac import CDSSM_FeynmanKac, CDSSM_SMC

# method for a ParticleHistory object: generates samples by tracing back the ancestral path.
# We bind this to the ParticleHistory class, so that we can call it as a method.

def backward_sampling_geneaology(self, M):
    """
    Extract M full trajectories from the particle history.

    M final states are chosen randomly, then the corresponding trajectory
    is constructed backwards, until time t=0.
    """
    idx = self._init_backward_sampling(M)
    for t in reversed(range(self.T - 1)):
        idx[t, :] = self.A[t + 1][idx[t + 1, :]]
    return self._output_backward_sampling(idx)

method_dict = {"geneaology": 'backward_sampling_geneaology',
               "FFBS_ON2": 'backward_sampling_ON2',
               "FFBS_purereject": 'backward_sampling_reject',
               "FFBS_hybrid": 'backward_sampling_reject',
               "FFBS_MCMC": 'backward_sampling_mcmc'
               }

def modif_smoothing_worker(
    method=None, N=100, fk=None, num=10, smc_cls=CDSSM_SMC, add_funcs=None):
    """Modified version of 'smoothing_worker' from particles.smoothing.
    Removed two-filter smoothing, enabled evaluation of multiple additive functions.

    This worker may be used in conjunction with utils.multiplexer in order to
    run in parallel off-line smoothing algorithms.

    Parameters
    ----------
    method : string
         ['geneaology', 'FFBS_purereject', 'FFBS_hybrid', FFBS_MCMC', 'FFBS_ON2']
    N : int
        number of particles
    fk : Feynman-Kac object
        The Feynman-Kac model for the forward filter
    num : int 
        Number of imputed points to use if fk is an instance of CDSSM_FeynmanKac
        and running CDSSM_SMC.
    smc_cls: The smc class to use: set to either CDSSM_SMC or SMC
    add_funcs : function, with signature (t, x, xf)
        list of additive functions, at time t, for particles x=x_t and xf=x_{t+1}


    Returns
    -------
    a dict with fields:
    
    * est: a ndarray of length T
    * cpu_time

    Notes
    -----
    'FFBS_hybrid' is the hybrid method that makes at most N attempts to
    generate an ancestor using rejection, and then switches back to the
    standard (expensive method). On the other hand, 'FFBS_purereject' is the
    original rejection-based FFBS method, where only rejection is used. See Dau
    & Chopin (2022) for a discussion.
    """
    T = fk.T
    fk_string = fk.__class__.__name__ if not isinstance(fk, CDSSM_FeynmanKac) else fk.sname
    dimX = fk.cdssm.dimX if isinstance(fk, CDSSM_FeynmanKac) else fk.ssm.cdssm.dimX
    ests = {add_func_name: np.zeros((T, dimX)) for add_func_name in add_funcs.keys()}
    if smc_cls is CDSSM_SMC:
        pf = CDSSM_SMC(fk=fk, N=N, num=num, store_history=True)
    else:
        pf = SMC(fk=fk, N=N, store_history=True)
    print(f'Running fk model: {fk_string}')
    tic = time.perf_counter()
    pf.run()
    # Bind the backward sampling geneaology method to the ParticleHistory object
    pf.hist.backward_sampling_geneaology = types.MethodType(backward_sampling_geneaology, pf.hist)
    if method in method_dict.keys():
        bound_smoothing_method = getattr(pf.hist, method_dict[method])
        if method == "FFBS_purereject":
            z = bound_smoothing_method(N, max_trials=N * 10 ** 9)
        else:
            z = bound_smoothing_method(N)
        # Once we have the backward samples, we can post-process them however we want!
        # Don't feel restricted here!
        for add_func_name, add_func in add_funcs.items(): 
            ests[add_func_name][0] = np.mean(add_func(0, None, z[0]), axis=0) # add_func(t, x, xf) ->  (M, dimX)/ (M, ) -> (dimX, )/scalar            
            for t in range(1, T):
                ests[add_func_name][t] = np.mean(add_func(t, z[t-1], z[t]), axis=0) # add_func(t, x, xf) ->  (M, dimX)/ (M, ) -> (dimX, )/scalar
    else:
        print("smoothing_worker: no such method?")
    cpu_time = time.perf_counter() - tic
    print(method + " took %.2f s for N=%i" % (cpu_time, N))
    return {"ests": ests, "cpu": cpu_time}