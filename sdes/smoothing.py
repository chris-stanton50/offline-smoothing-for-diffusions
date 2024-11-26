import numpy as np
import time

from particles.core import SMC

from sdes.feynman_kac import CDSSM_FeynmanKac, CDSSM_SMC

def modif_smoothing_worker(
    method=None, N=100, fk=None, num=10, smc_cls=CDSSM_SMC, add_funcs=None):
    """Modified version of 'smoothing_worker' from particles.smoothing.
    Removed two-filter smoothing, enabled evaluation of multiple additive functions.

    This worker may be used in conjunction with utils.multiplexer in order to
    run in parallel off-line smoothing algorithms.

    Parameters
    ----------
    method : string
         ['FFBS_purereject', 'FFBS_hybrid', FFBS_MCMC', 'FFBS_ON2']
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
    fk_string = fk.__class__.__name__
    dimX = fk.cdssm.dimX if isinstance(fk, CDSSM_FeynmanKac) else fk.ssm.cdssm.dimX
    for attr_name in ["auxiliary_bridge_cls", "end_pt_proposal_sde_cls", "proposal_sde_cls"]:
        if hasattr(fk, attr_name):
            fk_string += "_" + getattr(fk, attr_name).__name__
    ests = {add_func_name: np.zeros((T, dimX)) for add_func_name in add_funcs.keys()}
    if smc_cls is CDSSM_SMC:
        pf = CDSSM_SMC(fk=fk, N=N, num=num, store_history=True)
    else:
        pf = SMC(fk=fk, N=N, store_history=True)
    print(f'Running fk model: {fk_string}')
    tic = time.perf_counter()
    pf.run()
    if method.startswith("FFBS"):
        submethod = method.split("_")[-1]
        if submethod == "ON2":
            z = pf.hist.backward_sampling_ON2(N)
        elif submethod == "MCMC":
            z = pf.hist.backward_sampling_mcmc(N)
        elif submethod == "hybrid":
            z = pf.hist.backward_sampling_reject(N)
        elif submethod == "purereject":
            z = pf.hist.backward_sampling_reject(N, max_trials=N * 10 ** 9)
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