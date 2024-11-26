"""
We use this module for functions that evaluate path integrals that are used as weights within particle filters in continuous time.
"""
import numpy as np
import numpy.linalg as nla

def vec_eval(func, times, X):
    func_vals = np.stack([func(time, X[i]) for i, time in enumerate(times)], axis=0) # (num+1, N, dimX)
    return func_vals
    
#------------------ Univariate Path integrals ------------------#

def log_girsanov(X: np.ndarray, b_1, b_2, Cov, step) -> np.ndarray:
    """
    Function to evaluate the weights of sample paths according to Girsanov's formula.
    Two SDEs must have common drift and diffusion coefficients.
    
    Need to TEST following the change to see if it works.
    
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
    N, num_plus_1 = X.shape
    times = np.linspace(0., step*(num_plus_1-1), num=num_plus_1) # (num+1, )
    b_1s = b_1(times, X); b_2s = b_2(times, X) # (N, num+1)
    dX_integ = (b_1s - b_2s)/Cov(times, X) # (N, num+1)
    dt_integ = dX_integ * (b_1s + b_2s) # (N, num+1)
    dXs = X[:, 1:] - X[: , :-1]
    dts = times[1:] - times[:-1]
    dX_integral_vals = dX_integ[:, :-1] * dXs #(N, num), (N, num)
    dX_integral_vals = dX_integral_vals.sum(axis=1)
    dt_integral_vals = dt_integ[:, :-1] * dts # (N, num), (N,num)
    dt_integral_vals = dt_integral_vals.sum(axis=1)
    log_wgts = dX_integral_vals - 0.5 * dt_integral_vals
    return log_wgts

def log_delyon_hu(X: np.ndarray, b, Cov, step):
    """
    Function to calculate the weight emerging from stochastic integrals
    from the Delyon-Hu bridge proposal.
    
    Have attempted to extend to the third stochastic integral.
    Have used the formulation as in Papaspiliopoulos et al. 2012,
    where the integrator is taken to be 1/Cov. When we have a 
    constant diffusion coefficient, this stochastic integral will
    cancel out.
    
    
    Need to TEST this to see if it works.
    
    Inputs
    ------------
    X: np.ndarray (N, num+1)
    b: (N, num+1) -> (N, )
    Cov: (N, num+1) -> (N, )
    step: float
    
    Returns
    ------------
    log_wgts: np.ndarray (N, )
    """
    N, num_plus_1 = X.shape; Delta_s = step*(num_plus_1 - 1); x_end = X[:, -1].reshape(-1, 1) #
    times = np.linspace(0., Delta_s, num=num_plus_1)
    b_s = b(times, X); covs = Cov(times, X) # (N, num+1), (N, num+1)
    if type(covs) == float:
        covs = covs*np.ones(X.shape)
    if covs.shape == (N, ):
        covs = covs.reshape(-1, 1)
    dX_integ =  b_s/covs # (N, num+1) # (num+1, ), (N, num+1)
    dt_integ = dX_integ*b_s # (N, num+1)
    dZ_integ = np.square((x_end - X[:, 1:-1]))/(Delta_s - times[1:-1]) # (N, num-1)
    Z = 1./covs[:, :-1] # (N, num) # Taking as in Papspiliopoulos et al. 2012
    dXs = X[:, 1:] - X[: ,:-1] # (N, num)
    dts = times[1:] - times[:-1] # (num, )
    dZs = Z[:, 1:] - Z[:, :-1] # (N, num-1)
    dX_integral_vals = dX_integ[:, :-1] * dXs #(N, num), (N, num)
    dX_integral_vals = dX_integral_vals.sum(axis=1)
    dt_integral_vals = dt_integ[:, :-1] * dts # (N, num), (num, )
    dt_integral_vals = dt_integral_vals.sum(axis=1)
    dZ_integral_vals = dZ_integ * dZs # (N, num-1), (N, num-1)
    dZ_integral_vals = dZ_integral_vals.sum(axis=1) # (N, num-1) -> (N, )
    log_wgts = dX_integral_vals - 0.5 * dt_integral_vals - 0.5*dZ_integral_vals - 0.5*dZ_integral_vals
    return log_wgts

def log_drift_delyon_hu(X: np.ndarray, b, Cov, step):
    N, num_plus_1 = X.shape; Delta_s = step*(num_plus_1 - 1); x_end = X[:, -1].reshape(-1, 1) #
    times = np.linspace(0., Delta_s, num=num_plus_1)
    b_s = b(times, X); covs = Cov(times, X) # (N, num+1), (N, num+1)
    def dZ_integrand(t, x):
        numer = (x_end - x.T).T # x_end (N, ), x is (N, num+1) so use x.T so that broadcasting works.
        numer = numer[:, :-1] # (N, num) Remove end point to avoid zero division error
        denom = Delta_s - t # (num+1, )
        denom = denom[:-1] # (num, )
        return numer/denom # (N, num)
    def dt_integrand(t, x):
        t1 = dZ_integrand(t, x) # (N, num): last point removed to avoid zero division error
        t2 = b(t, x)/Cov(t, x) # (N, num+1)
        t2 = t2[:, :-1] # (N, num+1) -> (N, num): remove end points
        out = t1/t2
        return out 
    def Z_integrator(t, x): # Obtains the Z process from the X process: the integrator in the third stochastic integral.
        numer = (x_end - x.T).T # x_end (N, ), x is (N, num+1) so use x.T so that broadcasting works.
        denom = Cov(t, x)   
        out = numer/denom # (N, num+1)
        out = out[:, :-1] # (N, num) last part not needed as using RHS integration.
        return out
    dt_integrand_vals = dt_integrand(times, X) # (N, num+1)
    dZ_integrand_vals = dZ_integrand(times, X) # (N, num)
    Z = Z_integrator(times, X) # (N, num)
    dts = times[1:] - times[:-1]
    dZs = Z[:, 1:] - Z[:, :-1]
    dt_integral_vals = dt_integrand_vals[:, :-1] * dts # (N, num-1), (N,num-1)
    dt_integral_vals = dt_integral_vals.sum(axis=1)
    dZ_integral_vals = dZ_integrand_vals[:, 1:] * dZs # (N, num-1), (N, num-1)
    dZ_integral_vals = dZ_integral_vals.sum(axis=1)
    log_wgts = dt_integral_vals - 0.5 * dZ_integral_vals
    return log_wgts

def log_van_der_meulen_schauer(X: np.ndarray, b, Cov, LinearSDE, step):
    def b_tilde(t, x):
        return LinearSDE.b_vec(t, x)
    def Cov_tilde(t, x):
        return LinearSDE.Cov_vec(t, x)
    def r(t, x):
        return LinearSDE._vec_grad_log_px(t, Delta_s, x, x_end) # (N, num+1) 
    def H(t):
        return -1.*LinearSDE._vec_grad_grad_log_px(t, Delta_s) # (num+1, )
    def phi(t, x):
        return (b(t, x) - b_tilde(t, x))*r(t, x)-0.5*(Cov(t, x) - Cov_tilde(t, x))*(H(t) - (r(t, x) ** 2))
    num_plus_1 = X.shape[1]
    Delta_s = (num_plus_1 - 1) * step; x_end = X[:, -1]
    times = np.linspace(0., Delta_s, num=num_plus_1)
    dt_integrand_vals = phi(times, X)
    dts = times[1:] - times[:-1]
    dt_integral_vals = dt_integrand_vals[:, :-1] * dts # (N, num), (N,num)
    dt_integral_vals = dt_integral_vals.sum(axis=1) # (N, )
    log_wgts = dt_integral_vals
    return log_wgts

#------------------ Multivariate Path integrals ------------------#

def mv_log_girsanov(X: np.ndarray, b_1, b_2, Cov, step) -> np.ndarray:
    """
    Function to evaluate the weights of sample paths according to Girsanov's formula.
    Two SDEs must have common drift and diffusion coefficients.


    Inputs
    ------------
    X (np.ndarray):     A (num+1, N, dimX) array of N sample paths from SDE of dimension dimX 
                        imputed at num+1 times
    b_1:                Drift function of the target SDE. float, (N, dimX) -> (N, dimX)
    b_2:                Drift function of the simulated SDE float, (N, dimX) -> (N, dimX)
    Cov:                The common SDE covariance matrix of both SDEs: float, (N, dimX) -> (N, dimX, dimX)
    step (float):       The timestep between each imputation. So the end time is step * num

    Returns
    ------------
    log_wgts (np.ndarray): A (N, ) array of the log weight of each sample path.
    """
    num_plus_1, N, dimX = X.shape
    ts = np.linspace(0., step*(num_plus_1-1), num=num_plus_1)
    covs = vec_eval(Cov, ts, X); b_1s  = vec_eval(b_1, ts, X); b_2s = vec_eval(b_2, ts, X) # (num+1, N, dimX, dimX), (num+1, N, dimX) (num+1, N, dimX)
    dX_int_vals = nla.solve(covs, b_1s - b_2s) # (num+1, N, dimX, dimX), (num+1, N, dimX) -> (num+1, N, dimX)
    dt_int_vals = np.einsum('ijk,ijk->ji', b_1s + b_2s, dX_int_vals) # (N, num+1)
    dXs = X[1:] - X[:-1] # (num, N, dimX)
    dts = ts[1:] - ts[:-1] # (num, )
    dX_integral_vals = np.einsum('ijk,ijk->ji', dX_int_vals[:-1], dXs) # (N, num)
    dX_integral_vals = dX_integral_vals.sum(axis=1) # (N, )
    dt_integral_vals = dt_int_vals[:, :-1] * dts # (N, num)
    dt_integral_vals = dt_integral_vals.sum(axis=1) # (N, )
    log_wgts = dX_integral_vals - 0.5 * dt_integral_vals # (N, )
    return log_wgts

def mv_log_delyon_hu(X: np.ndarray, b, Cov, step):
    """
    For now, we have implemented a version that assumes that the diffusion 
    coefficient is constant.
    In this case, the third stochastic integral cancels, and we are left with 
    an expression that looks like the regular Girsanov.

    Inputs
    ------------
    X: (num+1, N, dimX)
    b: (N, dimX) -> (N, dimX)
    Cov: (N, dimX) -> (N, dimX, dimX)
    step: float
    
    Returns
    ------------
    log_wgts: (N, )
    """
    num_plus_1, N, dimX = X.shape
    x_end = X[:, -1]; Delta_s = step*(num_plus_1 - 1) # Only needed for the third stochastic integral.
    ts = np.linspace(0., step*(num_plus_1-1), num=num_plus_1)
    covs, b_s = (vec_eval(Cov, ts, X), vec_eval(b, ts, X))
    dX_int_vals = nla.solve(covs, b_s) # (num+1, N, dimX, dimX), (num+1, N, dimX) -> (num+1, N, dimX)
    dt_int_vals = np.einsum('ijk,ijk->ji', b_s, dX_int_vals) # (N, num+1)
    dXs, dts = (X[1:] - X[:-1], ts[1:] - ts[:-1]) # (num, N, dimX) # (num, )
    dX_integral_vals = np.einsum('ijk,ijk->ji', dX_int_vals[:-1], dXs) # (N, num)
    dX_integral_vals = dX_integral_vals.sum(axis=1) # (N, )
    dt_integral_vals = dt_int_vals[:, :-1] * dts # (N, num)
    dt_integral_vals = dt_integral_vals.sum(axis=1) # (N, )
    log_wgts = dX_integral_vals - 0.5 * dt_integral_vals # (N, )
    return log_wgts    

def mv_log_van_der_meulen_schauer(X: np.ndarray, b, Cov, LinearSDE, step):
    num_plus_1, N, dimX = X.shape
    x_end = X[-1]; Delta_s = step*(num_plus_1 - 1)
    ts = np.linspace(0., step*(num_plus_1-1), num=num_plus_1) # (num, )
    b_tilde = LinearSDE.b; Cov_tilde = LinearSDE.Cov
    def r(t, x):
        return LinearSDE.grad_log_px(t, Delta_s, x, x_end)
    def H(t, x):    
        return LinearSDE.grad_grad_log_px(t, Delta_s)
    funcs = [b, b_tilde, Cov, Cov_tilde, r, H]
    b_s, b_tildes, covs, cov_tildes, r_s, H = [vec_eval(f, ts[:-1], X) for f in funcs]
    first_term = np.einsum('ijk,ijk->ij', b_s - b_tildes, r_s) # (num, N)
    r_r_T = np.einsum('ijk,ijl->ijkl', r_s, r_s) # (num, N, dimX, dimX)
    second_term = np.einsum('ijkl,ijlm->ijkm', covs - cov_tildes, -H - r_r_T) # (num, N, dimX, dimX)
    second_term = np.einsum('ijkk->ij', second_term) # Trace of each matrix: (num, N)
    phi = first_term - 0.5*second_term # (num, N)
    dts = ts[1:] - ts[:-1] # (num, )
    dt_integral_vals = phi.T * dts # (N, num)
    dt_integral_vals = dt_integral_vals.sum(axis=1) # (N, )
    log_wgts = dt_integral_vals
    return log_wgts
    
    # def dZ_integrand(t, x):
    #     numer = (x_end - x.T).T # x_end (N, ), x is (N, num+1) so use x.T so that broadcasting works.
    #     numer = numer[:, :-1] # (N, num) Remove end point to avoid zero division error
    #     numer = -np.square(numer)
    #     denom = Delta_s - t # (num+1, )
    #     denom = denom[:-1] # (num, )
    #     return numer/denom # (N, num)
    # def Z_integrator(t, x): # Obtains the Z process from the X process: the integrator in the third stochastic integral.
    #     numer = (x_end - x.T).T # x_end (N, ), x is (N, num+1) so use x.T so that broadcasting works.
    #     denom = Cov(t, x)   
    #     out = numer/denom # (N, num+1)
    #     out = out[:, :-1] # (N, num) last part not needed as using RHS integration.
    #     return out

    # dZ_integrand_vals = dZ_integrand(times, X) # (N, num)
    # Z = Z_integrator(times, X) # (N, num)
    # dZs = Z[:, 1:] - Z[:, :-1]
    # dZ_integral_vals = dZ_integrand_vals[:, 1:] * dZs # (N, num-1), (N, num-1)
    # dZ_integral_vals = dZ_integral_vals.sum(axis=1)
    #log_wgts = dX_integral_vals - 0.5 * dt_integral_vals - 0.5 * dZ_integral_vals
