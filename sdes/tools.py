"""
For now, use this module as a miscellaneous dump for various useful functions.
These functions are predominantly used in notebooks to speed up the presentation
of some results.
"""

import numpy as np
import numpy.linalg as nla
import matplotlib.pyplot as plt
import seaborn as sb
import inspect
import arviz as az
import time

def log_abs_det(A):
    """
    Compute the log of the absolute determinant of a matrix A.
    """
    return np.log(np.abs(nla.det(A)))

def get_methods(instance):
    return [name for name, member in inspect.getmembers(instance, predicate=inspect.ismethod)]

def get_attrs(instance):
    return [name for name, member in inspect.getmembers(instance, predicate=inspect.isdatadescriptor)]

def get_properties(instance):
    return [name for name, attr in inspect.getmembers(type(instance))
                  if isinstance(attr, property)]

def match_first_dim(func):
    def new_func(*args):
        new_args = list(args)
        arr_args = [arg for arg in args if type(arg) is np.ndarray]
        arr_args_idx = [i for i, arg in enumerate(args) if type(arg) is np.ndarray]
        N = max([arr_arg.shape[0] for arr_arg in arr_args])
        for i, arg in enumerate(arr_args):
            idx = arr_args_idx[i]
            if arg.shape[0] != N and arg.shape[0] == 1:
                arg = np.concatenate([arg]*N)
            elif arg.shape[0] != N and arg.shape[0] != 1:
                arg = np.stack([arg]*N)
            else:
                pass
            new_args[idx] = arg
        return func(*new_args)
    return new_func

def method_match_first_dim(method):
    """
    Note: The logic of this decorator will break down if the number of particles
    is set to the dimension of either x or y. 
    """
    def new_func(self, *args):
        new_args = list(args)
        arr_args = [arg for arg in args if type(arg) is np.ndarray]
        arr_args_idx = [i for i, arg in enumerate(args) if type(arg) is np.ndarray]
        N = max([arr_arg.shape[0] for arr_arg in arr_args])
        for i, arg in enumerate(arr_args):
            idx = arr_args_idx[i]
            if arg.shape[0] != N and arg.shape[0] == 1 and arg.ndim == 2:
                arg = np.concatenate([arg]*N)
            elif arg.shape[0] != N and arg.ndim > 2:
                arg = np.stack([arg]*N)
            else:
                pass
            new_args[idx] = arg
        return method(self, *args)
    return new_func

# @match_first_dim
def mv_grad_log_linear_gaussian(x_s: np.ndarray, x_t: np.ndarray, A: np.ndarray, b: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    The gradient of the log of a linear Gaussian transition density w.r.t x_s
    X_t | X_s = x_s \sim \mathcal{N}(A x_s + b, S)

    $$ \nabla_{x_s} \log(p_{s, t}(x_t|x_s)) = [A^T S^{-1} (x_t - b) - A^T S^{-1} A x_s]$$
    
    Standard dimensions of the inputs:

    x_s (N, dimX)
    x_t (N, dimY)
    A (N, dimY, dimX)
    b (N, dimY)
    S (N, dimY, dimY)
    
    Dimension of output: 
    (N, dimX)
    """
    dimX = x_s.shape[1]; dimY = x_t.shape[1]
    N = max(x_s.shape[0], x_t.shape[0])
    A_trans = np.einsum('ijk->ikj', A)
    A_x_s = np.einsum('ijk,ik->ij', A, x_s)
    grad_log_lg = np.einsum('ijk,ik->ij', A_trans, nla.solve(S, x_t - b)) - np.einsum('ijk,ik->ij', A_trans, nla.solve(S, A_x_s))
    return grad_log_lg

def mv_grad_grad_log_linear_gaussian(A: np.ndarray, b: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    The hessian of the log of a linear Gaussian transition density w.r.t x_s 
    X_t | X_s = x_s \sim \mathcal{N}(A x_s + b, S)

    $$ \nabla_{x_s} \nabla_{x_s}^T \log(p_{s, t}(x_t|x_s)) = -A^T S^{-1} A $$
    
    Standard dimensions of the inputs:

    A (N, dimY, dimX)
    b (N, dimY)
    S (N, dimY, dimY)
    
    Dimension of output: 

    (N, dimX)
    """    
    return -1.*np.einsum('ijk,ijl->ikl', A, nla.solve(S, A)) # (N, dimX, dimX)
 
def grad_log_linear_gaussian(x_s: np.ndarray, x_t: np.ndarray, a, b, s):
    """
    The gradient of the log of a linear Gaussian transition density 
    X_t | X_s = x_s \sim \mathcal{N}(a x_s + b, s)

    Standard dimensions of the inputs:

    x_s (N, )
    x_t (N, )
    a float
    b float
    s float
    """
    return (a* (x_t - a*x_s - b))/s

def vec_grad_log_linear_gaussian(x_s: np.ndarray, x_t: np.ndarray, a, b, s): 
    """
    Vectorised implementation of the gradient of the log transition density to use
    when the gradient of the log transition density of a linear SDE is needed for the 
    evaluation of path integrals. For this implementation, the inputs to the function will be of
    the following dimension:

    x_s (N, num+1)
    x_t (N, )
    a (N, num+1)
    b (N, num+1)
    s (N, num+1)

    The matrix transpose .T is applies to x_s during the calculations, to ensure that the broadcasting
    is done correctly.
    """
    return ((x_t - (a*x_s).T).T - b)/s

def grad_grad_log_linear_gaussian(a, b, s):
    """
    The second derivative of the log of a linear Gaussian transition density 
    X_t | X_s = x_s \sim \mathcal{N}(a x_s + b, s)
    a: float
    s: float
    """
    return (-1.*(a * a))/s

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

def vectorise_param(param, num_plus_1):
    if type(param) is float:
        return param
    else:
        return np.stack([param]*num_plus_1).T

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

def mv_state_container_size(X: np.ndarray, x_start):
    X_shape_idx = 1 if X.shape == () else X.shape[1]
    x_start_shape_idx = 1 if x_start.shape == () else x_start.shape[1]
    size = max(X_shape_idx, x_start_shape_idx)
    return size

def use_end_point(phi):
    def phi_dec(t, x, xf):
        if x is not None:
            x = x if x.dtype in [np.float32, np.float64] else x[x.dtype.names[-1]]
        xf = xf if xf.dtype in [np.float32, np.float64] else xf[xf.dtype.names[-1]]
        out = phi(t, x, xf)
        return out
    return phi_dec

def init_kwargs_dict(cls, locals):
    signature = inspect.signature(cls.__init__)
    params = signature.parameters
    kwarg_names = [name for name, param in params.items() if param.default != inspect.Parameter.empty]
    init_kwargs = {k: locals[k] for k in locals if k in kwarg_names}
    return init_kwargs

def struct_arr_to_arr(struct_arr):
    """
    Input: struct_arr: numpy structured array of shape (M, )
    Returns: arr: numpy array of shape (M, num)
    """
    names = struct_arr.dtype.names # num names
    arr = np.stack([struct_arr[name] for name in names], axis=1)
    # List of length (num, ) of numpy arrays of shape (M, )
    return arr

def struct_arrs_to_arr(struct_arrs):
    """
    Input: struct_arrs: list of length T of numpy structured arrays of shape (M, )
            each structured array has the same dtype and consists of num fields: the imputed points.
    Returns: arr: numpy array of shape (M, num*T): representing the full paths.
    """
    arrs = [struct_arr_to_arr(struct_arr) for struct_arr in struct_arrs]
    arr = np.concatenate(arrs, axis=1)
    return arr

# Generate a random symmetric positive definite matrix
def generate_spd_matrix(d):
    while True:
        # Generate a random matrix
        A = np.random.rand(d, d)
        
        # Make it symmetric
        spd_matrix = np.dot(A, A.T)
        
        # Add d * I to shift eigenvalues up, which often helps but isn't strictly necessary
        spd_matrix += d * np.eye(d)
        
        # Check if the matrix is positive definite by confirming all eigenvalues are positive
        eigenvalues = np.linalg.eigvalsh(spd_matrix)
        if np.all(eigenvalues > 0):
            break
    
    return spd_matrix

def mcmc_to_inferencedata(mcmc):
    """
    Converts a particles MCMC object into an ArviZ InferenceData object, including posterior samples,
    prior samples, acceptance rate (stored as an attribute), latent variable x (if available), 
    observed data, and CPU time (stored as an attribute of the posterior).
    
    Observations are stored in the `observed_data` group, accounting for different data types:
    - Univariate observations: shape (1,) stored with a "time" coordinate.
    - Unnamed multivariate observations: shape (1, dimy) stored with "time" and "dimy" coordinates.
    - Named multivariate observations: structured arrays, stored with "time" and "parameter" coordinates.
    
    Parameters:
    - mcmc: An MCMC object from the particles package, which has a 'chain' attribute containing
            the output of the Markov chain and a 'prior' attribute representing the prior distribution.
    
    Returns:
    - idata: An ArviZ InferenceData object with the posterior, prior samples, latent variables (x),
             observed data, and sample statistics. The cpu_time and acceptance rate are stored as 
             attributes of the posterior.
    """

    # Extract the MCMC chain output (theta)
    theta = mcmc.chain.theta  # This is a structured numpy array with dimension (niter,)
    niter = mcmc.niter
    T = len(mcmc.data)
 
    # Get the parameter names from the structured array
    param_names = theta.dtype.names
    
    # Initialize dictionaries to store posterior and prior samples
    posterior_samples = {}
    prior_samples = {}

    # Generate prior samples using the prior distribution's 'rvs' method
    prior = mcmc.prior.rvs(size=niter)  # This is a structured array with the same dtype as theta
    
    # Collect posterior samples for parameters (theta)
    for param in param_names:
        posterior_samples[param] = theta[param].reshape((1, niter))  # Reshape to (1, niter) for chain and draw
        prior_samples[param] = prior[param]
    
    # Initialize posterior coordinates
    coords = {
        "chain": [0],               # Single chain assumed
        "draw": np.arange(niter)    # Iterations of the MCMC
    }

    # Add dims for posterior parameters (theta)
    dims = {param: ["chain", "draw"] for param in param_names}

    # Check if the MCMC chain has the latent variable 'x'
    if hasattr(mcmc.chain, 'x'):
        x = mcmc.chain.x  # Extract the latent variable x
        
        # If 'x' is a structured array (has .dtype.names), handle it with an extra parameter dimension
        if x.dtype.names:
            param_names_x = x.dtype.names  # Get the parameter names for 'x'
            T = x.shape[1]                 # Number of timesteps (second dimension of x)

            x_extend = np.stack([x[param] for param in param_names_x], axis=-1)
            posterior_samples['x'] = x_extend.reshape((1, niter, T, len(param_names_x)))
            coords["time"] = np.arange(T)
            coords["continuous_time"] = param_names_x
            
            # Add dims for structured x parameters        
            dims["x"] = ["chain", "draw", "time", "continuous_time"] 

        else:
            # If 'x' is a regular numpy array, treat it as (niter, T)
            T = x.shape[1]  # Number of timesteps
            posterior_samples['x'] = x.reshape((1, niter, T))  # Add to posterior samples
            
            # Add time as a coordinate
            coords["time"] = np.arange(T)
            dims["x"] = ["chain", "draw", "time"]  # Add dims for the non-structured latent variable x

    # Handle observed data in the 'data' attribute
    observed_data_samples = {}

    if mcmc.data[0].shape == (1,): # Univariate case
        observed_data_samples['obs'] = np.concatenate(mcmc.data)
    else:
        raise NotImplementedError("Multivariate/structured observations not yet supported.")

    # Set dimensions for observed data
    dims['obs'] = ['time']

    # Create the InferenceData object using arviz.from_dict
    idata = az.from_dict(
        posterior=posterior_samples,
        prior=prior_samples if prior_samples else None,
        observed_data=observed_data_samples if observed_data_samples else None,
        coords=coords,              # Coordinate system for the posterior
        dims={**dims}  # Combine dims for posterior and observed data
    )

    # Store cpu_time as an attribute of the posterior if it exists
    if hasattr(mcmc, 'cpu_time'):
        idata.posterior.attrs["cpu_time"] = mcmc.cpu_time  # Add cpu_time as an attribute to the posterior group

    # Store acceptance rate as an attribute of the posterior if it exists
    if hasattr(mcmc, 'acc_rate'):
        idata.posterior.attrs["acc_rate"] = mcmc.acc_rate  # Add acceptance rate as an attribute

    
    attrs_lst = ["prior", "scale", "L", "adaptive", "ssm_cls", "smc_cls", "smc_options", "cdssm_cls", "theta0", "N_steps", "Nx", "fk_cls", "regenerate_data", "backward_step", "num", "cdssm_options"]
    for attr in attrs_lst:
        if hasattr(mcmc, attr):
            idata.attrs[attr] = getattr(mcmc, attr)
    if hasattr(mcmc, 'mwgibbs'):
        mwgibbs_attrs = ["scale", "L", "adaptive"]
        for attr in mwgibbs_attrs:
            idata.attrs[f'mwgibbs_{attr}'] = getattr(mcmc.mwgibbs, attr)
    
    idata.attrs['name'] = mcmc.__class__.__name__
    idata.attrs['method_cls'] = mcmc.__class__
    idata.attrs['inference_library'] = 'particles'
    idata.attrs['inference_library_version'] = '0.3alpha'

    return idata

def timed_func(func):
    def timed(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start
    return timed