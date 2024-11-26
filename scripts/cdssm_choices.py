import numpy as np
import numpy.linalg as nla

from sdes.sdes import FitzHughNagumo, MvOrnsteinUhlenbeck, TS_MvOrnsteinUhlenbeck, OrnsteinUhlenbeck
from sdes.continuous_discrete_ssms import GaussianCDSSM, TimeSwitchingGaussianCDSSM
from sdes.tools import timed_func
import particles.state_space_models as ssms
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



# -----------------------------------------------------------------------------------------
# ------------------ TS_MvOrnsteinUhlenbeck + TimeSwitchingGaussianCDSSM ------------------

# Choose a model SDE:
sde_cls = TS_MvOrnsteinUhlenbeck

# SDE parameters
dimX = 2; t_switch = 2; 
# phi_1 = np.linalg.cholesky(generate_spd_matrix(dimX))
phi_1 = nla.cholesky(np.array([[1., 0.9,], [0.9, 1.]]))
phi_2 = 0.01 * phi_1 # Same correlation structure, but smaller variance
sde_params = {'dimX': dimX,
            't_switch': t_switch,
        'rho_1': 0.01*np.ones((1, dimX)),
        'mu_1': np.zeros((1, dimX)),
        'phi_1': phi_1,
        'rho_2': 0.01*np.ones((1, dimX)),
        'mu_2': np.zeros((1, dimX)),
        'phi_2': phi_2
        }

# Choose a CDSSM:
# cdssm_cls = GaussianCDSSM
cdssm_cls = TimeSwitchingGaussianCDSSM


# We can now define a CDSSM: we observe the latent SDE process at discrete points in time with additive noise:
# CDSSM parameters
x0 = np.array([0.]*dimX).reshape(1, -1) if dimX > 1 else 0.; delta_s = 1.; 
dimY = dimX; eta_sq = 0.4 ** 2; t_switchY = 10

L = np.eye(dimY) if (dimY > 1 or dimX > 1) else 1. 
CovY = np.eye(dimY)*eta_sq if dimY > 1 else eta_sq

if cdssm_cls == TimeSwitchingGaussianCDSSM:
    obs_params = {'L_1': L, 'covY_1': CovY, 'L_2': L, 'covY_2': 0.001*CovY, 't_switchY': t_switchY, 'x0': x0, 'delta_s': delta_s}
else:
    obs_params = {'L': L, 'covY': CovY, 'x0': x0, 'delta_s': delta_s}
    
# Data simulation parameters
T=10

# -----------------------------------------------------------------------------------------
# ----------------------- MvOrnsteinUhlenbeck + GaussianCDSSM -----------------------------

# Choose a model SDE:
sde_cls = MvOrnsteinUhlenbeck

# SDE parameters
dimX = 2; 

# phi = np.linalg.cholesky(generate_spd_matrix(dimX))
rho = 0.2*np.ones((1, dimX))
phi = nla.cholesky(np.array([[1., 0.9,], [0.9, 1.]]))
phi = 0.3 * phi # Same correlation structure, but smaller variance

sde_params = {'dimX': dimX,
        'rho': 0.2*np.ones((1, dimX)),
        'mu': np.zeros((1, dimX)),
        'phi': phi,
        }

# Choose a CDSSM:
cdssm_cls = GaussianCDSSM

# We can now define a CDSSM: we observe the latent SDE process at discrete points in time with additive noise:

# CDSSM parameters
x0 = np.array([0.]*dimX).reshape(1, -1) if dimX > 1 else 0.; delta_s = 1.; 
dimY = dimX; eta_sq = 0.01 ** 2

L = np.eye(dimY) if (dimY > 1 or dimX > 1) else 1. 
CovY = np.eye(dimY)*eta_sq if dimY > 1 else eta_sq
obs_params = {'L': L, 'covY': CovY, 'x0': x0, 'delta_s': delta_s}
    
# Data simulation parameters
T=100


# -----------------------------------------------------------------------------------------
# ----------------------- OrnsteinUhlenbeck + GaussianCDSSM -------------------------------

# Choose a model SDE:
sde_cls = OrnsteinUhlenbeck

# SDE parameters
dimX = 1; 

# phi = np.linalg.cholesky(generate_spd_matrix(dimX))
rho = 0.2
phi = 0.3

sde_params = {
        'rho': rho,
        'mu': 0.,
        'phi': phi,
        }

# Choose a CDSSM:
cdssm_cls = GaussianCDSSM

# We can now define a CDSSM: we observe the latent SDE process at discrete points in time with additive noise:
# CDSSM parameters
x0 = np.array([0.]*dimX).reshape(1, -1) if dimX > 1 else 0.; delta_s = 1.; 
dimY = dimX; eta_sq = 0.01 ** 2

L = np.eye(dimY) if (dimY > 1 or dimX > 1) else 1. 
CovY = np.eye(dimY)*eta_sq if dimY > 1 else eta_sq
obs_params = {'L': L, 'covY': CovY, 'x0': x0, 'delta_s': delta_s}
    
# Data simulation parameters
T=100

# -----------------------------------------------------------------------------------------
# ----------------------- FitzhughNagumo + GaussianCDSSM ----------------------------------
sde_cls = FitzHughNagumo

dimX = FitzHughNagumo.dimX

# Parameterised as in VanDerMeulen & Schauer (2017)
sde_params = {'rho': np.array([1.4, 1.5, 10.]),
            'phi': np.array([0.25, 0.2])
            }

# Choose a CDSSM:
cdssm_cls = GaussianCDSSM

# We can now define a CDSSM: we observe the latent SDE process at discrete points in time with additive noise:
# CDSSM parameters
x0 = np.array([0.]*dimX).reshape(1, -1) if dimX > 1 else 0.; delta_s = 1.; 
dimY = dimX; eta_sq = 0.01 ** 2

L = np.eye(dimY) if (dimY > 1 or dimX > 1) else 1. 
CovY = np.eye(dimY)*eta_sq if dimY > 1 else eta_sq
obs_params = {'L': L, 'covY': CovY, 'x0': x0, 'delta_s': delta_s}
    
# Data simulation parameters
T=20

# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------