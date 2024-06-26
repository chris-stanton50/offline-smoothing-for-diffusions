"""
Feynman Kac module: 

In this module, we construct the Feynman Kac models for continuous-discrete state space models, which include reparameterisations.
"""

from particles import FeynmanKac
import scipy.stats as stats
from sdes.sdes import OrnsteinUhlenbeck
from sdes.auxiliary_bridges import LocalLinearOUProp
import numpy as np

class BootstrapDA(FeynmanKac):
    """
    Basically the same as the standard Bootstrap PF.
    """
    Delta_s = 1. # Size of imputation
    ModelSDECls = OrnsteinUhlenbeck
    
    def __init__(self, data, num, eta, **model_kwargs):
        self.data = data
        self.T = len(data)
        self.model_sde = OrnsteinUhlenbeck(**model_kwargs)
        self.num = num # Number of imputed points
        self.eta = eta # Noise parameter
        
    def M0(self, N):
        return self.model_sde.simulate(N, 0., t_start=0., t_end=1., num=self.num, milstein=False)

    def M(self, t, xp):
        return self.model_sde.simulate(xp.shape[0], xp['1.0'], t_start=(t)*self.Delta_s, t_end=(t+1)*self.Delta_s, num=self.num, milstein=False)

    def logG(self, t, xp, x):
        return stats.norm.logpdf(self.data[t], loc=x['1.0'], scale=self.eta)

class ForwardGuidedDA(FeynmanKac):
    """
    Guided Forward Proposals
    """
    Delta_s = 1. # Size of imputation
    x_0 = 0. # Starting point
    ModelSDECls = OrnsteinUhlenbeck
    ProposalSDECls = LocalLinearOUProp
    
    def __init__(self, data, num, eta, **kwargs):
        self.data = data
        self.T = len(data)
        self.model_sde = OrnsteinUhlenbeck(**kwargs)
        self.num = num # Number of imputed points
        self.eta = eta # Noise parameter

    def M0(self, N):                            
        self.proposal_sde = self.ProposalSDECls(self.model_sde, self.x_0, self.Delta_s, self.data[0], self.eta ** 2)
        return self.proposal_sde.simulate(N, 0., num=self.num)
        
    def M(self, t, xp):
        self.proposal_sde = self.ProposalSDECls(self.model_sde, t*self.Delta_s, (t+1)*self.Delta_s, self.data[t], self.eta ** 2)
        return self.proposal_sde.simulate(xp.shape[0], xp['1.0'], num=self.num)

    def logG(self, t, xp, x):
        self.proposal_sde = self.ProposalSDECls(self.model_sde, t*self.Delta_s, (t+1)*self.Delta_s, self.data[t], self.eta ** 2)        
        if t == 0:
            N = x['1.0'].shape[0] 
            x_start = np.zeros(N)
            self.proposal_sde.build_linear_sde(x_start=0.) # Choice of starting pt does not change the params for OU process.
        else:
            x_start = xp['1.0']
            self.proposal_sde.build_linear_sde(x_start=0.) # Choice of starting pt does not change the params for OU process.
        return stats.norm.logpdf(self.data[t], loc=x['1.0'], scale=self.eta) + self.proposal_sde.log_girsanov(x_start, x)
